"""T1 record builder: AccessGraph → TwoCopySymbolicHBSolver records.

The Track 2 (global-memory) IR front-end of the hybrid race detector. It
lowers the shared TTIR reader's :class:`AccessGraph` — under the CONCRETE
scalar params and tensor base pointers of a real launch (tier T1) — into the
exact record shape the dynamic mode feeds the solver, so the solver is
reused verbatim ("same encoder, two capture front-ends"):

  * ``addr_expr`` is an absolute BYTE address: ``data_ptr + offset * elem``.
  * Program ids are the shared ``SymbolicExpr.PID0/1/2`` consts, which the
    solver alpha-renames into its two copies.
  * Each (make_range, dim) instance interns one summary variable in an
    ``ARANGE_DICT``-shaped registry; the solver rebuilds the range
    constraints from the registry keys.
  * The scf.for iteration is ONE symbolic index in ``copy_local_vars``
    (each copy gets its own iteration) with its range in ``premises``.
  * ``mask ∧ path`` land in ``active``; an atomic RMW is a single record
    with ``reads = writes = active`` (the solver's lowering rule).

Uncertainty discipline (mirrors ``oob.check_graph``): records built from a
``mask_dropped`` or ``guarded`` access are over-approximations — UNSAT over
them still proves race-freedom, but a SAT touching one must never be
reported as a definite race. Their event ids are returned in
``uncertain_event_ids`` and the client downgrades such reports.

Model boundary — the IN-BOUNDS premise: every record carries its tensor's
allocation bounds (``base ≤ addr < base + numel·elem``) as constraints.
With an unbounded symbolic grid, offsets would otherwise stray
arithmetically into OTHER tensors' address ranges and fabricate
cross-tensor races no launch can produce; real aliasing (two args sharing
storage) still surfaces because the bounds are the launch's actual
intervals. The flip side: a race REACHABLE ONLY through an out-of-bounds
access is out of scope here — that access is the compiled sanitizer's OOB
verdict, which proves exactly the premise this track assumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..data import AtomicKind, MemorySem

from z3 import And, Array, If, IntSort, IntVal, Or, Select, simplify
from z3 import Not as Z3Not

from ....core.data import AtomicCas, AtomicRMW, Load, Store
from ...common.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    IterArgOffset,
    Loaded,
    LoopVar,
    Not,
    NumPrograms,
    Observed,
    Param,
    Pid,
    Term,
    UnsupportedTTIR,
    loaded_leaves,
    mentions_loaded,
    observed_indices,
)
from ...common.ttir_reader import Select as TSelect

_KNOWN_SEMS = ("relaxed", "acquire", "release", "acq_rel")

# TTIR printer spellings → the solver's canonical RMW op names.
_RMW_OP_ALIASES = {"exch": "xchg"}


def _normalize_rmw_op(op: str | None) -> str | None:
    if not op:
        return None
    op = op.lower()
    return _RMW_OP_ALIASES.get(op, op)


@dataclass(frozen=True)
class GlobalTensor:
    """Launch-time facts about one pointer argument."""

    data_ptr: int
    elem_size: int  # bytes
    numel: int
    # The in-bounds premise equates the allocation extent with numel·elem,
    # which UNDERSTATES a strided view's footprint (legal accesses past
    # numel would be deactivated — a false proof). Non-contiguous tensors
    # therefore fail closed.
    contiguous: bool = True
    # PRE-LAUNCH element values for small integer tensors (spec part B):
    # captured at pre_warmup — before the real kernel mutates the storage —
    # so the solver's rf-init machinery and counting axiom see launch-time
    # initial values. None when uncaptured (float dtype, too large, or a
    # non-contiguous view): the solver then falls back to rf_unknown /
    # omits the counting axiom, the over-report direction.
    init_values: tuple[int, ...] | None = None
    # Route 2 (L2 only): the PRE-LAUNCH contents of an integer tensor up
    # to the address-snapshot bound, the source of every Loaded term's
    # value; ``snapshot_reason`` names why it is absent (float dtype,
    # too large, non-contiguous), so the refusal can say so.
    snapshot: tuple[int, ...] | None = None
    snapshot_reason: str = ""


class _InitValueTensor:
    """Duck-typed stand-in satisfying exactly the tensor surface
    ``_initial_atomic_source`` / ``_initial_value_at`` touch: the ORIGINAL
    launch base address with the PRE-LAUNCH values (finalize runs after the
    real kernel already mutated the original tensors, so the live objects
    must not be read)."""

    def __init__(self, meta: GlobalTensor) -> None:
        self._meta = meta

    def data_ptr(self) -> int:
        return self._meta.data_ptr

    def element_size(self) -> int:
        return self._meta.elem_size

    def numel(self) -> int:
        return self._meta.numel

    def is_contiguous(self) -> bool:
        return self._meta.contiguous

    def reshape(self, *_shape: Any) -> "_InitValueTensor":
        return self

    def tolist(self) -> list[int]:
        assert self._meta.init_values is not None
        return list(self._meta.init_values)


@dataclass
class GlobalEncoding:
    records: list[Any]
    arange_dict: dict[Any, Any]
    # event_ids of records built from over-approximated accesses
    # (mask_dropped / guarded): SAT reports touching them are not witnesses.
    uncertain_event_ids: set[int] = field(default_factory=set)
    # True when an await record's exit predicate is asserted (spec C1.2):
    # the verdict is then CONDITIONAL ON TERMINATION of the spin loop —
    # surfaced in the client's provenance as "+assumes-termination".
    assumes_termination: bool = False
    # True when the graph carries any atomic access. The used_pid_axes
    # pinning rule's justification — blocks differing only in an UNREAD
    # axis behave identically — FAILS for atomics: interleaving feeds back
    # into observations, so two no-pid blocks doing atomic_add are NOT
    # identical. symbolic_grid therefore sizes unread axes from the REAL
    # launch (not 1) for atomic-bearing graphs.
    has_atomics: bool = False
    # pid axes with a parsed tt.get_program_id (AccessGraph.pid_axes — the
    # PARSE-time set, never the axes that merely survive into modeled
    # terms: a pid read into a stored value, a dropped mask or an unmodeled
    # condition still distinguishes the blocks' behavior, and pinning such
    # an axis fabricated race-freedom proofs). The T1 grid is symbolic ONLY
    # along these; truly unread axes are pinned to 1 — otherwise every 1-D
    # kernel would "race" under a 2-D grid it never reads (identical
    # addresses from blocks differing only in an ignored axis: a
    # launch-contract violation, not a kernel bug). The claim: "race-free
    # for every grid along the axes the kernel reads".
    used_pid_axes: set[int] = field(default_factory=set)
    # Route 2: the snapshot equalities (``snap_<tensor>[i] == v``) the
    # solver asserts in its base; empty when no Loaded term was encoded.
    assumptions: tuple[Any, ...] = ()
    # True when any record's terms went through a snapshot Select: the
    # proof holds for this launch's tensor CONTENTS (content-qualified).
    content_qualified: bool = False


@dataclass
class _LoopBinding:
    """One scf.for's symbolic iteration: the index var k, its existence
    premise, the zero-trip flag (concrete bounds only), and the induction
    value's lower / step (induction = lower + k·step)."""

    var: Any
    premises: tuple[Any, ...]
    zero_trip: bool
    lower: Any
    step: int


def _graph_loops(graph: AccessGraph) -> list:
    """Every loop of the graph, outer before inner. Graphs built before
    multi-loop capture (hand-built fixtures) carry only ``loop``."""
    if graph.loops:
        return list(graph.loops)
    return [graph.loop] if graph.loop is not None else []


class _RaceEnv:
    """Term → Z3 in the solver's vocabulary (shared pid consts, interned
    arange summary vars, one symbolic loop index PER LOOP).

    ``symbolic_params=True`` is the T0 mode: scalar params become shared
    free Ints (NOT copy-local — both program copies live in one launch, so
    they see the same parameter values). Loop bounds that reference a param
    then fail to concretize and raise, which the tier selector catches to
    fall back to T1.

    ``multipath=True`` (Route 3, L2) additionally keeps a T1 loop bound
    SYMBOLIC when it is linear in the pid / iterator symbols after the
    params are pinned (the persistent grid-stride shape
    ``range(pid, M, NUM_PRGMS)``): the same iteration-existence premise T0
    uses, instead of the "not concrete at launch" refusal."""

    def __init__(
        self,
        graph: AccessGraph,
        params: dict[str, int],
        *,
        symbolic_params: bool = False,
        multipath: bool = False,
        tensors: "dict[str, GlobalTensor] | None" = None,
    ) -> None:
        from ...symbolic_engine import SymbolicExpr

        self._pids = (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)
        self.graph = graph
        self.params = params
        self.symbolic_params = symbolic_params
        # Route 2 state: per-tensor snapshot arrays and their equalities,
        # per-load free "padding" arrays (copy-local: masked-off lanes hold
        # an unspecified value that may differ between instances), the
        # loads whose value had to stay FREE (no usable snapshot: T0, a
        # float/large/non-contiguous source, or a source this kernel
        # writes) and the loops whose bounds depend on such a free value.
        self.tensors = tensors or {}
        self._snap_arrays: dict[str, Any] = {}
        self.snapshot_assumptions: list[Any] = []
        self._pad_arrays: dict[int, Any] = {}
        self.pad_vars: tuple[Any, ...] = ()
        self.used_snapshot = False
        self.unusable_sources: dict[str, str] = {}
        self.free_loaded: set[int] = set()
        self.free_reason: dict[int, str] = {}
        self.free_bound_loops: set[str] = set()
        self._param_vars: dict[str, Any] = {}
        self.arange_dict: dict[Any, Any] = {}
        self._arange_vars: dict[tuple[str, int], Any] = {}
        # One observation var per atomic access index (spec part B). An
        # index lands in modeled_obs when its record carries the var as
        # old_value (rf-justified); Observed leaves of UNMODELED indices
        # are free symbols — proof-only, and rejected in address position.
        self._observed_vars: dict[int, Any] = {}
        self.modeled_obs: set[int] = set()
        self.multipath = multipath
        # The FIRST loop's binding under the historical names (the single-
        # loop consumers and tests read these); every loop in ``_loops``.
        self.loop_var: Any = None  # the symbolic iteration INDEX k
        self.loop_premises: tuple[Any, ...] = ()
        self.zero_trip = False
        # Induction value = _loop_lower (Z3 expr) + k * _loop_step (int).
        self._loop_lower: Any = None
        self._loop_step: int = 1
        self._loops: dict[str, _LoopBinding] = {}
        self.loop_vars: tuple[Any, ...] = ()
        for index, lp in enumerate(_graph_loops(graph)):
            self._bind_loop(lp, index)

    # ── loop ─────────────────────────────────────────────────────────
    @staticmethod
    def _as_long(v: Any) -> int | None:
        try:
            return simplify(v).as_long()
        except Exception:
            return None

    def _binding(self, loop_ssa: str) -> _LoopBinding:
        b = self._loops.get(loop_ssa)
        if b is None:
            if len(self._loops) == 1:
                # Pre-multipath graphs name their single loop loosely.
                return next(iter(self._loops.values()))
            raise UnsupportedTTIR(f"unbound loop {loop_ssa!r}")
        return b

    def premises_for(self, access: AccessEvent) -> tuple[Any, ...]:
        """The iteration-existence premises of the access's enclosing
        loops (outer first); ``()`` outside every loop."""
        if access.loops:
            return tuple(p for ssa in access.loops for p in self._binding(ssa).premises)
        return self.loop_premises if access.in_loop else ()

    def zero_trip_for(self, access: AccessEvent) -> bool:
        """True when some enclosing loop has a concrete trip count of zero:
        the access never executes on this launch."""
        if access.loops:
            return any(self._binding(ssa).zero_trip for ssa in access.loops)
        return self.zero_trip if access.in_loop else False

    def _bind_loop(self, loop: Any, index: int) -> None:
        from z3 import Int

        assert loop is not None
        free_before = set(self.free_loaded)
        lower_z3 = self.eval(loop.lower)
        upper_z3 = self.eval(loop.upper)
        step_z3 = self.eval(loop.step)
        if self.free_loaded - free_before:
            # a bound over an unmodeled loaded value: the iteration range
            # is over-approximated, every access of the loop is widened
            self.free_bound_loops.add(loop.loop_ssa)
        lower_c = self._as_long(lower_z3)
        upper_c = self._as_long(upper_z3)
        step_c = self._as_long(step_z3)

        # The step must be a concrete positive constant in BOTH modes:
        # symbolic k·step is the nonlinear Z3-unknown bait the linearity
        # gate exists to keep out (real kernels' steps are constexpr
        # blocks, folded to constants in TTIR), and MLIR scf.for requires
        # a positive step (a violating launch is UB, outside every claim).
        if step_c is None:
            raise UnsupportedTTIR(
                "loop step is not a compile-time constant (symbolic k·step "
                "is nonlinear; T0 falls back per the ladder)"
            )
        if step_c <= 0:
            raise UnsupportedTTIR(f"loop step {step_c} <= 0 (descending unsupported)")

        var = Int("ttir_loop_k" if index == 0 else f"ttir_loop_k{index}")
        zero_trip = False
        if lower_c is not None and upper_c is not None:
            n_iters = max(0, (upper_c - lower_c + step_c - 1) // step_c)
            # A zero-trip loop has NO footprint: in-loop accesses are
            # skipped entirely (encode_graph). The premise must stay the
            # exact range — fabricating an iteration (max(1, n)) produced
            # definite race reports for launches that never run the body.
            zero_trip = n_iters == 0
            premises: tuple[Any, ...] = (And(var >= 0, var < n_iters),)
        else:
            # Route 3 (multipath, L2): a T1 bound that stays symbolic after
            # the params are pinned is a pid- or iterator-dependent bound
            # (``range(pid, M, NUM_PRGMS)``, a triangular inner loop). When
            # it is linear in those symbols it takes the T0 existence
            # premise below instead of the refusal; nonlinear bounds keep
            # refusing (the same Z3-unknown bait the linearity gate blocks).
            pid_linear = self.multipath and all(
                _linear_in(t, self.graph, _T1_SYMBOLIC_LEAVES)
                for t in (loop.lower, loop.upper)
            )
            if not self.symbolic_params and not pid_linear:
                what = "lower bound" if lower_c is None else "upper bound"
                raise UnsupportedTTIR(
                    f"loop {what} is not concrete at launch (T1 needs "
                    "concrete scalar params in loop bounds)"
                )
            # T0 SYMBOLIC LOOP BOUNDS (the S5 stretch): instead of a
            # concrete trip count, the k-th iteration EXISTS iff its
            # induction value stays below the (symbolic) upper bound:
            #   k >= 0  ∧  lower + k·step < upper
            # Linear (step is a constant), and it subsumes the zero-trip
            # rule: upper <= lower makes the premise UNSAT, so in-loop
            # events are inactive — no phantom footprint to skip.
            premises = (
                And(
                    var >= 0,
                    lower_z3 + var * IntVal(step_c) < upper_z3,
                ),
            )
        binding = _LoopBinding(var, premises, zero_trip, lower_z3, step_c)
        self._loops[loop.loop_ssa] = binding
        self.loop_vars = self.loop_vars + (var,)
        if index == 0:
            self.loop_var = var
            self.loop_premises = premises
            self.zero_trip = zero_trip
            self._loop_lower = lower_z3
            self._loop_step = step_c

    # ── leaves ───────────────────────────────────────────────────────
    def observed(self, access_index: int) -> Any:
        from z3 import Int

        var = self._observed_vars.get(access_index)
        if var is None:
            var = Int(f"ttir_obs_{access_index}")
            self._observed_vars[access_index] = var
        return var

    def _arange(self, ar: Arange) -> Any:
        from z3 import Int

        key = (ar.ssa, ar.dim)
        var = self._arange_vars.get(key)
        if var is None:
            clean = ar.ssa.strip("%").replace("#", "_")
            var = Int(f"arange_{ar.start}_{ar.end}_ttir_{clean}_d{ar.dim}")
            self._arange_vars[key] = var
            # ARANGE_DICT shape: key[0]/key[1] carry the range the solver
            # rebuilds per copy; the trailing components keep instances
            # distinct. value[0] is the original var to substitute.
            self.arange_dict[(ar.start, ar.end, "ttir", ar.ssa, ar.dim)] = (var, None)
        return var

    # ── the evaluator ────────────────────────────────────────────────
    def eval(self, term: Term) -> Any:
        if isinstance(term, Const):
            return IntVal(term.value)
        if isinstance(term, Param):
            if self.symbolic_params:
                from z3 import Int

                var = self._param_vars.get(term.name)
                if var is None:
                    var = Int(f"ttir_param_{term.name}")
                    self._param_vars[term.name] = var
                return var
            if term.name not in self.params:
                raise UnsupportedTTIR(
                    f"scalar param {term.name!r} not captured at launch"
                )
            return IntVal(self.params[term.name])
        if isinstance(term, Pid):
            return self._pids[term.axis]
        if isinstance(term, NumPrograms):
            from z3 import Int

            # The SAME grid var symbolic_grid() interns by name (the reader
            # put the axis in pid_axes, so the dim is never pinned to 1);
            # the solver bounds it with pid < grid and grid >= 1. This is
            # what lets a last-block gate `o == num_programs(0) - 1` prove
            # for EVERY grid instead of only the launch's.
            return Int(f"grid_{term.axis}")
        if isinstance(term, Arange):
            return self._arange(term)
        if isinstance(term, LoopVar):
            b = self._binding(term.loop_ssa)
            return b.lower + b.var * IntVal(b.step)
        if isinstance(term, IterArgOffset):
            info = self.graph.iter_args[term.arg_id]
            b = self._binding(info.loop_ssa) if info.loop_ssa else self._binding("")
            return self.eval(info.offset0) + b.var * self.eval(info.delta)
        if isinstance(term, Bin):
            a, b = self.eval(term.a), self.eval(term.b)
            if term.op == "+":
                return a + b
            if term.op == "-":
                return a - b
            if term.op == "*":
                return a * b
            if term.op == "//":
                return _trunc_div(a, b)
            if term.op == "%":
                return a - b * _trunc_div(a, b)
            if term.op == "min":
                return If(a <= b, a, b)
            if term.op == "max":
                return If(a >= b, a, b)
            raise UnsupportedTTIR(f"unknown arith op {term.op}")
        if isinstance(term, Cmp):
            a, b = self.eval(term.a), self.eval(term.b)
            table = {
                "slt": lambda: a < b, "sle": lambda: a <= b,
                "sgt": lambda: a > b, "sge": lambda: a >= b,
                "eq": lambda: a == b, "ne": lambda: a != b,
            }  # fmt: skip
            if term.pred not in table:
                raise UnsupportedTTIR(f"unknown cmp predicate {term.pred}")
            return table[term.pred]()
        if isinstance(term, BoolBin):
            a, b = _as_bool(self.eval(term.a)), _as_bool(self.eval(term.b))
            return And(a, b) if term.op == "and" else Or(a, b)
        if isinstance(term, TSelect):
            return If(
                _as_bool(self.eval(term.cond)), self.eval(term.t), self.eval(term.f)
            )
        if isinstance(term, Not):
            return Z3Not(_as_bool(self.eval(term.a)))
        if isinstance(term, Observed):
            return self.observed(term.access_index)
        if isinstance(term, Loaded):
            return self._loaded(term)
        if isinstance(term, DataDep):
            raise UnsupportedTTIR(f"data-dependent term ({term.why})")
        raise UnsupportedTTIR(f"unhandled term {type(term).__name__}")

    # ── Route 2: the snapshot Select ─────────────────────────────────
    def _snapshot_array(self, base: str) -> tuple[Any, int] | None:
        """The source tensor's snapshot array, or None with the reason
        recorded when the value must stay free (T0, no metadata, no
        snapshot, or a source this kernel writes)."""
        arr = self._snap_arrays.get(base)
        meta = self.tensors.get(base)
        if arr is not None:
            assert meta is not None
            return arr, meta.numel
        why: str | None = None
        if self.symbolic_params:
            why = "no launch at T0"
        elif meta is None:
            why = "not captured"
        elif base in self.unusable_sources:
            why = self.unusable_sources[base]
        elif meta.snapshot is None:
            why = meta.snapshot_reason or "no snapshot"
        if why is not None:
            self.unusable_sources.setdefault(base, why)
            return None
        assert meta is not None and meta.snapshot is not None
        arr = Array(f"snap_{base}", IntSort(), IntSort())
        for i, v in enumerate(meta.snapshot):
            self.snapshot_assumptions.append(Select(arr, IntVal(i)) == IntVal(int(v)))
        self._snap_arrays[base] = arr
        return arr, meta.numel

    def _pad(self, index: int) -> Any:
        pad = self._pad_arrays.get(index)
        if pad is None:
            pad = Array(f"pad_{index}", IntSort(), IntSort())
            self._pad_arrays[index] = pad
            self.pad_vars = self.pad_vars + (pad,)
        return pad

    def _loaded(self, term: Loaded) -> Any:
        """``If(mask ∧ in-domain, snap[off], other-or-free)``: on an active
        in-bounds lane the value is the snapshot element; a masked-off lane
        holds ``other`` when the load names one and an unspecified value
        otherwise (a free array, so no two lanes or instances are forced to
        agree); an out-of-domain offset is unspecified too (the load's own
        in-bounds premise excludes it on active lanes). Without a usable
        snapshot the whole value is free: the widening Route 3 applied to
        unmodeled loaded values, and the record is marked uncertain by the
        caller (an address built on it refuses instead, see _record_for)."""
        off = self.eval(term.offset)
        pad = self._pad(term.access_index)
        snap = self._snapshot_array(term.base_param)
        if snap is None:
            self.free_loaded.add(term.access_index)
            self.free_reason[term.access_index] = self.unusable_sources.get(
                term.base_param, "no snapshot"
            )
            return Select(pad, off)
        arr, numel = snap
        self.used_snapshot = True
        in_dom = And(off >= 0, off < IntVal(numel))
        value = If(in_dom, Select(arr, off), Select(pad, off))
        if term.mask is None:
            return value
        mask = _as_bool(self.eval(term.mask))
        other = self.eval(term.other) if term.other is not None else Select(pad, off)
        return If(mask, value, other)


def _as_bool(e: Any) -> Any:
    from z3 import is_bool

    return e if is_bool(e) else e != 0


def _trunc_div(a: Any, b: Any) -> Any:
    """arith.divsi truncates toward zero; Z3 Int division is Euclidean."""
    aa = If(a >= 0, a, -a)
    ab = If(b >= 0, b, -b)
    q = aa / ab
    return If((a >= 0) == (b >= 0), q, -q)


def _await_premises(
    graph: AccessGraph, env: _RaceEnv
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """The await abstraction's termination premises (spec C1.2), attached
    to EVERY record of the encoding.

    For each recognized spin loop, "reach(await) → o == exit-value" is an
    EXECUTION-LEVEL invariant of any terminating run (the final iteration's
    read observed the exit value). It must hold in every record's activity
    — asserting it only on the awaited event would let a model set
    o ≠ expected, deactivate the await, and dissolve the po→sw→po bridge
    while the post-loop accesses stay active (a SAT escape adversarial
    testing caught). Guarded awaits (unmodeled enclosing condition) emit NO
    premise: asserting their exit for instances that never reach the loop
    could over-constrain — omission is the over-report direction.

    Returns (premises, observation vars) — the vars ride copy_local_vars of
    every record so each program copy gets its own observation."""
    from z3 import Implies

    premises: list[Any] = []
    obs_vars: list[Any] = []
    for seq, access in enumerate(graph.accesses):
        if not access.awaited or access.exit_pred is None or access.guarded:
            continue
        exit_z3 = _as_bool(env.eval(access.exit_pred))
        guard: list[Any] = []
        if access.mask is not None:
            guard.append(_as_bool(env.eval(access.mask)))
        if access.path is not None:
            guard.append(_as_bool(env.eval(access.path)))
        premises.append(Implies(And(*guard), exit_z3) if guard else exit_z3)
        obs_vars.append(env.observed(seq))
    return tuple(premises), tuple(obs_vars)


def _record_for(
    access: AccessEvent,
    seq: int,
    env: _RaceEnv,
    kernel_name: str,
    meta: GlobalTensor | None,
    await_premises: tuple[Any, ...] = (),
    await_obs: tuple[Any, ...] = (),
) -> Any:
    """One solver record. ``meta`` present = T1 (real base address and the
    in-bounds premise); ``meta=None`` = T0, where addresses are byte offsets
    from the tensor's own base and conflicts are confined to that tensor's
    group by construction (see encode_graph_t0). ``await_premises`` /
    ``await_obs`` are the termination invariants of the graph's spin loops
    (see _await_premises) — conjoined into every record."""
    from ..data import AccessEventRecord

    elem = access.elem_bits // 8
    if elem <= 0:
        raise UnsupportedTTIR(
            f"unknown element width for {access.base_param!r} "
            f"(elem_bits={access.elem_bits})"
        )
    # Spec part B: the RMW observation is modeled for an integer-typed,
    # non-loop atomic (one observation var cannot stand for one-per-
    # iteration values; loops stay footprint-only). MUST happen before any
    # term evaluation below so downstream Observed uses of THIS access see
    # it as modeled.
    old_value: Any = None
    rmw_op: str | None = None
    rmw_operand: Any = None
    cas_cmp: Any = None
    cas_new: Any = None
    if access.kind == "atomic_rmw" and not access.elem_float and not access.in_loop:
        old_value = env.observed(seq)
        env.modeled_obs.add(seq)
        assert access.atomic is not None
        rmw_op = _normalize_rmw_op(access.atomic.rmw_op)
        if access.atomic_val is not None:
            try:
                rmw_operand = env.eval(access.atomic_val)
            except UnsupportedTTIR:
                rmw_operand = None  # unmodelable operand: write stays open
    elif access.kind == "atomic_cas":
        # Only the AWAITED CAS reaches here (encode_graph refuses the
        # rest); the solver's CAS lowering needs all three value pieces.
        if access.in_loop:
            raise UnsupportedTTIR(
                f"line {access.line_no}: awaited CAS inside scf.for "
                "(one observation cannot stand for one per iteration)",
                kind="control-flow",
            )
        if access.elem_float:
            raise UnsupportedTTIR(
                f"line {access.line_no}: float-typed CAS is outside the "
                "integer value model",
                kind="spin-shape",
            )
        if access.atomic_cmp is None or access.atomic_val is None:
            raise UnsupportedTTIR(
                f"line {access.line_no}: CAS cmp/val operands are not " "modelable",
                kind="spin-shape",
            )
        old_value = env.observed(seq)
        env.modeled_obs.add(seq)
        cas_cmp = env.eval(access.atomic_cmp)
        cas_new = env.eval(access.atomic_val)

    # An address may reference an observation only when that observation is
    # value-modeled (the solver then requires its counting axiom, B.1.5);
    # a free observation in an address would alias everything.
    unmodeled_in_addr = {
        i for i in observed_indices(access.offset) if i not in env.modeled_obs
    }
    if unmodeled_in_addr:
        raise UnsupportedTTIR(
            f"line {access.line_no}: address depends on an atomic "
            "observation that is not value-modeled (float-typed or "
            "loop-carried atomic)",
            kind="indirect-address",
        )
    free_before = set(env.free_loaded)
    addr_off = env.eval(access.offset)
    new_free = env.free_loaded - free_before
    if new_free:
        idx = min(new_free)
        why = env.free_reason.get(idx, "no snapshot")
        raise UnsupportedTTIR(
            f"line {access.line_no}: address depends on a loaded value with no "
            f"usable snapshot ({why})",
            kind="snapshot-bound"
            if why.startswith("too large")
            else "indirect-address",
        )
    bounds: tuple[Any, ...]
    if meta is not None:
        if meta.elem_size != elem:
            raise UnsupportedTTIR(
                f"element width mismatch for {access.base_param!r}: TTIR says "
                f"{elem} bytes, the launch tensor says {meta.elem_size}"
            )
        addr = IntVal(meta.data_ptr) + addr_off * IntVal(elem)
        # The in-bounds premise (see the module docstring's model boundary).
        bounds = (
            addr >= IntVal(meta.data_ptr),
            addr < IntVal(meta.data_ptr + meta.numel * meta.elem_size),
        )
    else:
        addr = addr_off * IntVal(elem)
        bounds = ()

    active: Any = True
    if access.mask is not None:
        active = _as_bool(env.eval(access.mask))
    if access.path is not None:
        path_z3 = _as_bool(env.eval(access.path))
        active = path_z3 if active is True else And(active, path_z3)
    access_mode: Literal["read", "write"]
    atomic_kind: "AtomicKind"
    sem: "MemorySem"
    if access.kind == "atomic_rmw":
        assert access.atomic is not None
        sem = (
            access.atomic.sem  # type: ignore[assignment]
            if access.atomic.sem in _KNOWN_SEMS
            else "relaxed"
        )
        op_type: type = AtomicRMW
        is_atomic, atomic_kind = True, "rmw"
        access_mode = "read"
        reads: Any = True
        writes: Any = True
        scope: str | None = access.atomic.scope
    elif access.kind == "atomic_cas":
        assert access.atomic is not None
        sem = (
            access.atomic.sem  # type: ignore[assignment]
            if access.atomic.sem in _KNOWN_SEMS
            else "relaxed"
        )
        op_type = AtomicCas
        is_atomic, atomic_kind = True, "cas"
        access_mode = "read"
        # The solver's CAS lowering recomputes reads/writes/written_value
        # per copy from old/cmp/new (writes fire only on success).
        reads, writes = True, None
        scope = access.atomic.scope
    else:
        sem = "plain"
        op_type = Store if access.kind == "store" else Load
        is_atomic, atomic_kind = False, "none"
        access_mode = "write" if access.kind == "store" else "read"
        reads, writes = None, None
        scope = None

    copy_local: tuple[Any, ...] = tuple(env.loop_vars)
    # Observations are per-program-instance nondeterminism: alpha-renamed
    # per copy exactly like the interpreter track's CAS/RMW return vars.
    # EVERY referenced observation is listed — not just this record's own —
    # because the solver unions copy_local_vars only over the records it is
    # given: a T0 per-tensor group (or a zero-trip-skipped RMW) would
    # otherwise leave a referenced var un-renamed, silently SHARING one
    # observation between the two copies and manufacturing UNSAT (a false
    # proof) for masks like ``o == 0`` vs ``o == 2``.
    ref_obs = observed_indices(access.offset)
    for t in (access.mask, access.path, access.exit_pred):
        if t is not None:
            ref_obs |= observed_indices(t)
    for i in sorted(ref_obs):
        copy_local = copy_local + (env.observed(i),)
    if old_value is not None:
        copy_local = copy_local + (old_value,)
    copy_local = copy_local + tuple(await_obs)
    # Route 2: every free padding array is copy-local (the solver unions
    # copy_local_vars over all records, so listing them here is enough)
    copy_local = copy_local + tuple(env.pad_vars)
    source = (
        (access.loc.file, access.loc.line, kernel_name)
        if access.loc is not None
        else None
    )
    # rf-init needs the pre-launch values at the ORIGINAL base; only an
    # atomic's observation ever consumes them.
    tensor = (
        _InitValueTensor(meta)
        if (old_value is not None and meta is not None and meta.init_values is not None)
        else None
    )

    return AccessEventRecord(
        op_type=op_type,
        access_mode=access_mode,
        tensor=tensor,
        tensor_name=access.base_param,
        addr_expr=addr,
        # The iteration range constrains only the accesses that iterate;
        # the spin-termination invariants constrain every record. Under
        # multipath the iteration ranges ride as LOCAL constraints: they
        # gate the record's activity exactly the same way, but stay out of
        # the solver's Feasible# base, which asserts every record's
        # premises jointly. With one loop that only demanded "some pid
        # runs an iteration"; with several loops whose pid-dependent
        # ranges are disjoint (fla's parallel_simple_gla: one loop runs for
        # pid 0 only, the next for pid >= 1 only) the joint assertion has
        # no model and every proof came back vacuous. A loop that runs
        # zero times for some instance is an execution, not vacuity; the
        # await invariants keep their global role.
        premises=(
            await_premises
            if env.multipath
            else env.premises_for(access) + await_premises
        ),
        local_constraints=(
            bounds + env.premises_for(access) if env.multipath else bounds
        ),
        source_location=source,
        program_seq=seq,
        debug_name=f"{kernel_name}:ttir{access.line_no}:{access.kind}",
        active=active,
        reads=reads,
        writes=writes,
        is_atomic=is_atomic,
        atomic_kind=atomic_kind,
        sem=sem,
        scope=scope,
        old_value=old_value,
        rmw_op=rmw_op,
        rmw_operand=rmw_operand,
        cas_cmp_value=cas_cmp,
        cas_new_value=cas_new,
        event_id=seq,
        elem_size=elem,
        copy_local_vars=copy_local,
    )


def _pre_exit_representative(poll: Any, access: AccessEvent, event_id: int) -> Any:
    """The pre-exit representative of an awaited poll: ONE value-model-free
    record standing for all FAILED iterations of the spin.

    The await abstraction keeps a single poll event whose termination
    premise pins its observation to the exit value; the failed iterations
    are dropped. For accesses that keep the closed world intact (a
    value-modeled weak atomic write, a plain read of the awaited location)
    that collapse silently LOSES races that exist only on the dropped
    iterations. The representative restores them: it mirrors the poll's
    footprint, activity, sem/scope and program_seq (equal seq = mutually
    po-unordered with the poll, po-ordered against everything else exactly
    like the poll), under a fresh event_id and with NO value model — rf
    sources need ``written_value``, readers need ``old_value``, so no
    rf/sw edge can touch it and the publisher's sw edge still targets the
    poll only: the failed iterations' unorderedness is preserved. Any
    unrolled race on a failed iteration maps to a race on the rep with the
    same footprint and modes and no more ordering (over-report direction);
    rep-vs-atomic pairs stay conflict-exempt exactly when the unrolled
    failed iterations are morally strong (exemption parity).
    """
    from dataclasses import replace

    overrides: dict[str, Any]
    if access.kind == "atomic_cas":
        # A FAILED CAS reads but writes nothing: the rep is a read-only
        # atomic. atomic_kind "rmw", NOT "cas" — the CAS lowering demands
        # the value triple the rep deliberately lacks — with record-level
        # statically-False writes, which the solver's RMW branch honors
        # for pre_exit records.
        overrides = dict(op_type=AtomicRMW, atomic_kind="rmw", reads=True, writes=False)
    elif access.kind == "atomic_rmw":
        # Identity-RMW poll: every failed iteration RE-WRITES the value it
        # read, so the rep keeps the RMW reads-and-writes footprint (the
        # write half is what catches plain readers of the awaited
        # location).
        overrides = dict(reads=True, writes=True)
    else:
        # Plain-load poll: a non-atomic read; the solver's plain path
        # lowers writes to And(active, False) from access_mode.
        overrides = dict(reads=None, writes=None)
    return replace(
        poll,
        event_id=event_id,
        debug_name=f"{poll.debug_name}:pre-exit",
        pre_exit=True,
        old_value=None,
        rmw_op=None,
        rmw_operand=None,
        cas_cmp_value=None,
        cas_new_value=None,
        **overrides,
    )


def encode_graph(
    graph: AccessGraph,
    params: dict[str, int],
    tensors: dict[str, GlobalTensor],
    *,
    multipath: bool = False,
) -> GlobalEncoding:
    """Lower every global access of ``graph`` into solver records under the
    concrete launch ``params``/``tensors`` (tier T1: pid, grid, arange lanes
    and loop iterations stay symbolic). Raises :class:`UnsupportedTTIR`
    (classified) when the kernel cannot be encoded. ``multipath`` enables
    the L2 pid-linear symbolic T1 bounds (see _RaceEnv)."""
    for access in graph.accesses:
        if access.kind == "atomic_cas" and not access.awaited:
            # A free-standing CAS has no static value model (its cmp/new
            # may be data-dependent and its synchronization shape open-
            # ended). The AWAITED CAS (spec C1) is the exception: the spin
            # contract pins cmp/new/exit, so it lowers to the solver's full
            # CAS machinery. Everything else routes to the interpreter
            # front-end.
            raise UnsupportedTTIR(
                f"line {access.line_no}: atomic_cas synchronization is not "
                "modeled statically",
                kind="cas-synchronization",
            )

    env = _RaceEnv(graph, params, multipath=multipath, tensors=tensors)
    if multipath:
        env.unusable_sources.update(_written_load_sources(graph, tensors))
    await_prems, await_obs = _await_premises(graph, env)
    records = []
    uncertain: set[int] = set()
    # Fresh event ids for pre-exit representatives live ABOVE the dense
    # access-seq range so they can never collide with a poll's seq (the
    # solver dedups reports on event_id and the client splits exact vs
    # widened by it).
    next_rep_id = len(graph.accesses)
    for seq, access in enumerate(graph.accesses):
        if env.zero_trip_for(access):
            # The launch's trip count is zero: these accesses never execute.
            continue
        meta = tensors.get(access.base_param)
        if meta is None:
            # Every access must be modeled or the verdict is a false proof —
            # same fail-closed rule as the compiled sanitizer.
            raise UnsupportedTTIR(
                f"missing tensor metadata for base pointer {access.base_param!r}"
            )
        if not meta.contiguous:
            raise UnsupportedTTIR(
                f"non-contiguous tensor {access.base_param!r}: the in-bounds "
                "premise needs the allocation extent (v1 assumes contiguous)"
            )
        rec = _record_for(
            access, seq, env, graph.kernel_name, meta, await_prems, await_obs
        )
        records.append(rec)
        is_uncertain = (
            access.mask_dropped
            or access.guarded
            or _references_unmodeled_observation(access, env)
            or _widened_by_free_loaded(access, env)
        )
        if is_uncertain:
            uncertain.add(seq)
        if access.awaited:
            records.append(_pre_exit_representative(rec, access, next_rep_id))
            if is_uncertain:
                # Uncertainty inheritance: the client splits exact vs
                # widened reports by event_id — a rep built from an
                # over-approximated poll (guarded / dropped mask) must
                # classify widened too, or its reports would surface as
                # definite races from an over-approximated record.
                uncertain.add(next_rep_id)
            next_rep_id += 1
    return GlobalEncoding(
        records=records,
        arange_dict=env.arange_dict,
        uncertain_event_ids=uncertain,
        used_pid_axes=set(graph.pid_axes),
        assumes_termination=any(a.awaited for a in graph.accesses),
        has_atomics=any(a.kind.startswith("atomic") for a in graph.accesses),
        assumptions=tuple(env.snapshot_assumptions),
        content_qualified=env.used_snapshot,
    )


def _graph_terms(graph: AccessGraph) -> list:
    terms: list = []
    for a in graph.accesses:
        terms.append(a.offset)
        for t in (a.mask, a.path, a.exit_pred, a.atomic_val, a.atomic_cmp):
            if t is not None:
                terms.append(t)
    for lp in _graph_loops(graph):
        terms.extend((lp.lower, lp.upper, lp.step))
    for info in graph.iter_args.values():
        terms.extend((info.offset0, info.delta))
    return terms


def graph_mentions_loaded(graph: AccessGraph) -> bool:
    return any(mentions_loaded(t) for t in _graph_terms(graph))


def _written_load_sources(
    graph: AccessGraph, tensors: dict[str, GlobalTensor]
) -> dict[str, str]:
    """The read-only-source premise (Route 2): a Loaded term's value is
    the PRE-LAUNCH snapshot, which stands for the value the load observes
    only if no instance writes the source before the load. The static
    frontend cannot order instances, so a source that overlaps any tensor
    the kernel writes has NO usable snapshot: its loads stay free (widened
    in mask/path position, refused in address position), the interpreter
    frontend's fail-stop (`_note_load_source_or_raise`) transposed."""
    sources = {lf.base_param for t in _graph_terms(graph) for lf in loaded_leaves(t)}
    if not sources:
        return {}
    written = {a.base_param for a in graph.accesses if a.kind != "load"}

    def interval(name: str) -> tuple[int, int] | None:
        m = tensors.get(name)
        if m is None:
            return None
        return (m.data_ptr, m.data_ptr + m.numel * m.elem_size)

    out: dict[str, str] = {}
    for src in sorted(sources):
        si = interval(src)
        for w in sorted(written):
            wi = interval(w)
            if src == w or (si and wi and max(si[0], wi[0]) < min(si[1], wi[1])):
                out[src] = f"the kernel writes {w!r}, which overlaps the source"
                break
    return out


def _widened_by_free_loaded(access: AccessEvent, env: _RaceEnv) -> bool:
    """A mask, path, or exit predicate built on a loaded value that had no
    usable snapshot is a free boolean: over-approximated activity, so the
    record is uncertain; likewise every access of a loop whose bounds went
    through such a value."""
    for t in (access.mask, access.path, access.exit_pred):
        if t is None:
            continue
        if any(lf.access_index in env.free_loaded for lf in loaded_leaves(t)):
            return True
    loops = access.loops or (
        (env.graph.loop.loop_ssa,) if access.in_loop and env.graph.loop else ()
    )
    return any(lp in env.free_bound_loops for lp in loops)


def _references_unmodeled_observation(access: AccessEvent, env: _RaceEnv) -> bool:
    """A mask/path referencing an observation WITHOUT value modeling (float
    or loop-carried atomic) is a free symbol: UNSAT over it still proves,
    but a SAT model may pick an observation the execution never yields —
    the same uncertainty discipline as ``mask_dropped``."""
    for t in (access.mask, access.path):
        if t is None:
            continue
        if any(i not in env.modeled_obs for i in observed_indices(t)):
            return True
    return False


def symbolic_grid(
    encoding: GlobalEncoding,
    launch_grid: tuple[int, ...] | None = None,
    t0: bool = False,
) -> tuple[Any, Any, Any]:
    """The T0/T1 grid: symbolic (all sizes ≥ 1) along the pid axes the
    kernel reads; along UNREAD axes, the REAL launch extent. The previous
    rule pinned unread axes to 1 under the launch-contract premise ("the
    launch only extends along axes the kernel reads"), but a premise must
    be CHECKED against the launch, not assumed: a launch that parallelizes
    an axis the kernel ignores is exactly the aiter#3091 caller bug (the
    fused caller runs _sum_bitmatrix_rows_fused's unpartitioned stores on
    every pid), and pinning below the real extent fabricated race-freedom
    proofs for it — a no-pid broadcast store at grid (4,) proved clean
    while the interpreter reported the WAW. Flooring unread axes at the
    launch extent keeps every contract-respecting launch unchanged (their
    unread extents are 1) and reports the violation otherwise. The T0
    claim is scoped accordingly: any grid along the read axes, the
    launch's extent along unread ones (a T0 premise the ladder audit
    treats as part of premise compatibility). Atomic-bearing graphs keep
    unread axes SYMBOLIC at T0 (see GlobalEncoding.has_atomics: the
    identical-behavior argument fails outright for atomics, and symbolic
    is the sound direction there; a nonlinear counting product just omits
    the axiom and the kernel falls to T1 per the ladder)."""
    from z3 import Int

    def dim(i: int) -> Any:
        if i in encoding.used_pid_axes:
            return Int(f"grid_{i}")
        if encoding.has_atomics and (t0 or launch_grid is None):
            return Int(f"grid_{i}")
        if launch_grid is not None:
            return int(launch_grid[i]) if i < len(launch_grid) else 1
        return 1

    return (dim(0), dim(1), dim(2))


# ───────────────────── tier selector support (§I.3) ─────────────────────

# Observed counts as symbolic: the observation var is free at T0, so a
# product with another symbol is exactly the Z3-unknown bait the gate
# exists to keep out. NumPrograms is a symbolic grid dim for the same
# reason.
_SYMBOLIC_LEAVES = (
    Pid, Param, Arange, LoopVar, IterArgOffset, Observed, NumPrograms, Loaded,
)  # fmt: skip
# At T1 the params are concrete, so only these leaves stay symbolic.
_T1_SYMBOLIC_LEAVES = (
    Pid, Arange, LoopVar, IterArgOffset, Observed, NumPrograms, Loaded,
)  # fmt: skip


def _has_symbols(term: Term, leaves: tuple = _SYMBOLIC_LEAVES) -> bool:
    if isinstance(term, leaves):
        return True
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None and _has_symbols(sub, leaves):
            return True
    return False


def _has_t0_symbols(term: Term) -> bool:
    return _has_symbols(term, _SYMBOLIC_LEAVES)


def _linear_in(term: Term, graph: AccessGraph, leaves: tuple) -> bool:
    """No symbolic×symbolic product and no symbolic divisor, ``leaves``
    naming the symbolic leaf classes of the tier."""
    if isinstance(term, Bin):
        if term.op == "*":
            if _has_symbols(term.a, leaves) and _has_symbols(term.b, leaves):
                return False
        elif term.op in ("//", "%"):
            if _has_symbols(term.b, leaves):
                return False
        return _linear_in(term.a, graph, leaves) and _linear_in(term.b, graph, leaves)
    if isinstance(term, IterArgOffset):
        info = graph.iter_args.get(term.arg_id)
        if info is None:
            return False
        # Expands to offset0 + k·delta: linear only for a constant delta.
        if _has_symbols(info.delta, leaves):
            return False
        return _linear_in(info.offset0, graph, leaves)
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None and not _linear_in(sub, graph, leaves):
            return False
    return True


def _linear_at_t0(term: Term, graph: AccessGraph) -> bool:
    return _linear_in(term, graph, _SYMBOLIC_LEAVES)


def t0_linearity_gate(graph: AccessGraph) -> bool:
    """The tier selector's cheap syntactic gate: attempt T0 only when every
    address/mask/path term stays LINEAR once the scalar params go symbolic
    (no symbolic×symbolic product, no symbolic divisor — Z3-unknown bait).
    T1, with params concrete, is linear again for the same terms."""
    terms: list[Term] = []
    for a in graph.accesses:
        terms.append(a.offset)
        if a.mask is not None:
            terms.append(a.mask)
        if a.path is not None:
            terms.append(a.path)
        if a.exit_pred is not None:
            terms.append(a.exit_pred)
    return all(_linear_at_t0(t, graph) for t in terms)


def encode_graph_t0(
    graph: AccessGraph, *, multipath: bool = False
) -> list[tuple[str, GlobalEncoding]]:
    """The T0 encoding: scalar params symbolic, one encoding PER TENSOR.

    T0 has no launch, hence no base addresses or extents. The non-aliasing
    premise (distinct pointer arguments are distinct allocations) is
    realized by PARTITIONING: accesses can only conflict within one base
    pointer's group, and addresses are byte offsets from that base.
    Aliased-argument launches sit outside the T0 claim — T1 covers them
    with the real bases. Read-only groups are skipped (read/read cannot
    conflict). SYMBOLIC LOOP BOUNDS are supported (the S5 stretch): a
    param-valued lower/upper becomes the iteration-existence premise
    ``k >= 0 ∧ lower + k·step < upper``, so the T0 claim quantifies over
    every trip count too. Raises UnsupportedTTIR when the kernel cannot be
    encoded at T0 (e.g. a non-constant loop STEP — symbolic k·step is
    nonlinear)."""
    for access in graph.accesses:
        if access.kind == "atomic_cas" and not access.awaited:
            raise UnsupportedTTIR(
                f"line {access.line_no}: atomic_cas synchronization is not "
                "modeled statically",
                kind="cas-synchronization",
            )

    # Route 2 at T0: there is no launch, so every Loaded value stays FREE
    # (the widening Route 3 applied), and an address built on one refuses
    # inside _record_for; the T1 rung is where the snapshot enters.
    env = _RaceEnv(graph, {}, symbolic_params=True, multipath=multipath)
    await_prems, await_obs = _await_premises(graph, env)
    # NO pre-exit representative at T0 — sound for a verified reason: T0
    # has no launch, hence no initial values, so the closed-world escape
    # (rf_unknown) is ALWAYS open for an awaited poll and no await corner
    # can silently prove at T0 (probe-verified: the corner kernels' flag
    # group SATs at T0 and falls to T1, where the rep lives). Trigger that
    # invalidates this argument: T0 gaining initial values
    # (GlobalTensor.init_values exists) — then mirror the rep emission of
    # encode_graph here.
    groups: dict[str, list[tuple[int, AccessEvent]]] = {}
    for seq, access in enumerate(graph.accesses):
        if env.zero_trip_for(access):
            continue
        groups.setdefault(access.base_param, []).append((seq, access))

    out: list[tuple[str, GlobalEncoding]] = []
    for name, items in groups.items():
        if all(a.kind == "load" for _, a in items):
            continue
        records = []
        uncertain: set[int] = set()
        for seq, access in items:
            records.append(
                _record_for(
                    access, seq, env, graph.kernel_name, None, await_prems, await_obs
                )
            )
            if access.mask_dropped or access.guarded:
                uncertain.add(seq)
            if _references_unmodeled_observation(access, env):
                uncertain.add(seq)
            if _widened_by_free_loaded(access, env):
                uncertain.add(seq)
        out.append(
            (
                name,
                GlobalEncoding(
                    records=records,
                    arange_dict=env.arange_dict,
                    uncertain_event_ids=uncertain,
                    used_pid_axes=set(graph.pid_axes),
                    assumes_termination=any(a.awaited for a in graph.accesses),
                    has_atomics=any(
                        a.kind.startswith("atomic") for a in graph.accesses
                    ),
                ),
            )
        )
    return out
