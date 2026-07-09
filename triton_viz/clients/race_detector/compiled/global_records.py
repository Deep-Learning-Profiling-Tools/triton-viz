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

from z3 import And, If, IntVal, Or, simplify
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
    LoopVar,
    Not,
    NumPrograms,
    Observed,
    Param,
    Pid,
    Select,
    Term,
    UnsupportedTTIR,
    observed_indices,
)

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


class _RaceEnv:
    """Term → Z3 in the solver's vocabulary (shared pid consts, interned
    arange summary vars, one symbolic loop index).

    ``symbolic_params=True`` is the T0 mode: scalar params become shared
    free Ints (NOT copy-local — both program copies live in one launch, so
    they see the same parameter values). Loop bounds that reference a param
    then fail to concretize and raise, which the tier selector catches to
    fall back to T1."""

    def __init__(
        self,
        graph: AccessGraph,
        params: dict[str, int],
        *,
        symbolic_params: bool = False,
    ) -> None:
        from ...symbolic_engine import SymbolicExpr

        self._pids = (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)
        self.graph = graph
        self.params = params
        self.symbolic_params = symbolic_params
        self._param_vars: dict[str, Any] = {}
        self.arange_dict: dict[Any, Any] = {}
        self._arange_vars: dict[tuple[str, int], Any] = {}
        # One observation var per atomic access index (spec part B). An
        # index lands in modeled_obs when its record carries the var as
        # old_value (rf-justified); Observed leaves of UNMODELED indices
        # are free symbols — proof-only, and rejected in address position.
        self._observed_vars: dict[int, Any] = {}
        self.modeled_obs: set[int] = set()
        self.loop_var: Any = None  # the symbolic iteration INDEX k
        self.loop_premises: tuple[Any, ...] = ()
        self.zero_trip = False
        self._loop_bounds: tuple[int, int, int] | None = None  # (lower, step, n)
        if graph.loop is not None:
            self._bind_loop()

    # ── loop ─────────────────────────────────────────────────────────
    def _concrete(self, term: Term, what: str) -> int:
        v = simplify(self.eval(term))
        try:
            return v.as_long()
        except Exception:
            raise UnsupportedTTIR(
                f"loop {what} is not concrete at launch (T1 needs concrete "
                "scalar params in loop bounds)"
            )

    def _bind_loop(self) -> None:
        from z3 import Int

        loop = self.graph.loop
        assert loop is not None
        lower = self._concrete(loop.lower, "lower bound")
        upper = self._concrete(loop.upper, "upper bound")
        step = self._concrete(loop.step, "step")
        if step <= 0:
            raise UnsupportedTTIR(f"loop step {step} <= 0 (descending unsupported)")
        n_iters = max(0, (upper - lower + step - 1) // step)
        # A zero-trip loop has NO footprint: in-loop accesses are skipped
        # entirely (encode_graph). The premise must stay the exact range —
        # fabricating an iteration (max(1, n)) produced definite race
        # reports for launches that never run the body.
        self.zero_trip = n_iters == 0
        self.loop_var = Int("ttir_loop_k")
        self.loop_premises = (And(self.loop_var >= 0, self.loop_var < n_iters),)
        self._loop_bounds = (lower, step, n_iters)

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
            lower, step, _ = self._loop_bounds  # type: ignore[misc]
            return IntVal(lower) + self.loop_var * IntVal(step)
        if isinstance(term, IterArgOffset):
            info = self.graph.iter_args[term.arg_id]
            return self.eval(info.offset0) + self.loop_var * self.eval(info.delta)
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
        if isinstance(term, Select):
            return If(
                _as_bool(self.eval(term.cond)), self.eval(term.t), self.eval(term.f)
            )
        if isinstance(term, Not):
            return Z3Not(_as_bool(self.eval(term.a)))
        if isinstance(term, Observed):
            return self.observed(term.access_index)
        if isinstance(term, DataDep):
            raise UnsupportedTTIR(f"data-dependent term ({term.why})")
        raise UnsupportedTTIR(f"unhandled term {type(term).__name__}")


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
    bounds: tuple[Any, ...]
    if meta is not None:
        if meta.elem_size != elem:
            raise UnsupportedTTIR(
                f"element width mismatch for {access.base_param!r}: TTIR says "
                f"{elem} bytes, the launch tensor says {meta.elem_size}"
            )
        addr = IntVal(meta.data_ptr) + env.eval(access.offset) * IntVal(elem)
        # The in-bounds premise (see the module docstring's model boundary).
        bounds = (
            addr >= IntVal(meta.data_ptr),
            addr < IntVal(meta.data_ptr + meta.numel * meta.elem_size),
        )
    else:
        addr = env.eval(access.offset) * IntVal(elem)
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

    copy_local: tuple[Any, ...] = (env.loop_var,) if env.loop_var is not None else ()
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
        # the spin-termination invariants constrain every record.
        premises=(env.loop_premises if access.in_loop else ()) + await_premises,
        local_constraints=bounds,
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


def encode_graph(
    graph: AccessGraph,
    params: dict[str, int],
    tensors: dict[str, GlobalTensor],
) -> GlobalEncoding:
    """Lower every global access of ``graph`` into solver records under the
    concrete launch ``params``/``tensors`` (tier T1: pid, grid, arange lanes
    and loop iterations stay symbolic). Raises :class:`UnsupportedTTIR`
    (classified) when the kernel cannot be encoded."""
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

    env = _RaceEnv(graph, params)
    await_prems, await_obs = _await_premises(graph, env)
    records = []
    uncertain: set[int] = set()
    for seq, access in enumerate(graph.accesses):
        if access.in_loop and env.zero_trip:
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
        records.append(
            _record_for(
                access, seq, env, graph.kernel_name, meta, await_prems, await_obs
            )
        )
        if access.mask_dropped or access.guarded:
            uncertain.add(seq)
        if _references_unmodeled_observation(access, env):
            uncertain.add(seq)
    return GlobalEncoding(
        records=records,
        arange_dict=env.arange_dict,
        uncertain_event_ids=uncertain,
        used_pid_axes=set(graph.pid_axes),
        assumes_termination=any(a.awaited for a in graph.accesses),
        has_atomics=any(a.kind.startswith("atomic") for a in graph.accesses),
    )


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
    encoding: GlobalEncoding, launch_grid: tuple[int, ...] | None = None
) -> tuple[Any, Any, Any]:
    """The T0/T1 grid: symbolic (all sizes ≥ 1) along the pid axes the
    kernel reads; along UNREAD axes, pinned to 1 — except for
    atomic-bearing graphs, where the identical-behavior justification
    fails (see GlobalEncoding.has_atomics): those unread axes take the
    REAL launch size when one is supplied (T1) and stay SYMBOLIC when not
    (T0 — the sound direction; a resulting nonlinear counting product just
    omits the axiom and the kernel falls to T1 per the ladder)."""
    from z3 import Int

    def dim(i: int) -> Any:
        if i in encoding.used_pid_axes:
            return Int(f"grid_{i}")
        if not encoding.has_atomics:
            return 1
        if launch_grid is not None:
            return int(launch_grid[i]) if i < len(launch_grid) else 1
        return Int(f"grid_{i}")

    return (dim(0), dim(1), dim(2))


# ───────────────────── tier selector support (§I.3) ─────────────────────

# Observed counts as symbolic: the observation var is free at T0, so a
# product with another symbol is exactly the Z3-unknown bait the gate
# exists to keep out. NumPrograms is a symbolic grid dim for the same
# reason.
_SYMBOLIC_LEAVES = (Pid, Param, Arange, LoopVar, IterArgOffset, Observed, NumPrograms)


def _has_t0_symbols(term: Term) -> bool:
    if isinstance(term, _SYMBOLIC_LEAVES):
        return True
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None and _has_t0_symbols(sub):
            return True
    return False


def _linear_at_t0(term: Term, graph: AccessGraph) -> bool:
    if isinstance(term, Bin):
        if term.op == "*":
            if _has_t0_symbols(term.a) and _has_t0_symbols(term.b):
                return False
        elif term.op in ("//", "%"):
            if _has_t0_symbols(term.b):
                return False
        return _linear_at_t0(term.a, graph) and _linear_at_t0(term.b, graph)
    if isinstance(term, IterArgOffset):
        info = graph.iter_args.get(term.arg_id)
        if info is None:
            return False
        # Expands to offset0 + k·delta: linear only for a T0-constant delta.
        if _has_t0_symbols(info.delta):
            return False
        return _linear_at_t0(info.offset0, graph)
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None and not _linear_at_t0(sub, graph):
            return False
    return True


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


def encode_graph_t0(graph: AccessGraph) -> list[tuple[str, GlobalEncoding]]:
    """The T0 encoding: scalar params symbolic, one encoding PER TENSOR.

    T0 has no launch, hence no base addresses or extents. The non-aliasing
    premise (distinct pointer arguments are distinct allocations) is
    realized by PARTITIONING: accesses can only conflict within one base
    pointer's group, and addresses are byte offsets from that base.
    Aliased-argument launches sit outside the T0 claim — T1 covers them
    with the real bases. Read-only groups are skipped (read/read cannot
    conflict). Raises UnsupportedTTIR when the kernel cannot be encoded at
    T0 (e.g. a loop bound referencing a scalar param)."""
    for access in graph.accesses:
        if access.kind == "atomic_cas" and not access.awaited:
            raise UnsupportedTTIR(
                f"line {access.line_no}: atomic_cas synchronization is not "
                "modeled statically",
                kind="cas-synchronization",
            )

    env = _RaceEnv(graph, {}, symbolic_params=True)
    await_prems, await_obs = _await_premises(graph, env)
    groups: dict[str, list[tuple[int, AccessEvent]]] = {}
    for seq, access in enumerate(graph.accesses):
        if access.in_loop and env.zero_trip:
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
