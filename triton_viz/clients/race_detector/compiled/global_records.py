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

from ....core.data import AtomicRMW, Load, Store
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
    Param,
    Pid,
    Select,
    Term,
    UnsupportedTTIR,
)

_KNOWN_SEMS = ("relaxed", "acquire", "release", "acq_rel")


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


@dataclass
class GlobalEncoding:
    records: list[Any]
    arange_dict: dict[Any, Any]
    # event_ids of records built from over-approximated accesses
    # (mask_dropped / guarded): SAT reports touching them are not witnesses.
    uncertain_event_ids: set[int] = field(default_factory=set)
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
    arange summary vars, one symbolic loop index)."""

    def __init__(self, graph: AccessGraph, params: dict[str, int]) -> None:
        from ...symbolic_engine import SymbolicExpr

        self._pids = (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)
        self.graph = graph
        self.params = params
        self.arange_dict: dict[Any, Any] = {}
        self._arange_vars: dict[tuple[str, int], Any] = {}
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
            if term.name not in self.params:
                raise UnsupportedTTIR(
                    f"scalar param {term.name!r} not captured at launch"
                )
            return IntVal(self.params[term.name])
        if isinstance(term, Pid):
            return self._pids[term.axis]
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


def _record_for(
    access: AccessEvent,
    seq: int,
    env: _RaceEnv,
    tensors: dict[str, GlobalTensor],
    kernel_name: str,
) -> Any:
    from ..data import AccessEventRecord

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
    elem = access.elem_bits // 8
    if elem <= 0:
        raise UnsupportedTTIR(
            f"unknown element width for {access.base_param!r} "
            f"(elem_bits={access.elem_bits})"
        )
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
    else:
        sem = "plain"
        op_type = Store if access.kind == "store" else Load
        is_atomic, atomic_kind = False, "none"
        access_mode = "write" if access.kind == "store" else "read"
        reads, writes = None, None
        scope = None

    copy_local = (env.loop_var,) if env.loop_var is not None else ()
    source = (
        (access.loc.file, access.loc.line, kernel_name)
        if access.loc is not None
        else None
    )

    return AccessEventRecord(
        op_type=op_type,
        access_mode=access_mode,
        tensor=None,
        tensor_name=access.base_param,
        addr_expr=addr,
        # The iteration range constrains only the accesses that iterate.
        premises=env.loop_premises if access.in_loop else (),
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
        if access.kind == "atomic_cas":
            # v1 has no static CAS synchronizes-with / coherence model (the
            # solver's CAS machinery needs value modeling the IR front-end
            # cannot provide). Route to the interpreter front-end.
            raise UnsupportedTTIR(
                f"line {access.line_no}: atomic_cas synchronization is not "
                "modeled statically",
                kind="cas-synchronization",
            )

    env = _RaceEnv(graph, params)
    records = []
    uncertain: set[int] = set()
    for seq, access in enumerate(graph.accesses):
        if access.in_loop and env.zero_trip:
            # The launch's trip count is zero: these accesses never execute.
            continue
        records.append(_record_for(access, seq, env, tensors, graph.kernel_name))
        if access.mask_dropped or access.guarded:
            uncertain.add(seq)
    return GlobalEncoding(
        records=records,
        arange_dict=env.arange_dict,
        uncertain_event_ids=uncertain,
        used_pid_axes=set(graph.pid_axes),
    )


def t1_grid(encoding: GlobalEncoding) -> tuple[Any, Any, Any]:
    """The T1 grid: symbolic (all sizes ≥ 1) along the pid axes the kernel
    reads, pinned to 1 along the axes it ignores (see used_pid_axes)."""
    from z3 import Int

    return tuple(  # type: ignore[return-value]
        Int(f"grid_{i}") if i in encoding.used_pid_axes else 1 for i in range(3)
    )
