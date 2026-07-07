"""Per-launch out-of-bounds query for the compiled sanitizer.

Given an :class:`AccessGraph` parsed from TTIR and the concrete launch
metadata (grid dims, scalar argument values, per-tensor element counts),
each access becomes a Z3 query over free variables — program ids, arange
lanes, the loop induction variable — with scalar args substituted as
constants:

    OOB  iff  SAT( mask  AND  (offset < 0  OR  offset >= numel) )

where ``offset`` is the element offset of the access relative to its base
tensor. SAT yields a witness (a concrete pid / lane / iteration whose mask
is live yet whose offset escapes the tensor), reported with the byte
violation address ``data_ptr + offset * elem_size``. UNSAT over all accesses
is a proof that the kernel is in-bounds for ALL inputs consistent with the
given scalar values and grid — for this launch's specialization.

Matches the eager sanitizer's contract: the valid element range is the
closed interval ``[0, numel-1]`` (eager uses inclusive byte bounds), and the
mask is ANDed into the access constraints so masked-off lanes cannot witness
a violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from z3 import And, ArithRef, BoolRef, If, Int, IntVal, Or, Solver, is_bool, sat
from z3 import Not as Z3Not

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


@dataclass(frozen=True)
class TensorMeta:
    numel: int
    elem_bits: int
    data_ptr: int
    contiguous: bool


@dataclass(frozen=True)
class LaunchContext:
    grid: tuple[int, int, int]
    params: dict[str, int]  # scalar arg name -> concrete value
    tensors: dict[str, TensorMeta]  # base ptr arg name -> metadata


@dataclass(frozen=True)
class CompiledOOB:
    kind: str  # "load" | "store" | "atomic_rmw" | "atomic_cas"
    base_param: str
    violation_offset: int
    violation_address: int
    loc_file: str | None
    loc_line: int | None
    line_no: int
    witness: dict[str, int]


class _Env:
    """Allocates and caches Z3 free variables for one access query, adding
    their range constraints to the solver.

    The loop free variable is the 0-based ITERATION INDEX ``iter``, not the
    induction value: scf.for ``%k = lower to upper step step`` visits
    ``lower, lower+step, ...``, so the induction value at iteration ``iter``
    is ``lower + iter*step`` and a loop-carried pointer advanced by ``delta``
    each iteration sits at ``offset0 + iter*delta``. Constraining ``iter``
    (with ``lower + iter*step < upper``) instead of bounding the induction
    value as ``[0, upper)`` is what makes non-zero ``lower`` / non-unit
    ``step`` loops sound rather than checking iterations that never run.
    """

    def __init__(
        self, ctx: LaunchContext, loop_bounds: tuple[int, int, int] | None
    ) -> None:
        self.ctx = ctx
        self.loop_bounds = loop_bounds  # (lower, step, upper) concrete
        self.constraints: list[BoolRef] = []
        self._pid = [Int(f"pid_{i}") for i in range(3)]
        self._arange: dict[tuple[str, int], ArithRef] = {}
        self._loop: dict[str, ArithRef] = {}
        for i in range(3):
            self.constraints.append(self._pid[i] >= 0)
            self.constraints.append(self._pid[i] < ctx.grid[i])

    def pid(self, axis: int) -> ArithRef:
        return self._pid[axis]

    def arange(self, ar: Arange) -> ArithRef:
        key = (ar.ssa, ar.dim)
        v = self._arange.get(key)
        if v is None:
            v = Int(f"arange_{ar.ssa.strip('%')}_d{ar.dim}")
            self._arange[key] = v
            self.constraints.append(v >= ar.start)
            self.constraints.append(v < ar.end)
        return v

    def loop_iter(self, loop_ssa: str) -> ArithRef:
        """0-based iteration index, constrained to the iterations that run."""
        v = self._loop.get(loop_ssa)
        if v is None:
            v = Int(f"iter_{loop_ssa.strip('%')}")
            self._loop[loop_ssa] = v
            self.constraints.append(v >= 0)
            if self.loop_bounds is not None:
                lower, step, upper = self.loop_bounds
                self.constraints.append(lower + v * step < upper)
        return v

    def induction_value(self, loop_ssa: str) -> ArithRef:
        it = self.loop_iter(loop_ssa)
        if self.loop_bounds is None:
            return it
        lower, step, _upper = self.loop_bounds
        return lower + it * step


def _trunc_div(a: ArithRef, b: ArithRef) -> ArithRef:
    """arith.divsi rounds toward zero, but Z3's Int `/` is Euclidean (floor
    for a positive divisor) — they disagree on negative dividends. Divide the
    magnitudes, where the two definitions coincide, and re-apply the sign."""
    aa = If(a >= 0, a, -a)
    ab = If(b >= 0, b, -b)
    q = aa / ab
    return If((a >= 0) == (b >= 0), q, -q)


def _as_bool(e: Any) -> Any:
    """Coerce an evaluated term into a Z3 Bool. Integer i1 constants (e.g.
    the dense<true> mask of an unmasked atomic parse to Const(1)) arrive as
    Int sort; a boolean position needs ``e != 0`` instead."""
    return e if is_bool(e) else e != 0


def _eval(term: Term, env: _Env, graph: AccessGraph) -> ArithRef:
    """Lower an integer/bool address term to Z3 under the launch context."""
    if isinstance(term, Const):
        return IntVal(term.value)
    if isinstance(term, Param):
        if term.name not in env.ctx.params:
            raise UnsupportedTTIR(f"scalar param {term.name} not provided at launch")
        return IntVal(env.ctx.params[term.name])
    if isinstance(term, Pid):
        return env.pid(term.axis)
    if isinstance(term, Arange):
        return env.arange(term)
    if isinstance(term, LoopVar):
        return env.induction_value(term.loop_ssa)
    if isinstance(term, IterArgOffset):
        info = graph.iter_args[term.arg_id]
        if graph.loop is None:
            raise UnsupportedTTIR("iter-arg offset outside a loop")
        it = env.loop_iter(graph.loop.loop_ssa)
        return _eval(info.offset0, env, graph) + it * _eval(info.delta, env, graph)
    if isinstance(term, Bin):
        a, b = _eval(term.a, env, graph), _eval(term.b, env, graph)
        if term.op == "+":
            return a + b
        if term.op == "-":
            return a - b
        if term.op == "*":
            return a * b
        if term.op == "//":
            return _trunc_div(a, b)
        if term.op == "%":
            # arith.remsi: remainder carries the dividend's sign.
            return a - b * _trunc_div(a, b)
        if term.op == "min":
            return If(a <= b, a, b)
        if term.op == "max":
            return If(a >= b, a, b)
        raise UnsupportedTTIR(f"unknown arith op {term.op}")
    if isinstance(term, Cmp):
        a, b = _eval(term.a, env, graph), _eval(term.b, env, graph)
        table = {
            "slt": a < b, "sle": a <= b, "sgt": a > b,
            "sge": a >= b, "eq": a == b, "ne": a != b,
        }  # fmt: skip
        if term.pred not in table:
            raise UnsupportedTTIR(f"unknown cmp predicate {term.pred}")
        return table[term.pred]
    if isinstance(term, BoolBin):
        a, b = _as_bool(_eval(term.a, env, graph)), _as_bool(_eval(term.b, env, graph))
        return And(a, b) if term.op == "and" else Or(a, b)
    if isinstance(term, Select):
        return If(_as_bool(_eval(term.cond, env, graph)), _eval(term.t, env, graph),
                  _eval(term.f, env, graph))  # fmt: skip
    if isinstance(term, Not):
        return Z3Not(_as_bool(_eval(term.a, env, graph)))
    if isinstance(term, DataDep):
        raise UnsupportedTTIR(f"data-dependent term ({term.why})")
    raise UnsupportedTTIR(f"unhandled term {type(term).__name__}")


def _loop_bounds(graph: AccessGraph, ctx: LaunchContext) -> tuple[int, int, int] | None:
    """Evaluate (lower, step, upper) concretely from the scalar args. Returns
    None when there's no loop; raises UnsupportedTTIR for a non-constant
    bound or a non-positive step (descending loops are not modeled)."""
    if graph.loop is None:
        return None
    from z3 import simplify

    tmp = _Env(ctx, None)

    def conc(term: Term, what: str) -> int:
        s = simplify(_eval(term, tmp, graph))
        try:
            return s.as_long()
        except Exception:
            raise UnsupportedTTIR(f"loop {what} is not concrete at launch")

    lower = conc(graph.loop.lower, "lower bound")
    step = conc(graph.loop.step, "step")
    upper = conc(graph.loop.upper, "upper bound")
    if step <= 0:
        raise UnsupportedTTIR(f"loop step {step} <= 0 (descending loops unsupported)")
    return (lower, step, upper)


def check_access(
    access: AccessEvent, graph: AccessGraph, ctx: LaunchContext
) -> CompiledOOB | None:
    """Run the OOB query for one access; return a violation or None."""
    meta = ctx.tensors.get(access.base_param)
    if meta is None:
        # The base pointer has no registered tensor metadata, so this access
        # cannot be bounded. Skipping it would let an unchecked load/store
        # slip through and still report last_status="ok" — a false proof.
        # A static proof is only valid once EVERY access is checked, so bail
        # to "unsupported" (the client surfaces it) rather than skip.
        raise UnsupportedTTIR(
            f"missing tensor metadata for base pointer {access.base_param!r}"
        )
    if not meta.contiguous:
        raise UnsupportedTTIR(
            f"non-contiguous tensor {access.base_param} (v1 assumes contiguous)"
        )

    loop_bounds = _loop_bounds(graph, ctx)
    env = _Env(ctx, loop_bounds)

    offset = _eval(access.offset, env, graph)
    solver = Solver()
    for c in env.constraints:
        solver.add(c)
    if access.mask is not None:
        solver.add(_as_bool(_eval(access.mask, env, graph)))
    if access.path is not None:
        # Modeled branch condition: the access only executes when its
        # scf.if path holds, so a SAT model under it is a REAL witness.
        solver.add(_as_bool(_eval(access.path, env, graph)))
    # OOB: element offset escapes [0, numel-1].
    solver.add(Or(offset < 0, offset >= meta.numel))

    if solver.check() != sat:
        return None

    model = solver.model()

    def mval(v: ArithRef) -> int:
        r = model.eval(v, model_completion=True)
        return r.as_long()

    off_val = mval(offset)
    elem_bytes = max(1, meta.elem_bits // 8)
    witness = {f"pid_{i}": mval(env.pid(i)) for i in range(3)}
    for (ssa, dim), var in env._arange.items():
        witness[f"arange_{ssa.strip('%')}_d{dim}"] = mval(var)
    for ssa, var in env._loop.items():
        witness[f"iter_{ssa.strip('%')}"] = mval(var)

    return CompiledOOB(
        kind=access.kind,
        base_param=access.base_param,
        violation_offset=off_val,
        violation_address=meta.data_ptr + off_val * elem_bytes,
        loc_file=access.loc.file if access.loc else None,
        loc_line=access.loc.line if access.loc else None,
        line_no=access.line_no,
        witness=witness,
    )


def check_graph(graph: AccessGraph, ctx: LaunchContext) -> list[CompiledOOB]:
    """Check every access; raises UnsupportedTTIR if any access can't be
    modeled (the client converts that into an ``unsupported`` verdict with
    empty records — it does not auto-fall back to interpreted checking).

    An access under a MODELED scf.if condition carries it as ``path`` and is
    checked precisely (the path constrains the query, so SAT is a real,
    reachable witness). Two over-approximations follow the same uncertainty
    discipline: ``guarded`` (an UNMODELABLE branch condition — checked as if
    unconditional) and ``mask_dropped`` (a data-dependent mask widened to
    free). For both, UNSAT is still a sound proof — dropping constraints
    only widens the footprint — but a SAT hit may sit in a state the launch
    never reaches, so it raises ``unsupported`` instead of being reported.
    SAT on an exact access is always a real witness and takes precedence
    over any uncertainty."""
    out: list[CompiledOOB] = []
    uncertain: AccessEvent | None = None
    for access in graph.accesses:
        v = check_access(access, graph, ctx)
        if v is None:
            continue
        if access.guarded or access.mask_dropped:
            uncertain = uncertain or access
            continue
        out.append(v)
    if not out and uncertain is not None:
        # Point the user at their source line when the loc resolved; the raw
        # TTIR line number means nothing to a CLI/API reader.
        if uncertain.loc is not None:
            where = f"{uncertain.loc.file}:{uncertain.loc.line}"
        else:
            where = f"TTIR line {uncertain.line_no}"
        if uncertain.guarded:
            raise UnsupportedTTIR(
                f"{where}: possible OOB on a branch-guarded "
                "access — the branch condition is not modeled, so the "
                "witness may not be reachable",
                kind="unmodelable-condition",
            )
        raise UnsupportedTTIR(
            f"{where}: possible OOB behind a data-dependent "
            "mask — the mask was over-approximated as free, so the witness "
            "may not be reachable",
            kind="data-dependent-mask",
        )
    return out
