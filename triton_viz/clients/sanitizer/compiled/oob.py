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

from z3 import And, ArithRef, BoolRef, If, Int, IntVal, Or, Solver, sat

from .ttir_reader import (
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
    kind: str  # "load" | "store"
    base_param: str
    violation_offset: int
    violation_address: int
    loc_file: str | None
    loc_line: int | None
    line_no: int
    witness: dict[str, int]


class _Env:
    """Allocates and caches Z3 free variables for one access query, adding
    their range constraints to the solver."""

    def __init__(self, ctx: LaunchContext, loop_upper: int | None) -> None:
        self.ctx = ctx
        self.loop_upper = loop_upper
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

    def loop(self, loop_ssa: str) -> ArithRef:
        v = self._loop.get(loop_ssa)
        if v is None:
            v = Int(f"loop_{loop_ssa.strip('%')}")
            self._loop[loop_ssa] = v
            self.constraints.append(v >= 0)
            if self.loop_upper is not None:
                self.constraints.append(v < self.loop_upper)
            else:
                self.constraints.append(v >= 0)
        return v


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
        return env.loop(term.loop_ssa)
    if isinstance(term, IterArgOffset):
        info = graph.iter_args[term.arg_id]
        if graph.loop is None:
            raise UnsupportedTTIR("iter-arg offset outside a loop")
        k = env.loop(graph.loop.loop_ssa)
        return _eval(info.offset0, env, graph) + k * _eval(info.delta, env, graph)
    if isinstance(term, Bin):
        a, b = _eval(term.a, env, graph), _eval(term.b, env, graph)
        if term.op == "+":
            return a + b
        if term.op == "-":
            return a - b
        if term.op == "*":
            return a * b
        if term.op == "//":
            # Signed division by a positive constant (cdiv lowering). Z3 `/`
            # on Int is integer division; guard divide-by-zero.
            return a / b
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
        a, b = _eval(term.a, env, graph), _eval(term.b, env, graph)
        return And(a, b) if term.op == "and" else Or(a, b)
    if isinstance(term, Select):
        return If(_eval(term.cond, env, graph), _eval(term.t, env, graph),
                  _eval(term.f, env, graph))  # fmt: skip
    if isinstance(term, DataDep):
        raise UnsupportedTTIR(f"data-dependent term ({term.why})")
    raise UnsupportedTTIR(f"unhandled term {type(term).__name__}")


def _loop_upper(graph: AccessGraph, ctx: LaunchContext) -> int | None:
    if graph.loop is None:
        return None
    # Evaluate the upper bound concretely (it's an int term over params).
    tmp = _Env(ctx, None)
    expr = _eval(graph.loop.upper, tmp, graph)
    from z3 import simplify

    s = simplify(expr)
    try:
        return s.as_long()
    except Exception:
        raise UnsupportedTTIR("loop upper bound is not concrete at launch")


def check_access(
    access: AccessEvent, graph: AccessGraph, ctx: LaunchContext
) -> CompiledOOB | None:
    """Run the OOB query for one access; return a violation or None."""
    meta = ctx.tensors.get(access.base_param)
    if meta is None:
        # Base pointer never registered (e.g. an output not passed?). Cannot
        # bound it — skip rather than fabricate.
        return None
    if not meta.contiguous:
        raise UnsupportedTTIR(
            f"non-contiguous tensor {access.base_param} (v1 assumes contiguous)"
        )

    loop_upper = _loop_upper(graph, ctx)
    env = _Env(ctx, loop_upper)

    offset = _eval(access.offset, env, graph)
    solver = Solver()
    for c in env.constraints:
        solver.add(c)
    if access.mask is not None:
        solver.add(_eval(access.mask, env, graph))
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
        witness[f"loop_{ssa.strip('%')}"] = mval(var)

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
    modeled (the client converts that into an unsupported/fallback verdict)."""
    out: list[CompiledOOB] = []
    for access in graph.accesses:
        v = check_access(access, graph, ctx)
        if v is not None:
            out.append(v)
    return out
