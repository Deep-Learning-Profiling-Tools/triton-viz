"""C3 — the differential cross-check (plan §I.4).

The static side of the diff: a CONCRETE, numpy-only evaluator of the shared
TTIR ``AccessGraph`` that enumerates one program instance's byte footprint
at a given (pid, params). It is deliberately independent of the Z3 encoding
(``global_records``): the whole point of the channel is that the compiled
lowering (TTIR reader semantics) and the interpreter's execution are
compared through two implementations that share nothing but the kernel.

Granularity: element-start byte addresses per ``(base_param, kind)``, with
one uniform element size per tensor. The interpreter-side footprints (the
replay client) use the same convention, so masked-off lanes are naturally
absent from both sides — no lane-convention alignment is needed at this
granularity.

Accesses that the static model over-approximates (``mask_dropped`` /
``guarded``) have no exact concrete footprint; they are reported in
``skipped`` rather than silently compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...common.ttir_reader import (
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

# kind → the footprint bucket shared with the replay client.
KIND_BUCKET = {
    "load": "load",
    "store": "store",
    "atomic_rmw": "atomic_rmw",
    "atomic_cas": "atomic_cas",
}


@dataclass
class StaticFootprints:
    """One program instance's statically-enumerated footprint."""

    # (base_param, kind bucket) -> set of element-start byte addresses
    footprints: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    # accesses with no exact concrete footprint (over-approximated)
    skipped: list[str] = field(default_factory=list)


class _ConcreteEnv:
    def __init__(
        self,
        graph: AccessGraph,
        params: dict[str, int],
        pid: tuple[int, int, int],
    ) -> None:
        self.graph = graph
        self.params = params
        self.pid = pid
        # (ssa, dim) -> meshgrid axis index, assigned on first sight per access
        self.axes: dict[tuple[str, int], int] = {}
        self.arange_ranges: list[tuple[int, int]] = []
        self.loop_iter = 0  # concrete iteration index, set by the caller

    def collect_aranges(self, term: Term) -> None:
        if isinstance(term, Arange):
            key = (term.ssa, term.dim)
            if key not in self.axes:
                self.axes[key] = len(self.arange_ranges)
                self.arange_ranges.append((term.start, term.end))
            return
        if isinstance(term, IterArgOffset):
            info = self.graph.iter_args[term.arg_id]
            self.collect_aranges(info.offset0)
            self.collect_aranges(info.delta)
            return
        for attr in ("a", "b", "cond", "t", "f"):
            sub = getattr(term, attr, None)
            if sub is not None:
                self.collect_aranges(sub)

    def grids(self) -> list[np.ndarray]:
        """One int64 meshgrid axis per distinct (arange, dim) instance."""
        if not self.arange_ranges:
            return []
        axes = [np.arange(s, e, dtype=np.int64) for s, e in self.arange_ranges]
        return list(np.meshgrid(*axes, indexing="ij", sparse=True))

    def eval(self, term: Term, grids: list[np.ndarray]) -> Any:
        if isinstance(term, Const):
            return np.int64(term.value)
        if isinstance(term, Param):
            if term.name not in self.params:
                raise UnsupportedTTIR(
                    f"scalar param {term.name!r} not available for the diff"
                )
            return np.int64(self.params[term.name])
        if isinstance(term, Pid):
            return np.int64(self.pid[term.axis])
        if isinstance(term, Arange):
            return grids[self.axes[(term.ssa, term.dim)]]
        if isinstance(term, LoopVar):
            loop = self.graph.loop
            assert loop is not None
            lower = int(self.eval(loop.lower, grids))
            step = int(self.eval(loop.step, grids))
            return np.int64(lower + self.loop_iter * step)
        if isinstance(term, IterArgOffset):
            info = self.graph.iter_args[term.arg_id]
            return self.eval(info.offset0, grids) + self.loop_iter * self.eval(
                info.delta, grids
            )
        if isinstance(term, Bin):
            a, b = self.eval(term.a, grids), self.eval(term.b, grids)
            if term.op == "+":
                return a + b
            if term.op == "-":
                return a - b
            if term.op == "*":
                return a * b
            if term.op == "//":
                # arith.divsi truncates toward zero (C semantics), while
                # numpy's // floors — divide magnitudes and re-sign.
                q = np.abs(a) // np.abs(b)
                return np.where((a >= 0) == (b >= 0), q, -q)
            if term.op == "%":
                q = np.abs(a) // np.abs(b)
                q = np.where((a >= 0) == (b >= 0), q, -q)
                return a - b * q
            if term.op == "min":
                return np.minimum(a, b)
            if term.op == "max":
                return np.maximum(a, b)
            raise UnsupportedTTIR(f"unknown arith op {term.op}")
        if isinstance(term, Cmp):
            a, b = self.eval(term.a, grids), self.eval(term.b, grids)
            return {
                "slt": a < b, "sle": a <= b, "sgt": a > b,
                "sge": a >= b, "eq": a == b, "ne": a != b,
            }[term.pred]  # fmt: skip
        if isinstance(term, BoolBin):
            a, b = self.eval(term.a, grids), self.eval(term.b, grids)
            return (a != 0) & (b != 0) if term.op == "and" else (a != 0) | (b != 0)
        if isinstance(term, Select):
            return np.where(
                self.eval(term.cond, grids) != 0,
                self.eval(term.t, grids),
                self.eval(term.f, grids),
            )
        if isinstance(term, Not):
            return ~(self.eval(term.a, grids) != 0)
        if isinstance(term, DataDep):
            raise UnsupportedTTIR(f"data-dependent term ({term.why})")
        raise UnsupportedTTIR(f"unhandled term {type(term).__name__}")


def _loop_trip(graph: AccessGraph, env: _ConcreteEnv) -> int:
    loop = graph.loop
    if loop is None:
        return 1
    lower = int(env.eval(loop.lower, []))
    upper = int(env.eval(loop.upper, []))
    step = int(env.eval(loop.step, []))
    if step <= 0:
        raise UnsupportedTTIR(f"loop step {step} <= 0")
    return max(0, (upper - lower + step - 1) // step)


def static_footprints(
    graph: AccessGraph,
    params: dict[str, int],
    tensor_bases: dict[str, tuple[int, int]],  # name -> (data_ptr, elem_size)
    pid: tuple[int, int, int],
) -> StaticFootprints:
    """Enumerate one program instance's footprint from the STATIC model:
    every arange lane × every loop iteration, masks and path conditions
    applied concretely."""
    out = StaticFootprints()
    for access in graph.accesses:
        if access.mask_dropped or access.guarded:
            out.skipped.append(
                f"line {access.line_no} ({access.kind}): over-approximated "
                "(dropped mask / unmodeled branch)"
            )
            continue
        base, elem = tensor_bases[access.base_param]

        env = _ConcreteEnv(graph, params, pid)
        env.collect_aranges(access.offset)
        if access.mask is not None:
            env.collect_aranges(access.mask)
        if access.path is not None:
            env.collect_aranges(access.path)
        trip = _loop_trip(graph, env) if access.in_loop else 1

        bucket = out.footprints.setdefault(
            (access.base_param, KIND_BUCKET[access.kind]), set()
        )
        for k in range(trip):
            env.loop_iter = k
            grids = env.grids()
            off = env.eval(access.offset, grids)
            active = np.broadcast_to(np.bool_(True), np.shape(off) or (1,))
            if access.mask is not None:
                m = env.eval(access.mask, grids)
                active = active & np.broadcast_to(m != 0, active.shape)
            if access.path is not None:
                p = env.eval(access.path, grids)
                active = active & np.broadcast_to(p != 0, active.shape)
            off = np.broadcast_to(off, active.shape)
            addrs = base + off[active].astype(np.int64) * elem
            bucket.update(int(a) for a in addrs.ravel())
    return out


def diff_footprints(
    static: dict[tuple[str, str], set[int]],
    dynamic: dict[tuple[str, str], set[int]],
) -> list[str]:
    """One-to-one comparison; returns human-readable mismatches (empty =
    the two implementations agree on this instance's footprint)."""
    issues: list[str] = []
    for key in sorted(set(static) | set(dynamic)):
        s = static.get(key, set())
        d = dynamic.get(key, set())
        if s == d:
            continue
        only_s = sorted(s - d)[:5]
        only_d = sorted(d - s)[:5]
        issues.append(
            f"{key}: static-only={only_s} ({len(s - d)} total), "
            f"interpreter-only={only_d} ({len(d - s)} total)"
        )
    return issues
