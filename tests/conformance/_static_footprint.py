"""A concrete evaluator of the TTIR reader's ``AccessGraph`` (D10b; D17 note).

The static side of the conformance suite: for one launch (grid + scalar
arguments) it enumerates every program id x arange lane x loop iteration
and returns, per access site ``(base_param, kind, source line)``, the set
of ``(pid_0, pid_1, pid_2, element offset)`` points (the offset relative
to that base argument) of the lanes that execute: a footprint per program
instance, as #361's ``differential.static_footprints`` compared it and the
race detector needs it (D17), never merged across programs. Each site also
records the ``AccessEvent.elem_bits`` of its accesses. Ported from #361's
evaluator and extended to this reader's graph:

* every ``graph.iter_args`` entry, the per-axis expanded tiles included
  (an ``IterArgOffset`` is ``offset0 + k * delta`` at iteration ``k``);
* ``IntCast`` and the signed / unsigned ``Bin`` and ``Cmp`` spellings;
* lanes as TTIR broadcasting defines them: every tensor one access
  combines has the access's shape, so all aranges along one dim with one
  extent index the SAME position there (``tl.arange(0, 16) +
  tl.arange(16, 32)`` is ``2i + 16``, not ``i + j + 16`` as #361's
  per-``(ssa, dim)`` meshgrid had it), and an extent-1 arange broadcast
  along a longer dim stays at position 0;
* terms are evaluated with UNBOUNDED integers (int64 while every operand
  bound stays below 2**62, Python ints past that), and the reader's
  ``width_obligations`` are checked concretely under their role's
  discharge discipline (loop bounds unconditionally, the loop increment
  where the loop runs, path where the loop iteration runs, mask under the
  path, offset under path and mask). The unbounded reading is the IR's
  fixed-width arithmetic only while the obligations hold, so an access
  with a failing obligation is reported and never compared.

Accesses without an exact concrete footprint are excluded, never compared:
``mask_dropped`` / ``guarded`` accesses (skipped: the model deliberately
over-approximates them), accesses whose terms reach an atomic observation
(``Observed``: interleaving-dependent), a ``DataDep`` (no value), a
failing width obligation, and a division by zero or non-positive loop
step (undefined in the IR). An excluded access excludes its whole site
key, so a site is compared only when every access on it is.

The reader's graph-aware ``mentions_observed`` must agree with this
module's own walk (a disagreement is a :class:`ContractError`). This module
deliberately shares no code with the compiled sanitizer
(``tilelens/clients/sanitizer/compiled/oob.py``): the suite checks the
reader, and the sanitizer is one of its consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

import numpy as np

from tilelens.ir.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    IntCast,
    IterArgOffset,
    LoopVar,
    Not,
    NumPrograms,
    Observed,
    Param,
    Pid,
    Select,
    mentions_observed,
    width_obligations,
)

SiteKey = tuple[str, str, int]  # (base_param, kind, source line)
Point = tuple[int, int, int, int]  # (pid_0, pid_1, pid_2, element offset)

# Why an access is not compared.
MASK_DROPPED = "mask-dropped"
GUARDED = "guarded"
OBSERVED = "observed"
DATA_DEPENDENT = "data-dependent"
OBLIGATION = "obligation"
UNDEFINED = "undefined"
SKIPPED = frozenset({MASK_DROPPED, GUARDED})  # over-approximated by design

_LIMIT = 1 << 62  # int64 evaluation stays exact while bounds stay below this
_MAX_POINTS = 1 << 24  # (pid, iteration, lane) points one access may enumerate


class ContractError(AssertionError):
    """The graph breaks a contract the reader documents."""


class EvaluationError(Exception):
    """The launch cannot be evaluated (a missing argument, a term outside
    the vocabulary, a space too large to enumerate)."""


@dataclass(frozen=True)
class Exclusion:
    access: int  # index into graph.accesses
    key: SiteKey
    reason: str
    detail: str


@dataclass(frozen=True)
class ObligationFailure:
    access: int
    key: SiteKey
    role: str
    bits: int
    signed: bool
    ttir_line: int | None
    source_line: int | None
    points: int  # failing (pid, iteration, lane) points
    example: int  # one failing value


@dataclass
class StaticFootprint:
    # compared sites: key -> (program, element offset) of the lanes that execute
    sites: dict[SiteKey, set[Point]] = field(default_factory=dict)
    # excluded sites: key -> why (every excluded access on it)
    excluded: dict[SiteKey, list[Exclusion]] = field(default_factory=dict)
    obligation_failures: list[ObligationFailure] = field(default_factory=list)
    # every site's source file (the key holds its line)
    files: dict[SiteKey, set[str]] = field(default_factory=dict)
    # every site's element widths (AccessEvent.elem_bits of its accesses)
    bits: dict[SiteKey, set[int]] = field(default_factory=dict)


def site_key(access: AccessEvent) -> SiteKey:
    if access.loc is None:
        raise EvaluationError(
            f"access at TTIR line {access.line_no} has no source location"
        )
    return (access.base_param, access.kind, access.loc.line)


# ─────────────────────────── graph walk ───────────────────────────


def _kids(t: object, graph: AccessGraph) -> tuple:
    """The terms ``t``'s value is computed from: operands, a loop-carried
    pointer's ``offset0`` / ``delta``, the loop's lower bound and step for
    the induction variable, a DataDep's modelable ``keep``."""
    if isinstance(t, (Bin, Cmp, BoolBin)):
        return (t.a, t.b)
    if isinstance(t, Select):
        return (t.cond, t.t, t.f)
    if isinstance(t, Not):
        return (t.a,)
    if isinstance(t, IntCast):
        return (t.x,)
    if isinstance(t, IterArgOffset):
        info = graph.iter_args[t.arg_id]
        if info.arg_id != t.arg_id:
            raise ContractError(f"iter_args[{t.arg_id}].arg_id is {info.arg_id}")
        return (info.offset0, info.delta)
    if isinstance(t, LoopVar):
        if graph.loop is None:
            raise ContractError("an induction variable without a loop")
        return (graph.loop.lower, graph.loop.step)
    if isinstance(t, DataDep):
        return () if t.keep is None else (t.keep,)
    if isinstance(t, (Const, Pid, NumPrograms, Arange, Param, Observed)):
        return ()
    raise EvaluationError(f"term outside the vocabulary: {type(t).__name__}")


def _walk(roots: Iterable[object], graph: AccessGraph) -> Iterator[object]:
    """Every node reachable from ``roots``, each once by identity (terms can
    be deeper than the recursion limit)."""
    seen: set[int] = set()
    stack = [r for r in roots if r is not None]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        stack.extend(_kids(t, graph))


# ─────────────────────────── unbounded integer arrays ───────────────────────────


def _leaf(v: int) -> np.ndarray:
    return np.asarray(v, dtype=np.int64 if -_LIMIT < v < _LIMIT else object)


def _int(a: np.ndarray) -> np.ndarray:
    return a.astype(np.int64) if a.dtype == np.bool_ else a


def _truth(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == np.bool_ else np.asarray(a != 0, dtype=np.bool_)


def _bound(a: np.ndarray) -> int:
    """max |a| as a Python int."""
    if a.size == 0:
        return 0
    if a.dtype == np.bool_:
        return 1
    return max(abs(int(a.max())), abs(int(a.min())))


def _obj(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == object else a.astype(object)


def _compact(a: object) -> np.ndarray:
    """``a`` as an array (arithmetic on 0-d arrays yields scalars), int64
    again once its values fit."""
    arr = np.asarray(a)
    if arr.dtype == object and _bound(arr) < _LIMIT:
        return arr.astype(np.int64)
    return arr


def _wide(a: np.ndarray, b: np.ndarray, bound: int) -> tuple[np.ndarray, np.ndarray]:
    return (_obj(a), _obj(b)) if bound >= _LIMIT else (a, b)


def _tdiv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quotient truncated toward zero (arith.divsi; numpy's // floors)."""
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def _unsigned(a: np.ndarray, bits: int) -> np.ndarray:
    return _compact(_obj(a) % (1 << bits))


def _signed(a: np.ndarray, bits: int) -> np.ndarray:
    a = _obj(a)
    half = 1 << (bits - 1)
    return _compact((a + half) % (1 << bits) - half)


# ─────────────────────────── one access ───────────────────────────


class _Access:
    """The evaluation space of one access: axes (pid_0, pid_1, pid_2,
    iteration, lane positions ...), every array broadcasting over them."""

    def __init__(
        self,
        graph: AccessGraph,
        access: AccessEvent,
        params: Mapping[str, int],
        grid: tuple[int, int, int],
        lanes: list[tuple[int, int]],
    ) -> None:
        self.graph = graph
        self.access = access
        self.params = params
        self.grid = grid
        self.lanes = {key: 4 + i for i, key in enumerate(lanes)}
        self.ndim = 4 + len(lanes)
        self.shape = [*grid, 1, *(extent for _, extent in lanes)]
        self.iteration: np.ndarray | None = None
        # id(term) -> (term, value); holding the term keeps its id unique
        self.memo: dict[int, tuple[object, np.ndarray]] = {}
        # divisors: (node, zero mask) for every division evaluated
        self.divisions: list[tuple[object, np.ndarray]] = []

    def axis(self, axis: int, n: int) -> np.ndarray:
        shape = [1] * self.ndim
        shape[axis] = n
        return np.arange(n, dtype=np.int64).reshape(shape)

    def set_iterations(self, trips: int) -> None:
        self.shape[3] = trips
        self.iteration = self.axis(3, trips)

    def param(self, name: str) -> np.ndarray:
        if name not in self.params:
            raise EvaluationError(f"scalar argument {name!r} has no launch value")
        arg = self.graph.arg(name)
        bits = arg.int_bits if arg is not None else 0
        v = int(self.params[name])
        if bits == 1:
            return np.asarray(v != 0)  # i1 terms are booleans
        if bits > 1:
            half = 1 << (bits - 1)
            v = (v + half) % (1 << bits) - half  # the IR's signed reading
        return _leaf(v)

    def value(self, root: object) -> np.ndarray:
        stack: list[tuple[object, bool]] = [(root, False)]
        memo = self.memo
        while stack:
            t, ready = stack.pop()
            if id(t) in memo:
                continue
            kids = self.operands(t)
            missing = [k for k in kids if id(k) not in memo]
            if missing and not ready:
                stack.append((t, True))
                stack.extend((k, False) for k in missing)
                continue
            memo[id(t)] = (t, np.asarray(self.apply(t, [memo[id(k)][1] for k in kids])))
        return memo[id(root)][1]

    def operands(self, t: object) -> tuple:
        # an Observed / DataDep leaf has no value: apply() refuses it
        return () if isinstance(t, DataDep) else _kids(t, self.graph)

    def apply(self, t: object, v: list[np.ndarray]) -> np.ndarray:
        if isinstance(t, Const):
            return _leaf(int(t.value))
        if isinstance(t, Pid):
            return self.axis(t.axis, self.grid[t.axis])
        if isinstance(t, NumPrograms):
            return _leaf(self.grid[t.axis])
        if isinstance(t, Param):
            return self.param(t.name)
        if isinstance(t, Arange):
            key = (t.dim, t.end - t.start)
            return self.axis(self.lanes[key], t.end - t.start) + t.start
        if isinstance(t, (LoopVar, IterArgOffset)):
            if self.iteration is None:
                raise ContractError(f"{type(t).__name__} in an access outside the loop")
            # offset0 + k * delta, lower + k * step
            base, step = _int(v[0]), _int(v[1])
            k, step = _wide(self.iteration, step, _bound(self.iteration) * _bound(step))
            scaled = k * step
            base, scaled = _wide(base, scaled, _bound(base) + _bound(scaled))
            return _compact(base + scaled)
        if isinstance(t, Bin):
            return self.bin(t, _int(v[0]), _int(v[1]))
        if isinstance(t, Cmp):
            return self.cmp(t, v[0], v[1])
        if isinstance(t, BoolBin):
            a, b = _truth(v[0]), _truth(v[1])
            return a & b if t.op == "and" else a | b
        if isinstance(t, Select):
            return np.where(_truth(v[0]), v[1], v[2])
        if isinstance(t, Not):
            return ~_truth(v[0])
        if isinstance(t, IntCast):
            # the model's value (exact while the cast's obligation holds)
            return _int(v[0])
        if isinstance(t, Observed):
            raise ContractError(
                f"evaluation reached Observed({t.access_index}), which the walk did not report"
            )
        if isinstance(t, DataDep):
            raise ContractError(
                f"evaluation reached DataDep({t.why!r}), which the walk did not report"
            )
        raise EvaluationError(f"term outside the vocabulary: {type(t).__name__}")

    def divisor(self, t: Bin, b: np.ndarray) -> np.ndarray:
        zero = np.asarray(b == 0)
        if zero.any():
            self.divisions.append((t, zero))
            b = np.where(zero, 1, b)
        return b

    def bin(self, t: Bin, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        op = t.op
        if op in ("+", "-"):
            a, b = _wide(a, b, _bound(a) + _bound(b))
            return _compact(a + b if op == "+" else a - b)
        if op == "*":
            a, b = _wide(a, b, _bound(a) * _bound(b))
            return _compact(a * b)
        if op in ("min", "max"):
            return np.minimum(a, b) if op == "min" else np.maximum(a, b)
        if op in ("//", "%"):
            b = self.divisor(t, b)
            q = _tdiv(a, b)
            return q if op == "//" else _compact(_obj(a) - _obj(b) * _obj(q))
        if op in ("u//", "u%", "umin", "umax"):
            if t.bits is None:
                raise ContractError(f"unsigned op {op} without a width")
            ua, ub = _unsigned(a, t.bits), _unsigned(b, t.bits)
            if op == "umin":
                r = np.minimum(ua, ub)
            elif op == "umax":
                r = np.maximum(ua, ub)
            else:
                ub = self.divisor(t, ub)
                r = ua // ub if op == "u//" else ua % ub
            return _signed(r, t.bits)
        raise EvaluationError(f"unknown integer op {op!r}")

    def cmp(self, t: Cmp, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _int(a), _int(b)
        pred = t.pred
        if pred[0] == "u":
            if not t.bits:
                raise ContractError(f"unsigned predicate {pred} without a width")
            a, b = _unsigned(a, t.bits), _unsigned(b, t.bits)
            pred = "s" + pred[1:]
        if pred == "eq":
            return np.asarray(a == b)
        if pred == "ne":
            return np.asarray(a != b)
        ops = {
            "slt": np.less,
            "sle": np.less_equal,
            "sgt": np.greater,
            "sge": np.greater_equal,
        }
        if pred not in ops:
            raise EvaluationError(f"unknown predicate {t.pred!r}")
        return np.asarray(ops[pred](a, b), dtype=np.bool_)

    def full(self, a: np.ndarray, *, pids_only: bool = False) -> np.ndarray:
        """``a`` over the whole space, or over the program ids alone (the
        loop's bounds, computed once per program whatever the trip count)."""
        shape = self.shape[:3] + [1] * (self.ndim - 3) if pids_only else self.shape
        return np.broadcast_to(a, tuple(shape))


def _lane_keys(nodes: Iterable[object]) -> list[tuple[int, int]]:
    return sorted({(n.dim, n.end - n.start) for n in nodes if isinstance(n, Arange)})


def _check_obligations(
    ev: _Access, index: int, key: SiteKey, bound_ids: set[int], valid, path, mask, trips
) -> list[ObligationFailure]:
    """The access's width obligations, each where its role says it matters."""
    out = []
    for ob in width_obligations(ev.graph, ev.access):
        pids_only = ob.role == "loop"
        if pids_only:
            # the bounds: unconditional; the increment (a term of its own):
            # where the loop runs at least once
            cond = (
                np.asarray(True) if id(ob.term) in bound_ids else np.asarray(trips > 0)
            )
        elif ob.role == "path":
            cond = valid
        elif ob.role == "mask":
            cond = valid & path
        elif ob.role == "offset":
            cond = valid & path & mask
        else:
            raise ContractError(f"unknown obligation role {ob.role!r}")
        v = _int(ev.value(ob.term))
        if ob.signed:
            lo, hi = -(1 << (ob.bits - 1)), 1 << (ob.bits - 1)
        else:
            lo, hi = 0, 1 << ob.bits
        fits = np.asarray((v >= lo) & (v < hi), dtype=np.bool_)
        bad = ev.full(cond, pids_only=pids_only) & ~ev.full(fits, pids_only=pids_only)
        if bad.any():
            example = int(ev.full(v, pids_only=pids_only)[bad].flat[0])
            out.append(
                ObligationFailure(
                    index,
                    key,
                    ob.role,
                    ob.bits,
                    ob.signed,
                    ob.line_no,
                    ob.loc.line if ob.loc is not None else None,
                    int(bad.sum()),
                    example,
                )
            )
    return out


def _evaluate(
    graph: AccessGraph,
    index: int,
    params: Mapping[str, int],
    grid: tuple[int, int, int],
    out: StaticFootprint,
) -> tuple[set[Point] | None, list[Exclusion]]:
    """One access's footprint, or None with the reasons it is excluded."""
    access = graph.accesses[index]
    key = site_key(access)
    if access.mask_dropped or access.guarded:
        why = [MASK_DROPPED] * access.mask_dropped + [GUARDED] * access.guarded
        return None, [
            Exclusion(index, key, r, "over-approximated by the model") for r in why
        ]
    loop = graph.loop
    if access.in_loop and loop is None:
        raise ContractError(f"access {index} is in_loop but the graph has no loop")
    roots = [access.offset, access.mask, access.path]
    if access.in_loop:
        roots += [loop.lower, loop.upper, loop.step]  # type: ignore[union-attr]
    nodes = list(_walk(roots, graph))
    observed = any(isinstance(n, Observed) for n in nodes)
    reader_says = any(t is not None and mentions_observed(t, graph) for t in roots)
    if observed != reader_says:
        raise ContractError(
            f"access {index} (line {key[2]}): mentions_observed says {reader_says}, the graph walk {observed}"
        )
    if observed:
        return None, [
            Exclusion(
                index,
                key,
                OBSERVED,
                "reads an atomic observation (interleaving-dependent)",
            )
        ]
    deps = [n.why for n in nodes if isinstance(n, DataDep)]
    if deps:
        return None, [
            Exclusion(index, key, DATA_DEPENDENT, "; ".join(sorted(set(deps))))
        ]

    ev = _Access(graph, access, params, grid, _lane_keys(nodes))
    trips = np.asarray(1)
    bound_ids: set[int] = set()
    if access.in_loop:
        assert loop is not None
        bound_ids = {id(n) for n in _walk((loop.lower, loop.upper, loop.step), graph)}
        lower, upper, step = (
            _int(ev.value(b)) for b in (loop.lower, loop.upper, loop.step)
        )
        if (np.asarray(step) <= 0).any():
            return None, [Exclusion(index, key, UNDEFINED, "a non-positive loop step")]
        trips = np.maximum(_obj(upper) - _obj(lower) + _obj(step) - 1, 0) // _obj(step)
        trips = _compact(np.asarray(trips))
        ev.set_iterations(int(np.max(trips)) if trips.size else 0)
        valid = np.asarray(ev.iteration < trips)
    else:
        valid = np.asarray(True)
    points = int(np.prod(ev.shape))
    if points > _MAX_POINTS:
        raise EvaluationError(
            f"access {index} spans {points} points (limit {_MAX_POINTS})"
        )

    path = (
        _truth(ev.value(access.path)) if access.path is not None else np.asarray(True)
    )
    mask = (
        _truth(ev.value(access.mask)) if access.mask is not None else np.asarray(True)
    )
    offset = _int(ev.value(access.offset))
    failures = _check_obligations(ev, index, key, bound_ids, valid, path, mask, trips)
    if failures:
        out.obligation_failures += failures
        detail = ", ".join(
            f"{f.role} i{f.bits} at source line {f.source_line}: e.g. {f.example}"
            for f in failures
        )
        return None, [Exclusion(index, key, OBLIGATION, detail)]
    for node, zero in ev.divisions:
        # a divisor of the loop's bounds counts in every program, others
        # at the iterations that run
        hit = (
            ev.full(zero, pids_only=True).any()
            if id(node) in bound_ids
            else ev.full(zero & valid).any()
        )
        if hit:
            return None, [
                Exclusion(
                    index,
                    key,
                    UNDEFINED,
                    f"a division by zero (TTIR line {getattr(node, 'line_no', None)})",
                )
            ]
    active = ev.full(valid & path & mask)
    # np.nonzero and boolean indexing both walk the space in C order
    p0, p1, p2 = (i.tolist() for i in np.nonzero(active)[:3])
    offsets = ev.full(offset)[active].tolist()
    return {
        (a, b, c, int(o)) for a, b, c, o in zip(p0, p1, p2, offsets, strict=True)
    }, []


def static_footprint(
    graph: AccessGraph,
    params: Mapping[str, int],
    grid: tuple[int, ...],
) -> StaticFootprint:
    """The model's footprint of one launch: ``params`` maps every scalar
    kernel argument to its launch value, ``grid`` is the launch grid."""
    grid3 = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
    assert len(grid3) == 3 and all(g >= 1 for g in grid3), grid
    out = StaticFootprint()
    for index in range(len(graph.accesses)):
        offsets, why = _evaluate(graph, index, params, grid3, out)  # type: ignore[arg-type]
        access = graph.accesses[index]
        key = site_key(access)
        out.files.setdefault(key, set()).add(access.loc.file)  # type: ignore[union-attr]
        out.bits.setdefault(key, set()).add(access.elem_bits)
        if why:
            out.excluded.setdefault(key, []).extend(why)
        else:
            assert offsets is not None
            out.sites.setdefault(key, set()).update(offsets)
    for key in out.excluded:
        out.sites.pop(key, None)
    return out
