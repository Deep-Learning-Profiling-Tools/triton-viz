"""Shared HB primitives for the race-detector solvers.

Both ``hb_solver.py`` (single-copy synthetic) and ``two_copy_symbolic_hb_solver.py``
(production) import from here. The production path must NOT import from
``hb_solver.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from z3 import And, BoolSort, BoolVal, Or, substitute
from z3.z3 import BoolRef, IntNumRef


class UnsupportedSymbolicRaceQuery(Exception):
    """Raised when a record's shape cannot be reduced to Z3 templates.

    Either capture-side normalisation or solver-side lowering may raise this.
    Callers either propagate (when ``abort_on_error``) or swallow and emit
    zero reports — never fall back to concrete replay.
    """


def as_bool(value: Any) -> BoolRef:
    """Coerce ``value`` to a Z3 ``BoolRef``.

    Accepts Python bool/int/float, Z3 ``IntNumRef``, and non-Bool-sort Z3
    expressions (returning ``value != 0``). ``None`` is intentionally NOT
    handled — passing ``None`` indicates a caller-side bug and should
    surface as a Z3 type error rather than silently inactivating the
    event.
    """
    if isinstance(value, bool):
        return BoolVal(value)
    if isinstance(value, IntNumRef):
        return BoolVal(value.as_long() != 0)
    if isinstance(value, (int, float)):
        return BoolVal(value != 0)
    if hasattr(value, "sort") and value.sort() != BoolSort():
        return value != 0
    return value


def iter_constraints(value: Any) -> Iterator[Any]:
    """Flatten ``None`` / scalar / list / tuple into individual constraints."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_constraints(item)
        return
    yield value


def to_lanes(value: Any) -> tuple[Any, ...]:
    """Return ``value`` as a tuple of per-lane components."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def lane_value(value: Any, lane: int, n_lanes: int) -> Any:
    """Return the component of ``value`` at ``lane``.

    - ``None`` → ``None``.
    - Python scalar / Z3 expression → broadcast to every lane.
    - list/tuple of length ``n_lanes`` → indexed lookup.
    - list/tuple of length 1 → broadcast.
    - any other shape → :class:`UnsupportedSymbolicRaceQuery`.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == n_lanes:
            return value[lane]
        if len(value) == 1:
            return value[0]
        raise UnsupportedSymbolicRaceQuery(
            f"lane_value: vector of length {len(value)} cannot be aligned with "
            f"n_lanes={n_lanes}"
        )
    return value


def iter_lane(value: Any, lane: int, n_lanes: int) -> tuple[Any, ...]:
    """Return per-lane constraints flattened into a tuple."""
    return tuple(iter_constraints(lane_value(value, lane, n_lanes)))


def apply_sub(value: Any, substitutions: tuple[tuple[Any, Any], ...]) -> Any:
    """Recursively apply Z3 substitutions through ``None`` / list / tuple / scalar."""
    if value is None:
        return None
    if isinstance(value, list):
        return [apply_sub(v, substitutions) for v in value]
    if isinstance(value, tuple):
        return tuple(apply_sub(v, substitutions) for v in value)
    if isinstance(value, (bool, int)):
        return value
    if not substitutions:
        return value
    return substitute(value, *substitutions)


def is_release_sem(sem: Any) -> bool:
    return sem in ("release", "acq_rel")


def is_acquire_sem(sem: Any) -> bool:
    return sem in ("acquire", "acq_rel")


def build_transitive_hb(
    events: list[Any],
    edge_fn: Callable[[Any, Any], BoolRef],
) -> list[list[BoolRef]]:
    """Floyd-Warshall transitive closure over symbolic HB edges."""
    n = len(events)
    reach = [[edge_fn(events[i], events[j]) for j in range(n)] for i in range(n)]
    for k in range(n):
        reach = [
            [Or(reach[i][j], And(reach[i][k], reach[k][j])) for j in range(n)]
            for i in range(n)
        ]
    return reach


def conflicting_access_modes(first: Any, second: Any) -> BoolRef:
    """``(write,read|write)`` conflict ∧ at-least-one-non-atomic.

    Preserves the rule that atomic-vs-atomic never races.
    Built with explicit ``And``/``Or`` to avoid Python operator-precedence
    pitfalls between Z3 expressions and Python booleans.
    """
    access_conflict = Or(
        And(first.writes, Or(second.reads, second.writes)),
        And(second.writes, Or(first.reads, first.writes)),
    )
    at_least_one_non_atomic = BoolVal((not first.is_atomic) or (not second.is_atomic))
    return And(access_conflict, at_least_one_non_atomic)


def minimal_atomic_read_from(
    writer: Any,
    reader: Any,
    *,
    same_atomic_addr_fn: Callable[[Any, Any], BoolRef],
) -> BoolRef:
    """Minimal CAS read-from predicate.

    CAS-only — generic RMW must not participate in CAS-style synchronisation.
    The address predicate is supplied by the caller (single-copy uses simple
    ``addr ==``; two-copy uses ``addr ==`` plus matching ``elem_size``).
    """
    if writer.atomic_kind != "cas" or reader.atomic_kind != "cas":
        return BoolVal(False)
    if not writer.is_atomic or not reader.is_atomic:
        return BoolVal(False)
    if writer.written_value is None or reader.old_value is None:
        return BoolVal(False)
    return And(
        writer.writes,
        reader.reads,
        same_atomic_addr_fn(writer, reader),
        writer.written_value == reader.old_value,
    )


def normalize_copy_local_vars(values: Iterable[Any]) -> tuple[Any, ...]:
    """Flatten and dedup copy-local Z3 vars.

    Dedup key is ``(v.hash(), str(v.sort()), v.decl().name())`` rather than
    ``v.hash()`` alone, so distinct vars that happen to share a hash are not
    collapsed. ``None`` and Python scalars are skipped — only Z3 vars are
    retained.
    """
    out: list[Any] = []
    seen: set[tuple[int, str, str]] = set()

    def visit(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
            return
        if isinstance(v, (bool, int)):
            return
        try:
            key = (v.hash(), str(v.sort()), v.decl().name())
        except Exception:
            return
        if key in seen:
            return
        seen.add(key)
        out.append(v)

    for value in values:
        visit(value)
    return tuple(out)


__all__ = [
    "UnsupportedSymbolicRaceQuery",
    "apply_sub",
    "as_bool",
    "build_transitive_hb",
    "conflicting_access_modes",
    "is_acquire_sem",
    "is_release_sem",
    "iter_constraints",
    "iter_lane",
    "lane_value",
    "minimal_atomic_read_from",
    "normalize_copy_local_vars",
    "to_lanes",
]
