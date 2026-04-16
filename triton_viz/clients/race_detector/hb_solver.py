"""Concrete happens-before solver for the race detector.

Builds a per-epoch graph over concrete events:
  * ``po`` edges - per ``grid_idx``, consecutive events ordered by
    ``program_seq`` (upstream #346 fix: event_id is not program order
    under multi-SM concurrency).
  * ``sw`` edges - only between atomic operations where:
      * writer.atomic_sem is release or acq_rel;
      * reader.atomic_sem is acquire or acq_rel;
      * writer.atomic_scope and reader.atomic_scope are in
        ``{gpu, sys}`` - CTA scope is block-local and can't be inferred
        as cross-block sync (upstream #345);
      * reader.atomic_old equals writer.written_value at a shared
        element base address (reads-from gate, upstream #345 - naive
        release/acquire pairing over-suppresses);
      * no ``same-value`` ambiguous writer exists (upstream #350 same-value
        ABA - if a third-party writer also wrote the same value, the
        reads-from evidence is inconclusive).

Given a ``RaceCandidate``, ``check_candidate`` returns ``HBResult`` with
``ordered=True`` when either direction is reachable, else ``ordered=False``
(the candidate survives HB suppression and becomes a race report).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .data import AccessEventRecord, RaceCandidate


_RELEASE_WRITER_SEMS: frozenset[str] = frozenset({"release", "acq_rel"})
_ACQUIRE_READER_SEMS: frozenset[str] = frozenset({"acquire", "acq_rel"})
_VALID_SW_SCOPES: frozenset[str] = frozenset({"gpu", "sys"})


@dataclass
class HBResult:
    """Outcome of an HB reachability query.

    ``ordered=True`` means the candidate pair is ordered by
    ``po | sw`` - the race is suppressed. ``reason`` identifies the edge
    class used so race reports (and tests) can distinguish ``po_path``
    from ``release_acquire_reads_from`` suppression.
    """

    ordered: bool
    reason: str | None = None
    sync_addr: int | None = None


def _iter_element_bases(event: AccessEventRecord):
    """Yield element base addresses for every active lane of ``event``.

    Unlike the candidate finder's byte iterator, sw edges are per-element
    (release/acquire operate on the full element), so the HB solver keys
    on element base addresses - not byte addresses.
    """
    if event.lane_addrs is None:
        return
    active = (
        event.active_mask
        if event.active_mask is not None
        else np.ones_like(event.lane_addrs, dtype=bool)
    )
    for idx, hit in enumerate(active):
        if not hit:
            continue
        yield int(event.lane_addrs[idx])


def _value_at_addr(event: AccessEventRecord, elem_base: int, which: str) -> Any | None:
    """Return ``event.atomic_old`` or ``event.written_value`` at a specific
    element base, or ``None`` if the event has no atomic data or does not
    touch that address on an active lane.
    """
    arr = getattr(event, which)
    if arr is None or event.lane_addrs is None:
        return None
    mask = event.lane_addrs == elem_base
    if event.active_mask is not None:
        mask = mask & event.active_mask
    hits = np.nonzero(mask)[0]
    if hits.size == 0:
        return None
    return arr[hits[0]]


def _reads_from(
    writer: AccessEventRecord, reader: AccessEventRecord, elem_base: int
) -> bool:
    """True iff reader's ``atomic_old`` at ``elem_base`` equals writer's
    ``written_value`` at the same address. This is the reads-from gate
    that keeps release/acquire pairs from over-suppressing - just
    matching sem/scope is not enough (upstream #345).
    """
    w_val = _value_at_addr(writer, elem_base, "written_value")
    r_old = _value_at_addr(reader, elem_base, "atomic_old")
    if w_val is None or r_old is None:
        return False
    return bool(np.array_equal(w_val, r_old))


def _has_ambiguous_writer(
    wi: AccessEventRecord,
    ri: AccessEventRecord,
    elem_base: int,
    sync_events: Sequence[AccessEventRecord],
) -> bool:
    """Return True if another writer in the same epoch wrote the same value
    at ``elem_base``. With matching values from different writers, the
    reads-from gate can't uniquely attribute ``ri.atomic_old`` to ``wi``;
    sound behavior is to block the sw edge (upstream #350).

    Skips ``wi`` itself (obvious), ``ri`` itself (``cas(1,1)`` polling
    loops writeback the same value - upstream #347), and events strictly
    in ``ri``'s future within ``ri``'s own grid (reader hasn't executed
    past its own program point yet so those aren't visible).
    """
    wi_val = _value_at_addr(wi, elem_base, "written_value")
    if wi_val is None:
        return False
    for x in sync_events:
        if x is wi or x is ri:
            continue
        if (
            x.grid_idx == ri.grid_idx
            and x.program_seq is not None
            and (ri.program_seq is None or x.program_seq > ri.program_seq)
        ):
            continue
        x_val = _value_at_addr(x, elem_base, "written_value")
        if x_val is None:
            continue
        if np.array_equal(x_val, wi_val):
            return True
    return False


class HBSolver:
    """Concrete ``po + sw`` reachability solver for one epoch.

    ``events`` must all share the same ``epoch`` - the caller groups
    events by epoch and builds one solver per group (see
    ``SymbolicRaceDetector.finalize``).
    """

    def __init__(self, events: Sequence[AccessEventRecord]):
        self._events = list(events)
        # Adjacency keyed on id(event) so two events that happen to
        # compare equal stay distinct nodes.
        self._adj: dict[int, list[tuple[int, str, int | None]]] = defaultdict(list)
        self._build_po()
        self._build_sw()

    # ── Graph construction ────────────────────────────────────────────────

    def _add_edge(
        self,
        src: AccessEventRecord,
        dst: AccessEventRecord,
        reason: str,
        sync_addr: int | None = None,
    ) -> None:
        self._adj[id(src)].append((id(dst), reason, sync_addr))

    def _build_po(self) -> None:
        by_grid: dict[Any, list[AccessEventRecord]] = defaultdict(list)
        for ev in self._events:
            by_grid[ev.grid_idx].append(ev)
        for grid_events in by_grid.values():
            grid_events.sort(
                key=lambda e: (
                    e.program_seq if e.program_seq is not None else -1,
                    e.event_id,
                )
            )
            for i in range(len(grid_events) - 1):
                self._add_edge(grid_events[i], grid_events[i + 1], "po_path")

    def _build_sw(self) -> None:
        by_addr: dict[int, list[AccessEventRecord]] = defaultdict(list)
        sync_events: list[AccessEventRecord] = []
        for ev in self._events:
            if ev.atomic_op is None:
                continue
            sync_events.append(ev)
            for base in _iter_element_bases(ev):
                by_addr[base].append(ev)

        for addr, bucket in by_addr.items():
            for writer in bucket:
                if writer.atomic_sem not in _RELEASE_WRITER_SEMS:
                    continue
                if writer.atomic_scope not in _VALID_SW_SCOPES:
                    continue
                for reader in bucket:
                    if reader is writer:
                        continue
                    if reader.atomic_sem not in _ACQUIRE_READER_SEMS:
                        continue
                    if reader.atomic_scope not in _VALID_SW_SCOPES:
                        continue
                    if reader.grid_idx == writer.grid_idx:
                        # Same-block release/acquire is already po-ordered.
                        continue
                    if not _reads_from(writer, reader, addr):
                        continue
                    if _has_ambiguous_writer(writer, reader, addr, sync_events):
                        continue
                    self._add_edge(writer, reader, "release_acquire_reads_from", addr)

    # ── Reachability ──────────────────────────────────────────────────────

    def _reachable(
        self, src: AccessEventRecord, dst: AccessEventRecord
    ) -> tuple[bool, str | None, int | None]:
        """BFS from ``src``. Records the last sw edge traversed (if any) so
        ``check_candidate`` can annotate why the race was suppressed.
        Pure po paths return ``"po_path"``.
        """
        if src is dst:
            return True, "po_path", None
        start = id(src)
        target = id(dst)
        seen: set[int] = {start}
        queue: deque[tuple[int, str | None, int | None]] = deque([(start, None, None)])
        while queue:
            node, last_reason, last_addr = queue.popleft()
            for next_node, edge_reason, edge_addr in self._adj.get(node, ()):
                if next_node in seen:
                    continue
                new_reason = last_reason
                new_addr = last_addr
                if edge_reason != "po_path":
                    new_reason = edge_reason
                    new_addr = edge_addr
                if next_node == target:
                    return True, new_reason or "po_path", new_addr
                seen.add(next_node)
                queue.append((next_node, new_reason, new_addr))
        return False, None, None

    def check_candidate(self, cand: RaceCandidate) -> HBResult:
        ok_fwd, r_fwd, a_fwd = self._reachable(cand.first, cand.second)
        if ok_fwd:
            return HBResult(ordered=True, reason=r_fwd, sync_addr=a_fwd)
        ok_bwd, r_bwd, a_bwd = self._reachable(cand.second, cand.first)
        if ok_bwd:
            return HBResult(ordered=True, reason=r_bwd, sync_addr=a_bwd)
        return HBResult(ordered=False)
