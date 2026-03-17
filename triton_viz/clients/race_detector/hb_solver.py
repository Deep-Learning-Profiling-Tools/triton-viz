"""Local happens-before solver for race detection.

Uses graph reachability over po (program order) and sw (synchronizes-with)
edges to determine whether a candidate race pair (a, b) must be ordered
in all valid executions.

v1 uses a conservative model: any structurally valid release/acquire pair
on the same sync address with compatible scope unconditionally creates a
sw edge. This is sound in the "fewer false positives" direction — it may
suppress races that could theoretically occur if the acquire doesn't
actually read from the release, but in practice synchronization patterns
are designed so that the acquire does see the release's write.
"""

from __future__ import annotations

from collections import defaultdict, deque

from .data import MemoryAccess


def _release_like(sem: str | None) -> bool:
    return sem in ("release", "acq_rel")


def _acquire_like(sem: str | None) -> bool:
    return sem in ("acquire", "acq_rel")


def _scope_ok(w_scope: str | None, r_scope: str | None) -> bool:
    """Check if scopes are compatible for cross-block synchronization.

    Only gpu or sys scopes can form cross-block sw edges.
    cta scope is thread-block local and cannot synchronize across blocks.
    """
    return (w_scope in ("gpu", "sys")) and (r_scope in ("gpu", "sys"))


def _active_addrs(acc: MemoryAccess) -> set[int]:
    return {int(o) for o, m in zip(acc.offsets, acc.masks) if m}


class HBSolver:
    """Local happens-before solver for a candidate race pair.

    acc_a and acc_b are always included as nodes in the HB graph
    (they are the po endpoints that need ordering). Sync events are
    the atomic accesses that may create sw edges between blocks.
    """

    def __init__(self, acc_a: MemoryAccess, acc_b: MemoryAccess):
        self._acc_a = acc_a
        self._acc_b = acc_b
        self._sync_events: list[MemoryAccess] = []

    def add_sync_events(self, events: list[MemoryAccess]) -> None:
        """Add atomic events from same epoch, all blocks, per sync address.

        Must NOT include legacy_atomic accesses.
        """
        self._sync_events = [e for e in events if not e.legacy_atomic]

    def check_race_possible(self) -> bool:
        """Return True if a race is possible (a and b unordered).

        False means all valid executions order a and b (suppress race).
        """
        a = self._acc_a
        b = self._acc_b
        sync = self._sync_events

        # Build graph: nodes are all accesses (a, b, + sync events)
        all_nodes = [a, b] + sync
        node_ids = {id(n): i for i, n in enumerate(all_nodes)}

        # Adjacency list for HB edges
        adj: dict[int, set[int]] = defaultdict(set)

        # ─── Program order (po) edges ───
        by_block: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for node in all_nodes:
            idx = node_ids[id(node)]
            by_block[node.grid_idx].append(idx)

        for grid_idx, indices in by_block.items():
            sorted_indices = sorted(indices, key=lambda i: all_nodes[i].event_id)
            for k in range(len(sorted_indices) - 1):
                adj[sorted_indices[k]].add(sorted_indices[k + 1])

        # ─── Synchronizes-with (sw) edges ───
        # For v1: unconditional sw for any qualifying release/acquire pair
        sync_indices = [node_ids[id(e)] for e in sync]

        for wi in sync_indices:
            w = all_nodes[wi]
            if not _release_like(w.atomic_sem):
                continue
            w_addrs = _active_addrs(w)
            for ri in sync_indices:
                if wi == ri:
                    continue
                r = all_nodes[ri]
                if w.grid_idx == r.grid_idx:
                    continue  # sw is cross-block only
                if not _acquire_like(r.atomic_sem):
                    continue
                if not _scope_ok(w.atomic_scope, r.atomic_scope):
                    continue
                r_addrs = _active_addrs(r)
                if w_addrs & r_addrs:
                    adj[wi].add(ri)

        # ─── Reachability check ───
        ai = node_ids[id(a)]
        bi = node_ids[id(b)]

        if _reachable(adj, ai, bi) or _reachable(adj, bi, ai):
            return False  # ordered → no race possible

        return True  # unordered → race possible


def _reachable(adj: dict[int, set[int]], src: int, dst: int) -> bool:
    """BFS reachability check."""
    visited: set[int] = set()
    queue: deque[int] = deque([src])
    visited.add(src)
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        for neighbor in adj.get(node, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
