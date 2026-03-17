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

from .data import MemoryAccess, SymbolicMemoryAccess, AccessType


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


class SymbolicHBSolver:
    """HB solver for symbolic accesses.

    Models two copies of the program (block A and block B) since the symbolic
    trace is a template executed by all blocks. po edges are within each copy,
    sw edges are between copies.

    Uses symbolic ptr signatures to determine address overlap for sw edges.
    Like HBSolver, uses unconditional sw for qualifying release/acquire pairs.
    """

    def __init__(
        self,
        acc_a: SymbolicMemoryAccess,
        acc_b: SymbolicMemoryAccess,
        all_accesses: list[SymbolicMemoryAccess],
        ptr_signature_fn,
    ):
        self._acc_a = acc_a
        self._acc_b = acc_b
        self._all_accesses = all_accesses
        self._ptr_sig = ptr_signature_fn

    def check_race_possible(self) -> bool:
        """Return True if a race is possible (a and b unordered).

        False means sw+po edges create an hb path between a and b.
        """
        a = self._acc_a
        b = self._acc_b

        # Collect sync accesses in same epoch
        sync = [
            acc
            for acc in self._all_accesses
            if acc.access_type == AccessType.ATOMIC
            and acc.epoch == a.epoch
            and acc.atomic_sem is not None
        ]

        if not sync:
            return True  # no sync events → can't prove ordering

        # Build all accesses in the epoch (for po chains)
        epoch_accesses = sorted(
            [acc for acc in self._all_accesses if acc.epoch == a.epoch],
            key=lambda x: x.event_id,
        )

        # Create two copies: block A nodes (indices 0..n-1), block B nodes (n..2n-1)
        n = len(epoch_accesses)
        acc_to_idx = {id(acc): i for i, acc in enumerate(epoch_accesses)}

        adj: dict[int, set[int]] = defaultdict(set)

        # po edges within block A (indices 0..n-1)
        for k in range(n - 1):
            adj[k].add(k + 1)

        # po edges within block B (indices n..2n-1)
        for k in range(n - 1):
            adj[n + k].add(n + k + 1)

        # sw edges: from block A's release to block B's acquire (and vice versa)
        sync_sigs = {}
        for acc in sync:
            sync_sigs[id(acc)] = self._ptr_sig(acc.ptr_expr)

        for w in sync:
            if not _release_like(w.atomic_sem):
                continue
            w_sig = sync_sigs[id(w)]
            w_idx_a = acc_to_idx[id(w)]  # block A copy
            w_idx_b = n + acc_to_idx[id(w)]  # block B copy

            for r in sync:
                if id(w) == id(r):
                    continue
                if not _acquire_like(r.atomic_sem):
                    continue
                if not _scope_ok(w.atomic_scope, r.atomic_scope):
                    continue
                r_sig = sync_sigs[id(r)]
                if w_sig != r_sig:
                    continue

                r_idx_a = acc_to_idx[id(r)]  # block A copy
                r_idx_b = n + acc_to_idx[id(r)]  # block B copy

                # sw: block A's release → block B's acquire
                adj[w_idx_a].add(r_idx_b)
                # sw: block B's release → block A's acquire
                adj[w_idx_b].add(r_idx_a)

        # acc_a is in block A, acc_b is in block B
        ai = acc_to_idx[id(a)]
        bi = n + acc_to_idx[id(b)]

        if _reachable(adj, ai, bi) or _reachable(adj, bi, ai):
            return False  # ordered → no race

        return True  # unordered → race possible
