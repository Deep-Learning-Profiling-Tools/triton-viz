"""Local happens-before solver for race detection.

Uses graph reachability over po (program order) and sw (synchronizes-with)
edges to determine whether a candidate race pair (a, b) must be ordered
in all valid executions.

sw edges are gated on reads-from: an acquire must observe the value
written by the release (via atomic_old matching) to establish ordering.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

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


def _rmw_op_suffix(atomic_op: str) -> str:
    """Extract op suffix from 'rmw:add' or 'rmw:atomic_op.add' format."""
    for sep in (".", ":"):
        if sep in atomic_op:
            atomic_op = atomic_op.rsplit(sep, 1)[-1]
    return atomic_op


def _compute_rmw_written(op_suffix: str, old, val):
    """Compute value written by an RMW op. Returns None for unknown ops."""
    if op_suffix == "xchg":
        return val
    elif op_suffix in ("add", "fadd"):
        return old + val
    elif op_suffix == "sub":
        return old - val
    elif op_suffix == "and":
        return old & val
    elif op_suffix == "or":
        return old | val
    elif op_suffix == "xor":
        return old ^ val
    elif op_suffix in ("max", "umax"):
        return max(old, val)
    elif op_suffix in ("min", "umin"):
        return min(old, val)
    return None


def _written_value_at_addr(acc: MemoryAccess, addr: int) -> object | None:
    """Return the value written by a sync op at addr, or None if no write."""
    lane_indices = np.where(acc.masks & (acc.offsets == addr))[0]
    if len(lane_indices) == 0 or acc.atomic_val is None:
        return None

    if acc.atomic_op == "cas":
        # CAS only writes on success (write_mask=True for that lane)
        for lane in lane_indices:
            if acc.write_mask is not None and acc.write_mask[lane]:
                return acc.atomic_val[lane]
        return None  # all CAS attempts failed
    else:
        # xchg / rmw:add etc. always write
        lane = lane_indices[0]
        if acc.atomic_op is None:
            return None
        op_suffix = _rmw_op_suffix(acc.atomic_op)
        if op_suffix == "xchg":
            return acc.atomic_val[lane]
        if acc.atomic_old is None:
            return None
        return _compute_rmw_written(op_suffix, acc.atomic_old[lane], acc.atomic_val[lane])


def _reads_from(writer: MemoryAccess, reader: MemoryAccess, addr: int) -> bool:
    """Return True if reader observed the value written by writer at addr."""
    written = _written_value_at_addr(writer, addr)
    if written is None:
        return False

    if reader.atomic_old is None:
        return False

    lane_indices = np.where(reader.masks & (reader.offsets == addr))[0]
    for lane in lane_indices:
        if reader.atomic_old[lane] == written:
            return True
    return False


def _has_ambiguous_writer(
    all_nodes, sync_indices, wi, ri, addr, written,
):
    """Check if another sync node visible to reader writes the same value."""
    r = all_nodes[ri]
    for oi in sync_indices:
        if oi == wi or oi == ri:          # skip candidate writer AND reader itself
            continue
        o = all_nodes[oi]
        if addr not in _active_addrs(o):
            continue
        ov = _written_value_at_addr(o, addr)
        if ov != written:
            continue
        # o writes same value — skip if po-after r (invisible)
        if o.grid_idx == r.grid_idx and o.event_id > r.event_id:
            continue
        return True
    return False


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
        # sw requires reads-from proof: acquire must observe release's write
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
                common_addrs = w_addrs & r_addrs
                if common_addrs:
                    for addr in common_addrs:
                        written = _written_value_at_addr(w, addr)
                        if written is None:
                            continue
                        if not _reads_from(w, r, addr):
                            continue
                        if not _has_ambiguous_writer(
                            all_nodes, sync_indices, wi, ri, addr, written,
                        ):
                            adj[wi].add(ri)
                            break

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
        """Return True -- symbolic HB cannot currently prove ordering.

        Sound suppression requires both:
        1. Must-alias proof (not just signature equality)
        2. Reads-from proof (atomic_old unavailable in symbolic accesses)

        Until the symbolic engine provides these, conservatively report all
        candidate races and let the concrete fallback handle suppression.
        """
        return True
