"""Step 4 candidate-finder invariants, tested in isolation.

These tests build ``AccessEventRecord`` instances by hand so they cover
the pairing logic without kernel-launch overhead or interpreter behavior
drift. End-to-end coverage lives in ``tests/end_to_end/test_race_detector_sync.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_viz.clients.race_detector.data import (
    AccessEventRecord,
    RaceType,
    flatten_np,
)
from triton_viz.clients.race_detector.race_detector import (
    _canonical_pair,
    _classify_candidate,
    SymbolicRaceDetector,
)
from triton_viz.core.data import Load, Store


def _plain_event(
    *,
    event_id: int,
    grid_idx: tuple[int, ...],
    program_seq: int,
    lane_addrs: list[int],
    elem_size: int = 4,
    access_mode: str,
    epoch: int = 0,
) -> AccessEventRecord:
    lane_arr = np.array(lane_addrs, dtype=np.int64)
    active = np.ones(len(lane_addrs), dtype=bool)
    if access_mode == "read":
        read_mask = active.copy()
        write_mask = np.zeros_like(active)
        op_type = Load
    else:
        read_mask = np.zeros_like(active)
        write_mask = active.copy()
        op_type = Store
    return AccessEventRecord(
        event_id=event_id,
        op_type=op_type,
        grid_idx=grid_idx,
        program_seq=program_seq,
        lane_addrs=lane_arr,
        active_mask=active,
        elem_size=elem_size,
        read_mask=read_mask,
        write_mask=write_mask,
        epoch=epoch,
    )


class _Concretizable:
    """Minimal SymbolicExpr-alike with a ``concretize()`` method so
    ``maybe_concretize`` has something to dispatch on."""

    def __init__(self, inner: np.ndarray):
        self._inner = inner

    def concretize(self) -> np.ndarray:
        return self._inner


def test_flatten_np_concretizes_symbolic_expr():
    """Locks the Step 0 plumbing: helpers must unwrap a SymbolicExpr-alike
    before treating it as numeric data. Upstream #358 xfailed atomic E2E
    tests over this exact gap."""
    concrete = np.array([10, 20, 30], dtype=np.int64)
    wrapped = _Concretizable(concrete)
    np.testing.assert_array_equal(flatten_np(wrapped), concrete)


def test_classify_race_type_directional_raw_war_waw():
    """WAR must be a live return value, not a dead enum entry (#351).

    Fixed canonical order places the write-first event as ``first``; the
    WAR case has ``first=reader, second=writer`` by canonical order — so
    the RAW/WAR distinction actually falls out of the ordering.
    """
    w0 = _plain_event(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    r1 = _plain_event(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="read",
    )
    # Canonical order: block 0 precedes block 1. first=w0, second=r1 -> RAW.
    first, second = _canonical_pair(w0, r1)
    assert first is w0 and second is r1
    assert _classify_candidate(first, second, 100) is RaceType.RAW

    # Swap grid order to produce a WAR: reader in block 0, writer in block 1.
    r0 = _plain_event(
        event_id=3,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="read",
    )
    w1 = _plain_event(
        event_id=4,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    first, second = _canonical_pair(r0, w1)
    assert first is r0 and second is w1
    assert _classify_candidate(first, second, 100) is RaceType.WAR

    # Both write -> WAW.
    w0b = _plain_event(
        event_id=5,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    w1b = _plain_event(
        event_id=6,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    assert _classify_candidate(w0b, w1b, 100) is RaceType.WAW

    # Both read -> None.
    r0b = _plain_event(
        event_id=7,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="read",
    )
    r1b = _plain_event(
        event_id=8,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="read",
    )
    assert _classify_candidate(r0b, r1b, 100) is None


def test_candidate_finder_skips_same_grid_idx():
    """Intra-block pairs are po-ordered by construction, not a race. The
    finder must drop them before classification."""
    detector = SymbolicRaceDetector(abort_on_error=False)

    e_w = _plain_event(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    e_r = _plain_event(
        event_id=2,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[100],
        access_mode="read",
    )
    assert detector._find_race_candidates([e_w, e_r]) == []


def test_candidate_finder_dedups_same_pair_seen_on_multiple_bytes():
    """A 4-byte element-size overlap produces 4 byte-bucket hits for the
    same event pair — the finder must dedupe on ``(min_event_id,
    max_event_id)`` so one RaceCandidate emerges, not four."""
    detector = SymbolicRaceDetector(abort_on_error=False)

    w0 = _plain_event(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        elem_size=4,
        access_mode="write",
    )
    r1 = _plain_event(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        elem_size=4,
        access_mode="read",
    )
    cands = detector._find_race_candidates([w0, r1])
    assert len(cands) == 1
    assert cands[0].race_type is RaceType.RAW


def test_candidate_finder_respects_epoch():
    """Events in different epochs never pair, even on the same address.

    Step 6 barrier-based partitioning relies on this — otherwise cross-
    phase pairs would still show up as races.
    """
    detector = SymbolicRaceDetector(abort_on_error=False)

    pre = _plain_event(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
        epoch=0,
    )
    post = _plain_event(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="read",
        epoch=1,
    )
    assert detector._find_race_candidates([pre, post]) == []

    # Same epoch -> pair is produced.
    pre.epoch = 0
    post.epoch = 0
    cands = detector._find_race_candidates([pre, post])
    assert len(cands) == 1
    assert cands[0].epoch == 0


def test_canonical_pair_is_stable_under_swap():
    a = _plain_event(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    b = _plain_event(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="read",
    )
    first_ab, second_ab = _canonical_pair(a, b)
    first_ba, second_ba = _canonical_pair(b, a)
    assert (first_ab, second_ab) == (first_ba, second_ba)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
