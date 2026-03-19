"""End-to-end integration tests for atomic synchronization race detection.

These tests construct concrete MemoryAccess sequences that mimic what Triton
kernels would produce, and verify the full detect_races pipeline including
epoch annotations, effect-based classification, and HB solver integration.
"""

import numpy as np

from triton_viz.clients.race_detector.data import (
    AccessType,
    MemoryAccess,
    RaceType,
)
from triton_viz.clients.race_detector.race_detector import detect_races


def _store(block_idx, offset, event_id):
    return MemoryAccess(
        access_type=AccessType.STORE,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(block_idx, 0, 0),
        event_id=event_id,
    )


def _load(block_idx, offset, event_id):
    return MemoryAccess(
        access_type=AccessType.LOAD,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(block_idx, 0, 0),
        event_id=event_id,
    )


def _atomic(
    block_idx,
    offset,
    event_id,
    atomic_op="rmw:add",
    sem="acq_rel",
    scope="gpu",
    read_mask=None,
    write_mask=None,
    atomic_val=None,
    atomic_old=None,
    atomic_cmp=None,
):
    masks = np.array([True], dtype=np.bool_)
    return MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=masks,
        grid_idx=(block_idx, 0, 0),
        event_id=event_id,
        atomic_op=atomic_op,
        atomic_sem=sem,
        atomic_scope=scope,
        read_mask=read_mask if read_mask is not None else masks.copy(),
        write_mask=write_mask if write_mask is not None else masks.copy(),
        atomic_val=atomic_val,
        atomic_old=atomic_old,
        atomic_cmp=atomic_cmp,
    )


# ── Producer-consumer release/acquire ──


def test_producer_consumer_release_acquire_no_race():
    """Producer writes payload + release flag; consumer acquire flag + load payload.

    The HB solver should suppress the race via:
    store_data ->po-> release_flag ->sw-> acquire_flag ->po-> load_data
    """
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0 (producer): store data, then release flag (writes 1)
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1 (consumer): acquire flag (reads old=1), then load data
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) == 0


def test_producer_consumer_relaxed_still_races():
    """Same pattern but sem='relaxed' — no sw edge, race should be reported."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="relaxed",
            scope="gpu",
        ),
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="relaxed",
            scope="gpu",
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) > 0
    assert any(r.race_type == RaceType.RAW for r in races)


# ── CAS spinlock ──


def test_cas_spinlock_protects_payload_no_race():
    """CAS spinlock guarding shared data.

    Block 0: cas_lock(acquire) -> store data -> xchg_unlock(release)
    Block 1: cas_lock(acquire) -> load data -> xchg_unlock(release)

    The HB chain:
    store_data ->po-> xchg_unlock(release) ->sw-> cas_lock(acquire) ->po-> load_data
    """
    DATA_ADDR = 100
    LOCK_ADDR = 200

    accesses = [
        # Block 0: acquire lock (CAS 0→1), write data, release lock (xchg→0)
        _atomic(
            block_idx=0,
            offset=LOCK_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
            atomic_cmp=np.array([0], dtype=np.int32),
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        _store(block_idx=0, offset=DATA_ADDR, event_id=1),
        _atomic(
            block_idx=0,
            offset=LOCK_ADDR,
            event_id=2,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        # Block 1: acquire lock (CAS 0→1, reads old=0 from block 0's unlock), read data, release lock
        _atomic(
            block_idx=1,
            offset=LOCK_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
            atomic_cmp=np.array([0], dtype=np.int32),
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
        _atomic(
            block_idx=1,
            offset=LOCK_ADDR,
            event_id=2,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
    ]

    races = detect_races(accesses)
    assert len(races) == 0


# ── Failed CAS mixed with plain access ──


def test_failed_cas_mixed_with_plain_access():
    """Failed CAS (read-only) at addr X vs plain load at addr X → no conflict."""
    ADDR = 100

    accesses = [
        # Block 0: failed CAS (write_mask=False)
        MemoryAccess(
            access_type=AccessType.ATOMIC,
            ptr=0,
            offsets=np.array([ADDR], dtype=np.int64),
            masks=np.array([True], dtype=np.bool_),
            grid_idx=(0, 0, 0),
            event_id=0,
            atomic_op="cas",
            atomic_sem="acq_rel",
            atomic_scope="gpu",
            read_mask=np.array([True], dtype=np.bool_),
            write_mask=np.array([False], dtype=np.bool_),
        ),
        # Block 1: plain load
        _load(block_idx=1, offset=ADDR, event_id=0),
    ]

    races = detect_races(accesses)
    assert len(races) == 0  # read-read → no race


# ── Atomic xchg handoff ──


def test_atomic_xchg_handoff():
    """Atomic xchg unlock/handoff pattern with release/acquire semantics."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0: store data, then xchg flag with release (writes 1)
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1: xchg flag with acquire (reads old=1), then store data
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        _store(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) == 0  # ordered via release/acquire handoff


# ── Existing behavior preserved ──


def test_atomic_histogram_still_no_race():
    """Two atomic_add at same addr → no race (preserved behavior)."""
    ADDR = 100

    accesses = [
        _atomic(
            block_idx=0,
            offset=ADDR,
            event_id=0,
            atomic_op="rmw:add",
            sem="relaxed",
            scope="gpu",
        ),
        _atomic(
            block_idx=1,
            offset=ADDR,
            event_id=0,
            atomic_op="rmw:add",
            sem="relaxed",
            scope="gpu",
        ),
    ]

    races = detect_races(accesses)
    assert len(races) == 0  # atomic-atomic → suppressed


def test_epoch_barrier_still_works():
    """The add+cas epoch/barrier pattern is preserved."""
    BARRIER_ADDR = 4096

    accesses = []
    for i in range(4):
        # Phase 0: block i writes slot i
        accesses.append(_store(block_idx=i, offset=i, event_id=i))
        # Barrier: add (release) writes count, cas (acquire) reads count
        # Each block's add writes i+1, the next block's cas reads that value
        accesses.append(
            _atomic(
                block_idx=i,
                offset=BARRIER_ADDR,
                event_id=100 + i,
                atomic_op="rmw:add",
                sem="acq_rel",
                scope="gpu",
                atomic_val=np.array([1], dtype=np.int32),
                atomic_old=np.array([i], dtype=np.int32),
            )
        )
        accesses.append(
            _atomic(
                block_idx=i,
                offset=BARRIER_ADDR,
                event_id=200 + i,
                atomic_op="cas",
                sem="acq_rel",
                scope="gpu",
                atomic_cmp=np.array([4], dtype=np.int32),
                atomic_val=np.array([4], dtype=np.int32),
                atomic_old=np.array([i + 1], dtype=np.int32),
                write_mask=np.array([i + 1 == 4], dtype=np.bool_),
            )
        )
        # Phase 1: block i writes slot (i+1) % 4
        accesses.append(_store(block_idx=i, offset=(i + 1) % 4, event_id=300 + i))

    races = detect_races(accesses)
    assert len(races) == 0  # cross-phase accesses separated by barrier


# ── Regression tests for correctness bugs ──


def test_duplicate_lane_same_address_aggregation():
    """Multi-lane CAS (lane 0 fail, lane 1 success) + plain store → WAW race found.

    Regression: effects_at_addr() previously only checked lane_indices[0],
    missing the write from lane 1. This locks Issue 1 (single-lane bug).
    """
    ADDR = 100

    # Block 0: 2-lane CAS at same addr, lane 0 fails, lane 1 succeeds
    cas_access = MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([ADDR, ADDR], dtype=np.int64),
        masks=np.array([True, True], dtype=np.bool_),
        grid_idx=(0, 0, 0),
        event_id=0,
        atomic_op="cas",
        atomic_sem="relaxed",
        atomic_scope="gpu",
        read_mask=np.array([True, True], dtype=np.bool_),
        write_mask=np.array([False, True], dtype=np.bool_),
    )

    # Block 1: plain store at same addr
    store = _store(block_idx=1, offset=ADDR, event_id=0)

    races = detect_races([cas_access, store])
    assert len(races) > 0
    assert any(r.race_type == RaceType.WAW for r in races)


def test_failed_acquire_no_sw_edge():
    """Acquire CAS reads wrong old value → race reported despite release/acquire structure.

    Regression: HBSolver previously created sw edges unconditionally for any
    release/acquire pair on the same address. This locks Issue 2 (unconditional sw).
    """
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0: store data, then release flag (writes 1)
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1: acquire CAS reads old=42 (NOT 1) → no reads-from → no sw
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
            atomic_cmp=np.array([1], dtype=np.int32),
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([42], dtype=np.int32),
            read_mask=np.array([True], dtype=np.bool_),
            write_mask=np.array([False], dtype=np.bool_),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) > 0  # no sw → race reported
    assert any(r.race_type == RaceType.RAW for r in races)


def test_per_block_private_sync_no_spurious_suppress():
    """Two blocks with different flag addresses → no sw edge, race reported.

    Regression: SymbolicHBSolver previously matched by ptr signature equality,
    which could falsely unify distinct addresses. This locks Issue 3
    (ptr signature unsoundness). With the concrete path, different addresses
    naturally produce no sw edge overlap.
    """
    DATA_ADDR = 100
    FLAG_ADDR_0 = 200  # block 0's private flag
    FLAG_ADDR_1 = 300  # block 1's private flag (different!)

    accesses = [
        # Block 0: store data, then release its own flag
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR_0,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1: acquire its own (different) flag, then load data
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR_1,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) > 0  # different flag addrs → no sw → race
    assert any(r.race_type == RaceType.RAW for r in races)


def test_aba_same_value_no_spurious_suppress_e2e():
    """Two writers write same value → ambiguous reads-from → race reported.

    Regression: ABA pattern where value equality is insufficient to identify
    the writer. sw edge must not be created when written value is not unique.
    """
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0: store data, then release xchg writes 1
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 2: relaxed xchg also writes 1 (same value!)
        _atomic(
            block_idx=2,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="relaxed",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1: acquire reads old=1 → ambiguous (could be from either writer)
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) > 0  # ambiguous reads-from → race


def test_producer_consumer_release_xchg_acquire_cas_same_value_poll_no_race():
    """Acquire CAS polling with same-value writeback must not block sw edge.

    B0 stores data + release xchg(1); B1 acquire cas(1,1) reads old=1 + writes back 1.
    Reader's own writeback must be excluded from the ambiguity scan.
    """
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0 (producer): store data, then release xchg writes 1
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1 (consumer): acquire cas(1,1) reads old=1, writes back 1
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
            atomic_cmp=np.array([1], dtype=np.int32),
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) == 0


def test_acquire_cas_same_value_poll_still_races_with_third_party_same_value_writer():
    """Third-party same-value writer makes reads-from ambiguous despite CAS polling.

    B0 release xchg(1), B2 relaxed xchg(1), B1 acquire cas(1,1) reads old=1.
    Third-party writer means we can't tell who wrote the 1 → ambiguous → race.
    """
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0 (producer): store data, then release xchg writes 1
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 2 (interloper): relaxed xchg also writes 1
        _atomic(
            block_idx=2,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="relaxed",
            scope="gpu",
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([0], dtype=np.int32),
        ),
        # Block 1 (consumer): acquire cas(1,1) reads old=1, writes back 1
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
            atomic_cmp=np.array([1], dtype=np.int32),
            atomic_val=np.array([1], dtype=np.int32),
            atomic_old=np.array([1], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) > 0


def test_add_written_value_correctness_e2e():
    """add(5) with old=10 writes 15; acquire reads old=5 (operand) → no reads-from → race.

    Regression: _written_value_at_addr previously returned atomic_val (operand)
    instead of computing old+val for add ops.
    """
    DATA_ADDR = 100
    FLAG_ADDR = 200

    accesses = [
        # Block 0: store data, then release add(5) with old=10 → writes 15
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:add",
            sem="release",
            scope="gpu",
            atomic_val=np.array([5], dtype=np.int32),
            atomic_old=np.array([10], dtype=np.int32),
        ),
        # Block 1: acquire reads old=5 (the operand, NOT the written value 15)
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
            atomic_val=np.array([0], dtype=np.int32),
            atomic_old=np.array([5], dtype=np.int32),
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
    ]

    races = detect_races(accesses)
    assert len(races) > 0  # no reads-from (5 ≠ 15) → race
