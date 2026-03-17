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
        # Block 0 (producer): store data, then release flag
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
        ),
        # Block 1 (consumer): acquire flag, then load data
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
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
        # Block 0: acquire lock, write data, release lock
        _atomic(
            block_idx=0,
            offset=LOCK_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
        ),
        _store(block_idx=0, offset=DATA_ADDR, event_id=1),
        _atomic(
            block_idx=0,
            offset=LOCK_ADDR,
            event_id=2,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
        ),
        # Block 1: acquire lock, read data, release lock
        _atomic(
            block_idx=1,
            offset=LOCK_ADDR,
            event_id=0,
            atomic_op="cas",
            sem="acquire",
            scope="gpu",
        ),
        _load(block_idx=1, offset=DATA_ADDR, event_id=1),
        _atomic(
            block_idx=1,
            offset=LOCK_ADDR,
            event_id=2,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
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
        # Block 0: store data, then xchg flag with release
        _store(block_idx=0, offset=DATA_ADDR, event_id=0),
        _atomic(
            block_idx=0,
            offset=FLAG_ADDR,
            event_id=1,
            atomic_op="rmw:xchg",
            sem="release",
            scope="gpu",
        ),
        # Block 1: xchg flag with acquire, then store data
        _atomic(
            block_idx=1,
            offset=FLAG_ADDR,
            event_id=0,
            atomic_op="rmw:xchg",
            sem="acquire",
            scope="gpu",
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
        # Barrier: add + cas
        accesses.append(
            _atomic(
                block_idx=i,
                offset=BARRIER_ADDR,
                event_id=100 + i,
                atomic_op="rmw:add",
                sem="acq_rel",
                scope="gpu",
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
            )
        )
        # Phase 1: block i writes slot (i+1) % 4
        accesses.append(_store(block_idx=i, offset=(i + 1) % 4, event_id=300 + i))

    races = detect_races(accesses)
    assert len(races) == 0  # cross-phase accesses separated by barrier
