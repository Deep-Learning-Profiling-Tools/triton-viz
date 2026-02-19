import numpy as np

from triton_viz.clients.race_detector.data import AccessType, MemoryAccess, RaceType
from triton_viz.clients.race_detector.race_detector import (
    _apply_epoch_annotations,
    _detect_global_barrier_addresses,
    detect_races,
)


def _store_access(block_idx: int, offset: int, event_id: int) -> MemoryAccess:
    return MemoryAccess(
        access_type=AccessType.STORE,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(block_idx, 0, 0),
        event_id=event_id,
    )


def _atomic_access(
    block_idx: int, offset: int, event_id: int, atomic_op: str
) -> MemoryAccess:
    return MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(block_idx, 0, 0),
        event_id=event_id,
        atomic_op=atomic_op,
    )


def _two_phase_accesses_with_full_barrier(n: int = 4) -> list[MemoryAccess]:
    barrier_addr = 4096
    accesses: list[MemoryAccess] = []
    for i in range(n):
        accesses.append(_store_access(block_idx=i, offset=i, event_id=i))
        accesses.append(
            _atomic_access(
                block_idx=i,
                offset=barrier_addr,
                event_id=100 + i,
                atomic_op="rmw:add",
            )
        )
        accesses.append(
            _atomic_access(
                block_idx=i,
                offset=barrier_addr,
                event_id=200 + i,
                atomic_op="cas",
            )
        )
        accesses.append(
            _store_access(
                block_idx=i,
                offset=(i + 1) % n,
                event_id=300 + i,
            )
        )
    return accesses


def _three_phase_accesses_with_two_full_barriers(n: int = 4) -> list[MemoryAccess]:
    barrier_addr = 4096
    accesses: list[MemoryAccess] = []
    for i in range(n):
        # Phase 0
        accesses.append(_store_access(block_idx=i, offset=i, event_id=10 + i))
        # Barrier 1
        accesses.append(
            _atomic_access(
                block_idx=i,
                offset=barrier_addr,
                event_id=100 + i,
                atomic_op="rmw:add",
            )
        )
        accesses.append(
            _atomic_access(
                block_idx=i,
                offset=barrier_addr,
                event_id=200 + i,
                atomic_op="cas",
            )
        )
        # Phase 1
        accesses.append(
            _store_access(
                block_idx=i,
                offset=(i + 1) % n,
                event_id=300 + i,
            )
        )
        # Barrier 2
        accesses.append(
            _atomic_access(
                block_idx=i,
                offset=barrier_addr,
                event_id=400 + i,
                atomic_op="rmw:add",
            )
        )
        accesses.append(
            _atomic_access(
                block_idx=i,
                offset=barrier_addr,
                event_id=500 + i,
                atomic_op="cas",
            )
        )
        # Phase 2
        accesses.append(
            _store_access(
                block_idx=i,
                offset=(i + 2) % n,
                event_id=600 + i,
            )
        )
    return accesses


def test_detect_global_barrier_addresses_requires_add_and_cas_from_all_blocks():
    n = 4
    barrier_addr = 4096
    non_barrier_addr = 8192
    accesses = []

    for i in range(n):
        accesses.append(
            _atomic_access(i, barrier_addr, event_id=10 + i, atomic_op="rmw:add")
        )
        accesses.append(
            _atomic_access(i, barrier_addr, event_id=20 + i, atomic_op="cas")
        )

    # Partial signal on another address should not qualify as a global barrier.
    for i in range(2):
        accesses.append(
            _atomic_access(i, non_barrier_addr, event_id=30 + i, atomic_op="rmw:add")
        )
        accesses.append(
            _atomic_access(i, non_barrier_addr, event_id=40 + i, atomic_op="cas")
        )

    assert _detect_global_barrier_addresses(accesses) == {barrier_addr}


def test_apply_epoch_annotations_splits_store_epochs_after_detected_barrier():
    accesses = _two_phase_accesses_with_full_barrier()
    _apply_epoch_annotations(accesses)

    phase0_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and acc.event_id < 100
    ]
    barrier_ops = [acc for acc in accesses if acc.access_type == AccessType.ATOMIC]
    phase1_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and acc.event_id >= 300
    ]

    assert phase0_stores and phase1_stores and barrier_ops
    assert all(acc.epoch == 0 for acc in phase0_stores)
    assert all(acc.epoch == 0 for acc in barrier_ops)
    assert all(acc.epoch == 1 for acc in phase1_stores)


def test_detect_races_filters_cross_phase_waw_after_global_barrier():
    accesses = _two_phase_accesses_with_full_barrier()
    races = detect_races(accesses)

    assert races == []
    phase1_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and acc.event_id >= 300
    ]
    assert all(acc.epoch == 1 for acc in phase1_stores)


def test_detect_races_without_barrier_keeps_single_epoch_and_reports_waw():
    n = 4
    accesses = []
    for i in range(n):
        accesses.append(_store_access(block_idx=i, offset=i, event_id=i))
        accesses.append(
            _store_access(block_idx=i, offset=(i + 1) % n, event_id=100 + i)
        )

    races = detect_races(accesses)

    assert len(races) == 4
    assert all(r.race_type == RaceType.WAW for r in races)
    assert all(acc.epoch == 0 for acc in accesses)


def test_partial_barrier_signal_does_not_advance_epoch():
    n = 4
    barrier_addr = 4096
    accesses = []
    for i in range(n):
        accesses.append(_store_access(block_idx=i, offset=i, event_id=i))
        if i < n - 1:
            accesses.append(
                _atomic_access(
                    block_idx=i,
                    offset=barrier_addr,
                    event_id=100 + i,
                    atomic_op="rmw:add",
                )
            )
            accesses.append(
                _atomic_access(
                    block_idx=i,
                    offset=barrier_addr,
                    event_id=200 + i,
                    atomic_op="cas",
                )
            )
        accesses.append(
            _store_access(block_idx=i, offset=(i + 1) % n, event_id=300 + i)
        )

    races = detect_races(accesses)

    assert len(races) == 4
    assert all(r.race_type == RaceType.WAW for r in races)
    assert all(acc.epoch == 0 for acc in accesses)


def test_apply_epoch_annotations_supports_multiple_completed_barriers():
    accesses = _three_phase_accesses_with_two_full_barriers()
    _apply_epoch_annotations(accesses)

    phase0_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and acc.event_id < 100
    ]
    phase1_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and 300 <= acc.event_id < 400
    ]
    phase2_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and acc.event_id >= 600
    ]

    assert phase0_stores and phase1_stores and phase2_stores
    assert all(acc.epoch == 0 for acc in phase0_stores)
    assert all(acc.epoch == 1 for acc in phase1_stores)
    assert all(acc.epoch == 2 for acc in phase2_stores)


def test_incomplete_second_barrier_does_not_globally_reach_epoch_two():
    accesses = _three_phase_accesses_with_two_full_barriers()

    # Remove block 3's second barrier atomics: second pass is not globally complete.
    accesses = [
        acc
        for acc in accesses
        if not (
            acc.grid_idx == (3, 0, 0)
            and acc.access_type == AccessType.ATOMIC
            and acc.event_id >= 400
        )
    ]

    _apply_epoch_annotations(accesses)
    phase2_stores = [
        acc
        for acc in accesses
        if acc.access_type == AccessType.STORE and acc.event_id >= 600
    ]
    assert phase2_stores
    assert all(acc.epoch == 1 for acc in phase2_stores)
