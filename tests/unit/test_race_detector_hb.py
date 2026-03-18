import numpy as np

from triton_viz.clients.race_detector.data import (
    AccessType,
    MemoryAccess,
    RaceType,
)
from triton_viz.clients.race_detector.hb_solver import HBSolver
from triton_viz.clients.race_detector.race_detector import (
    _classify_race,
    detect_races,
)


def _make_access(
    access_type,
    block_idx,
    offset,
    event_id,
    atomic_op=None,
    atomic_sem=None,
    atomic_scope=None,
    read_mask=None,
    write_mask=None,
    legacy_atomic=False,
    atomic_old=None,
    atomic_cmp=None,
):
    masks = np.array([True], dtype=np.bool_)
    return MemoryAccess(
        access_type=access_type,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=masks,
        grid_idx=(block_idx, 0, 0),
        event_id=event_id,
        atomic_op=atomic_op,
        atomic_sem=atomic_sem,
        atomic_scope=atomic_scope,
        read_mask=read_mask
        if read_mask is not None
        else (
            masks.copy()
            if access_type == AccessType.ATOMIC and not legacy_atomic
            else None
        ),
        write_mask=write_mask
        if write_mask is not None
        else (
            masks.copy()
            if access_type == AccessType.ATOMIC and not legacy_atomic
            else None
        ),
        legacy_atomic=legacy_atomic,
        atomic_old=atomic_old,
        atomic_cmp=atomic_cmp,
    )


def test_release_acquire_flag_handoff_no_race():
    """Producer store data + release flag, consumer acquire flag + load data → no race."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    # Block 0 (producer): store data, then release-store flag (writes 1)
    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    release_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
        atomic_old=np.array([0], dtype=np.int32),
    )
    release_flag.atomic_val = np.array([1], dtype=np.int32)

    # Block 1 (consumer): acquire-load flag (reads old=1), then load data
    acquire_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="rmw:xchg",
        atomic_sem="acquire",
        atomic_scope="gpu",
        atomic_old=np.array([1], dtype=np.int32),
    )
    acquire_flag.atomic_val = np.array([0], dtype=np.int32)
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    # Check: store_data vs load_data should be ordered via
    # store_data ->po-> release_flag ->sw-> acquire_flag ->po-> load_data
    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([release_flag, acquire_flag])
    assert not solver.check_race_possible()  # UNSAT → no race


def test_relaxed_atomics_still_race():
    """Same pattern but sem='relaxed' → no sw edge → race reported."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    relaxed_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )

    relaxed_read = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="rmw:xchg",
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([relaxed_flag, relaxed_read])
    assert solver.check_race_possible()  # SAT → race exists


def test_cta_scope_no_cross_block_suppression():
    """scope='cta' on release/acquire → no cross-block sw → race reported."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    release_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="cta",
    )

    acquire_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="rmw:xchg",
        atomic_sem="acquire",
        atomic_scope="cta",
    )
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([release_flag, acquire_flag])
    assert solver.check_race_possible()  # SAT → race (cta can't cross blocks)


def test_failed_cas_no_write_effect():
    """Failed CAS at addr X vs plain load at addr X → no conflict (read-read)."""
    ADDR = 100

    # CAS that failed: read_mask=True, write_mask=False
    failed_cas = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=ADDR,
        event_id=0,
        atomic_op="cas",
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([False], dtype=np.bool_),
    )
    load = _make_access(AccessType.LOAD, block_idx=1, offset=ADDR, event_id=0)

    race_type = _classify_race(failed_cas, load, ADDR)
    assert race_type is None  # read-read → no race


def test_successful_cas_write_effect():
    """Successful CAS at addr X vs plain store at addr X → WAW."""
    ADDR = 100

    success_cas = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=ADDR,
        event_id=0,
        atomic_op="cas",
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    store = _make_access(AccessType.STORE, block_idx=1, offset=ADDR, event_id=0)

    race_type = _classify_race(success_cas, store, ADDR)
    assert race_type == RaceType.WAW


def test_atomic_atomic_still_suppressed():
    """Two atomic_add at same addr → no race (preserved behavior)."""
    ADDR = 100

    add_a = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=ADDR,
        event_id=0,
        atomic_op="rmw:add",
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )
    add_b = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=ADDR,
        event_id=0,
        atomic_op="rmw:add",
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )

    race_type = _classify_race(add_a, add_b, ADDR)
    assert race_type is None  # atomic-atomic → suppressed


def test_legacy_atomic_skips_hb():
    """legacy_atomic=True access → HB solver not invoked, conservative classify."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    # Even with release/acquire pattern, legacy atomics should not use HB
    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    release_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
        legacy_atomic=True,
    )
    acquire_flag = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="rmw:xchg",
        atomic_sem="acquire",
        atomic_scope="gpu",
        legacy_atomic=True,
    )
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    # Build accesses list and run detect_races
    all_accesses = [store_data, release_flag, acquire_flag, load_data]
    races = detect_races(all_accesses)

    # Should report race because legacy atomics prevent HB reasoning
    assert len(races) > 0
    assert races[0].race_type == RaceType.RAW


# ── Reads-from gated sw edge tests ──


def test_sw_xchg_release_cas_acquire_matching_old():
    """xchg(release) writes 1, CAS(acquire) reads old=1 → sw edge → suppress."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    release_xchg = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
        atomic_old=np.array([0], dtype=np.int32),
    )
    release_xchg.atomic_val = np.array([1], dtype=np.int32)

    acquire_cas = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="cas",
        atomic_sem="acquire",
        atomic_scope="gpu",
        atomic_old=np.array([1], dtype=np.int32),
        atomic_cmp=np.array([1], dtype=np.int32),
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    acquire_cas.atomic_val = np.array([0], dtype=np.int32)
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([release_xchg, acquire_cas])
    assert not solver.check_race_possible()  # sw edge → no race


def test_sw_spinlock_unlock_lock_pattern():
    """Spinlock: unlock xchg(0) + lock CAS(cmp=0) reads old=0 → suppress."""
    DATA_ADDR = 100
    LOCK_ADDR = 200

    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    unlock = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=LOCK_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
        atomic_old=np.array([1], dtype=np.int32),
    )
    unlock.atomic_val = np.array([0], dtype=np.int32)

    lock = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=LOCK_ADDR,
        event_id=0,
        atomic_op="cas",
        atomic_sem="acquire",
        atomic_scope="gpu",
        atomic_old=np.array([0], dtype=np.int32),
        atomic_cmp=np.array([0], dtype=np.int32),
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    lock.atomic_val = np.array([1], dtype=np.int32)
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([unlock, lock])
    assert not solver.check_race_possible()  # sw edge → no race


def test_sw_acquire_reads_wrong_value_no_suppress():
    """CAS reads old=42 (not 1) → no reads-from → no sw → race reported."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    release_xchg = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
        atomic_old=np.array([0], dtype=np.int32),
    )
    release_xchg.atomic_val = np.array([1], dtype=np.int32)

    acquire_cas = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="cas",
        atomic_sem="acquire",
        atomic_scope="gpu",
        atomic_old=np.array([42], dtype=np.int32),  # wrong value
        atomic_cmp=np.array([1], dtype=np.int32),
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([False], dtype=np.bool_),
    )
    acquire_cas.atomic_val = np.array([0], dtype=np.int32)
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([release_xchg, acquire_cas])
    assert solver.check_race_possible()  # no sw → race


def test_sw_intermediate_write_breaks_chain():
    """Intermediate relaxed xchg overwrites; acquire reads overwritten value → no sw with original release."""
    DATA_ADDR = 100
    FLAG_ADDR = 200

    store_data = _make_access(
        AccessType.STORE, block_idx=0, offset=DATA_ADDR, event_id=0
    )
    # Block 0 release writes 1
    release_xchg = _make_access(
        AccessType.ATOMIC,
        block_idx=0,
        offset=FLAG_ADDR,
        event_id=1,
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
        atomic_old=np.array([0], dtype=np.int32),
    )
    release_xchg.atomic_val = np.array([1], dtype=np.int32)

    # Block 2 (interloper): relaxed xchg overwrites flag to 99
    interloper = _make_access(
        AccessType.ATOMIC,
        block_idx=2,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="rmw:xchg",
        atomic_sem="relaxed",
        atomic_scope="gpu",
        atomic_old=np.array([1], dtype=np.int32),
    )
    interloper.atomic_val = np.array([99], dtype=np.int32)

    # Block 1 acquire reads old=99 (from interloper, not from release)
    acquire_xchg = _make_access(
        AccessType.ATOMIC,
        block_idx=1,
        offset=FLAG_ADDR,
        event_id=0,
        atomic_op="rmw:xchg",
        atomic_sem="acquire",
        atomic_scope="gpu",
        atomic_old=np.array([99], dtype=np.int32),
    )
    acquire_xchg.atomic_val = np.array([0], dtype=np.int32)
    load_data = _make_access(AccessType.LOAD, block_idx=1, offset=DATA_ADDR, event_id=1)

    solver = HBSolver(store_data, load_data)
    solver.add_sync_events([release_xchg, interloper, acquire_xchg])
    assert solver.check_race_possible()  # no sw with release → race
