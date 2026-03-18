import numpy as np

from triton_viz.clients.race_detector.data import (
    AccessType,
    MemoryAccess,
    effects_at_addr,
)
from triton_viz.clients.race_detector.race_detector import _sem_to_str, _scope_to_str


def _make_access(
    access_type,
    offset=100,
    read_mask=None,
    write_mask=None,
    **kwargs,
):
    return MemoryAccess(
        access_type=access_type,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(0, 0, 0),
        read_mask=read_mask,
        write_mask=write_mask,
        **kwargs,
    )


# ── effects_at_addr tests ──


def test_effects_at_addr_load():
    acc = _make_access(AccessType.LOAD)
    assert effects_at_addr(acc, 100) == (True, False)


def test_effects_at_addr_store():
    acc = _make_access(AccessType.STORE)
    assert effects_at_addr(acc, 100) == (False, True)


def test_effects_at_addr_atomic_legacy():
    acc = _make_access(AccessType.ATOMIC)
    assert effects_at_addr(acc, 100) == (True, True)


def test_effects_at_addr_cas_success():
    acc = _make_access(
        AccessType.ATOMIC,
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    assert effects_at_addr(acc, 100) == (True, True)


def test_effects_at_addr_cas_fail():
    acc = _make_access(
        AccessType.ATOMIC,
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([False], dtype=np.bool_),
    )
    assert effects_at_addr(acc, 100) == (True, False)


def test_effects_at_addr_rmw():
    acc = _make_access(
        AccessType.ATOMIC,
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    assert effects_at_addr(acc, 100) == (True, True)


def test_effects_at_addr_multi_lane_mixed_cas():
    """2 lanes same addr, lane 0 CAS fail (write=False), lane 1 CAS success (write=True) → (True, True)."""
    acc = MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([100, 100], dtype=np.int64),
        masks=np.array([True, True], dtype=np.bool_),
        grid_idx=(0, 0, 0),
        read_mask=np.array([True, True], dtype=np.bool_),
        write_mask=np.array([False, True], dtype=np.bool_),
    )
    assert effects_at_addr(acc, 100) == (True, True)


def test_effects_at_addr_multi_lane_all_cas_fail():
    """3 lanes same addr, all CAS fail → (True, False)."""
    acc = MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([100, 100, 100], dtype=np.int64),
        masks=np.array([True, True, True], dtype=np.bool_),
        grid_idx=(0, 0, 0),
        read_mask=np.array([True, True, True], dtype=np.bool_),
        write_mask=np.array([False, False, False], dtype=np.bool_),
    )
    assert effects_at_addr(acc, 100) == (True, False)


def test_effects_at_addr_first_lane_masked_off():
    """2 lanes same addr, first masked off, second active → (True, True)."""
    acc = MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([100, 100], dtype=np.int64),
        masks=np.array([False, True], dtype=np.bool_),
        grid_idx=(0, 0, 0),
        read_mask=np.array([False, True], dtype=np.bool_),
        write_mask=np.array([False, True], dtype=np.bool_),
    )
    assert effects_at_addr(acc, 100) == (True, True)


# ── Metadata round-trip tests ──


def test_metadata_round_trip_cas():
    cmp_val = np.array([42], dtype=np.int32)
    val_val = np.array([99], dtype=np.int32)
    old_val = np.array([42], dtype=np.int32)
    acc = MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([100], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(0, 0, 0),
        atomic_op="cas",
        atomic_sem="acq_rel",
        atomic_scope="gpu",
        atomic_cmp=cmp_val,
        atomic_val=val_val,
        atomic_old=old_val,
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    assert acc.atomic_sem == "acq_rel"
    assert acc.atomic_scope == "gpu"
    np.testing.assert_array_equal(acc.atomic_cmp, cmp_val)
    np.testing.assert_array_equal(acc.atomic_val, val_val)
    np.testing.assert_array_equal(acc.atomic_old, old_val)
    np.testing.assert_array_equal(acc.read_mask, [True])
    np.testing.assert_array_equal(acc.write_mask, [True])
    assert acc.legacy_atomic is False


def test_metadata_round_trip_rmw():
    val_val = np.array([1], dtype=np.int32)
    old_val = np.array([5], dtype=np.int32)
    acc = MemoryAccess(
        access_type=AccessType.ATOMIC,
        ptr=0,
        offsets=np.array([200], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(1, 0, 0),
        atomic_op="rmw:add",
        atomic_sem="release",
        atomic_scope="sys",
        atomic_val=val_val,
        atomic_old=old_val,
        read_mask=np.array([True], dtype=np.bool_),
        write_mask=np.array([True], dtype=np.bool_),
    )
    assert acc.atomic_sem == "release"
    assert acc.atomic_scope == "sys"
    np.testing.assert_array_equal(acc.atomic_val, val_val)
    np.testing.assert_array_equal(acc.atomic_old, old_val)
    assert acc.atomic_cmp is None


# ── sem/scope conversion tests ──


def test_sem_to_str_handles_string_and_enum():
    assert _sem_to_str("acq_rel") == "acq_rel"
    assert _sem_to_str("RELEASE") == "release"
    assert _sem_to_str(None) is None

    # Simulate enum-like object
    class FakeEnum:
        def __str__(self):
            return "ATOMIC_ORDERING.ACQ_REL"

    assert _sem_to_str(FakeEnum()) == "acq_rel"


def test_scope_to_str_handles_string_and_enum():
    assert _scope_to_str("gpu") == "gpu"
    assert _scope_to_str("CTA") == "cta"
    assert _scope_to_str(None) is None

    class FakeEnum:
        def __str__(self):
            return "MEMORY_SCOPE.GPU"

    assert _scope_to_str(FakeEnum()) == "gpu"
