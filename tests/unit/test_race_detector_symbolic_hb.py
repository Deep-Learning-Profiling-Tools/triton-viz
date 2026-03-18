"""Tests for symbolic path HB solver integration."""

from unittest.mock import MagicMock

from triton_viz.clients.race_detector.data import (
    AccessType,
    SymbolicMemoryAccess,
)
from triton_viz.clients.race_detector.hb_solver import SymbolicHBSolver


def _make_sym_access(
    access_type,
    event_id,
    ptr_sig="data_ptr",
    atomic_op=None,
    atomic_sem=None,
    atomic_scope=None,
    epoch=0,
):
    """Create a SymbolicMemoryAccess with a mock ptr_expr."""
    ptr_expr = MagicMock()
    ptr_expr._test_sig = ptr_sig  # used by our fake signature fn
    return SymbolicMemoryAccess(
        access_type=access_type,
        ptr_expr=ptr_expr,
        mask_expr=None,
        is_data_dependent=False,
        event_id=event_id,
        atomic_op=atomic_op,
        atomic_sem=atomic_sem,
        atomic_scope=atomic_scope,
        epoch=epoch,
    )


def _ptr_sig_fn(ptr_expr):
    """Fake ptr signature function that returns the test signature."""
    return (ptr_expr._test_sig,)


def test_symbolic_release_acquire_conservative():
    """Release/acquire flag pattern in symbolic mode → conservatively reports race.

    SymbolicHBSolver cannot prove reads-from or must-alias, so it
    conservatively returns True and defers to the concrete fallback.
    """
    store_data = _make_sym_access(AccessType.STORE, event_id=0, ptr_sig="data_ptr")
    release_flag = _make_sym_access(
        AccessType.ATOMIC,
        event_id=1,
        ptr_sig="flag_ptr",
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
    )
    acquire_flag = _make_sym_access(
        AccessType.ATOMIC,
        event_id=2,
        ptr_sig="flag_ptr",
        atomic_op="rmw:xchg",
        atomic_sem="acquire",
        atomic_scope="gpu",
    )
    load_data = _make_sym_access(AccessType.LOAD, event_id=3, ptr_sig="data_ptr")

    all_accesses = [store_data, release_flag, acquire_flag, load_data]

    solver = SymbolicHBSolver(store_data, load_data, all_accesses, _ptr_sig_fn)
    assert solver.check_race_possible()  # conservative → race reported


def test_symbolic_relaxed_still_races():
    """Same pattern with sem='relaxed' → race."""
    store_data = _make_sym_access(AccessType.STORE, event_id=0, ptr_sig="data_ptr")
    relaxed_flag = _make_sym_access(
        AccessType.ATOMIC,
        event_id=1,
        ptr_sig="flag_ptr",
        atomic_op="rmw:xchg",
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )
    relaxed_read = _make_sym_access(
        AccessType.ATOMIC,
        event_id=2,
        ptr_sig="flag_ptr",
        atomic_op="rmw:xchg",
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )
    load_data = _make_sym_access(AccessType.LOAD, event_id=3, ptr_sig="data_ptr")

    all_accesses = [store_data, relaxed_flag, relaxed_read, load_data]

    solver = SymbolicHBSolver(store_data, load_data, all_accesses, _ptr_sig_fn)
    assert solver.check_race_possible()  # no sw → race


def test_symbolic_fallback_on_cas_control_flow():
    """CAS return used in control flow triggers concrete fallback.

    This is tested implicitly: when _need_concrete_fallback is True,
    the symbolic path is skipped and concrete execution happens instead.
    The symbolic HB solver is never invoked in that case.

    We just verify the solver handles the no-sync-events case correctly.
    """
    store = _make_sym_access(AccessType.STORE, event_id=0, ptr_sig="data_ptr")
    load = _make_sym_access(AccessType.LOAD, event_id=1, ptr_sig="data_ptr")

    # No sync events at all
    all_accesses = [store, load]
    solver = SymbolicHBSolver(store, load, all_accesses, _ptr_sig_fn)
    assert solver.check_race_possible()  # no sync → race possible


def test_symbolic_hb_signature_same_no_suppress():
    """Matching ptr signatures + release/acquire pair → still returns True (no suppress).

    Even though signatures match, symbolic HB cannot prove must-alias or
    reads-from, so it conservatively reports a race.
    """
    store_data = _make_sym_access(AccessType.STORE, event_id=0, ptr_sig="data_ptr")
    release_flag = _make_sym_access(
        AccessType.ATOMIC,
        event_id=1,
        ptr_sig="flag_ptr",
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="gpu",
    )
    acquire_flag = _make_sym_access(
        AccessType.ATOMIC,
        event_id=2,
        ptr_sig="flag_ptr",  # same signature as release
        atomic_op="cas",
        atomic_sem="acquire",
        atomic_scope="gpu",
    )
    load_data = _make_sym_access(AccessType.LOAD, event_id=3, ptr_sig="data_ptr")

    all_accesses = [store_data, release_flag, acquire_flag, load_data]

    solver = SymbolicHBSolver(store_data, load_data, all_accesses, _ptr_sig_fn)
    assert solver.check_race_possible()  # no suppress — conservative


def test_symbolic_hb_acquire_unprovable_no_suppress():
    """Textbook producer-consumer symbolic pattern → returns True.

    The symbolic engine lacks atomic_old to prove reads-from, so
    even a structurally perfect release/acquire pattern cannot be
    proven to establish ordering.
    """
    # Producer pattern: store + release
    store = _make_sym_access(AccessType.STORE, event_id=0, ptr_sig="shared_buf")
    release = _make_sym_access(
        AccessType.ATOMIC,
        event_id=1,
        ptr_sig="ready_flag",
        atomic_op="rmw:xchg",
        atomic_sem="release",
        atomic_scope="sys",
    )
    # Consumer pattern: acquire + load
    acquire = _make_sym_access(
        AccessType.ATOMIC,
        event_id=2,
        ptr_sig="ready_flag",
        atomic_op="rmw:xchg",
        atomic_sem="acquire",
        atomic_scope="sys",
    )
    load = _make_sym_access(AccessType.LOAD, event_id=3, ptr_sig="shared_buf")

    all_accesses = [store, release, acquire, load]

    solver = SymbolicHBSolver(store, load, all_accesses, _ptr_sig_fn)
    assert solver.check_race_possible()  # unprovable → race reported
