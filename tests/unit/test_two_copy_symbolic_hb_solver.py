"""Synthetic unit tests for ``TwoCopySymbolicHBSolver``.

These tests construct ``AccessEventRecord`` inputs by hand to exercise the
solver's invariants in isolation from the (Triton-dependent) capture pipeline.
"""

from __future__ import annotations

import torch
from z3 import (
    And,
    BoolVal,
    Int,
    IntVal,
    Solver,
    sat,
)

from triton_viz.clients.race_detector.data import (
    AccessEventRecord,
    RaceType,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.clients.symbolic_engine import SymbolicExpr
from triton_viz.core.data import (
    AtomicCas,
    AtomicRMW,
    Load,
    Store,
)


# ──────────────────────── Helpers ────────────────────────


def _scalar_store(addr_expr, *, event_id, program_seq=0, elem_size=1, mask=None):
    return AccessEventRecord(
        op_type=Store,
        access_mode="write",
        addr_expr=addr_expr,
        local_constraints=() if mask is None else (mask,),
        active=True,
        reads=False,
        writes=True,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def _scalar_load(addr_expr, *, event_id, program_seq=0, elem_size=1, mask=None):
    return AccessEventRecord(
        op_type=Load,
        access_mode="read",
        addr_expr=addr_expr,
        local_constraints=() if mask is None else (mask,),
        active=True,
        reads=True,
        writes=False,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def _cas_record(
    addr_expr,
    cmp_value,
    new_value,
    old_value,
    *,
    event_id,
    program_seq,
    sem="acq_rel",
    scope="gpu",
    elem_size=4,
    tensor=None,
    extra_local_vars=(),
):
    return AccessEventRecord(
        op_type=AtomicCas,
        access_mode="read",
        tensor=tensor,
        addr_expr=addr_expr,
        active=True,
        reads=True,
        writes=None,
        is_atomic=True,
        atomic_kind="cas",
        sem=sem,
        scope=scope,
        old_value=old_value,
        written_value=None,
        cas_cmp_value=cmp_value,
        cas_new_value=new_value,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
        copy_local_vars=(old_value,) + tuple(extra_local_vars),
    )


def _rmw_record(addr_expr, *, event_id, program_seq, elem_size=4):
    return AccessEventRecord(
        op_type=AtomicRMW,
        access_mode="read",
        addr_expr=addr_expr,
        active=True,
        reads=True,
        writes=True,
        is_atomic=True,
        atomic_kind="rmw",
        sem="acq_rel",
        scope="gpu",
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def _solve(records, *, grid=(2, 1, 1), arange_dict=None):
    return TwoCopySymbolicHBSolver(records, grid=grid, arange_dict=arange_dict or {})


# ──────────────────────── Tests ────────────────────────


def test_pid_alpha_renaming_reports_race_for_same_template():
    """elem_size=4 stride=1 means adjacent pids overlap by 3 bytes."""
    base = 1_000_000
    addr = IntVal(base) + 1 * SymbolicExpr.PID0
    record = _scalar_store(addr, event_id=0, program_seq=0, elem_size=4)

    solver = _solve([record], grid=(2, 1, 1))
    reports = solver.find_races()
    assert len(reports) == 1
    assert reports[0].race_type == RaceType.WAW


def test_arange_alpha_renaming_independent_lanes():
    """An arange-strided WAW where two blocks alias on different lanes."""
    base = 2_000_000
    arange_var = Int("arange_lane")
    arange_dict = {(0, 8): (arange_var, And(arange_var >= 0, arange_var < 8))}
    addr = IntVal(base) + 28 * SymbolicExpr.PID0 + 4 * arange_var
    record = _scalar_store(addr, event_id=0, program_seq=0, elem_size=4)

    solver = _solve([record], grid=(4, 1, 1), arange_dict=arange_dict)
    reports = solver.find_races()
    assert len(reports) == 1


def test_different_block_constraint_is_required():
    """Without different-block, identical blocks would self-race; ensure it's added."""
    base = 3_000_000
    addr = IntVal(base) + 4 * SymbolicExpr.PID0
    record = _scalar_store(addr, event_id=0, program_seq=0, elem_size=4)

    solver = _solve([record], grid=(2, 1, 1))
    # different_blocks must be a non-trivial constraint (Or over pid components).
    s = Solver()
    s.add(solver.different_blocks)
    s.add(solver.ctx_a.pid[0] == solver.ctx_b.pid[0])
    s.add(solver.ctx_a.pid[1] == solver.ctx_b.pid[1])
    s.add(solver.ctx_a.pid[2] == solver.ctx_b.pid[2])
    assert s.check() != sat  # cannot be satisfied — different_blocks does its job


def test_mask_false_suppresses_race():
    base = 4_000_000
    addr = IntVal(base) + 4 * SymbolicExpr.PID0
    # mask = False means access is inactive.
    record = _scalar_store(
        addr, event_id=0, program_seq=0, elem_size=4, mask=BoolVal(False)
    )
    solver = _solve([record], grid=(2, 1, 1))
    assert solver.find_races() == []


def test_byte_overlap_for_elem_size_gt_1():
    """elem_size=4: addresses 100 and 102 overlap; addresses 100 and 104 do not."""
    base_a = IntVal(100)
    base_b = IntVal(102)
    rec_a = _scalar_store(base_a, event_id=0, program_seq=0, elem_size=4)
    rec_b = _scalar_load(base_b, event_id=1, program_seq=0, elem_size=4)
    # Two distinct blocks reference the same template — race expected.
    addr_a = base_a + 0 * SymbolicExpr.PID0
    addr_b = base_b + 0 * SymbolicExpr.PID0
    rec_a.addr_expr = addr_a
    rec_b.addr_expr = addr_b
    assert _solve(
        [rec_a, rec_b]
    ).find_races(), "elem_size=4 with addrs 100/102 should byte-overlap → race"


def test_tensor_base_pointer_is_not_renamed():
    """A Z3 var that is not in copy_local_vars must appear identically in both copies."""
    base_var = Int("base_ptr_const")
    addr = base_var + 4 * SymbolicExpr.PID0
    record = _scalar_store(addr, event_id=0, program_seq=0, elem_size=4)
    solver = _solve([record], grid=(2, 1, 1))
    events_a = [e for e in solver.events if e.copy == "a"]
    events_b = [e for e in solver.events if e.copy == "b"]
    # Solver must agree that base_var has the SAME value across copies.
    s = Solver()
    s.add(
        events_a[0].addr - solver.ctx_a.pid[0] * 4
        != events_b[0].addr - solver.ctx_b.pid[0] * 4
    )
    assert s.check() != sat  # the only difference is pid_a vs pid_b, not base_var


def test_launch_level_cas_return_rename_across_records():
    """A CAS return referenced in a downstream load's mask must use the launch-level
    copy-local rename — not the original Z3 var.
    """
    cas_old = Int("cas_old")
    cas_addr = IntVal(8_000_000)
    cas_cmp = IntVal(0)
    cas_new = IntVal(1)
    cas_record = _cas_record(
        cas_addr, cas_cmp, cas_new, cas_old, event_id=0, program_seq=0
    )
    base = 9_000_000
    load_addr = IntVal(base) + 4 * SymbolicExpr.PID0
    load_record = _scalar_load(
        load_addr, event_id=1, program_seq=1, elem_size=4, mask=cas_old == 1
    )
    solver = _solve([cas_record, load_record], grid=(2, 1, 1))

    # Check that the load events in copy A and B use distinct CAS-return vars.
    load_a = [e for e in solver.events if e.copy == "a" and e.record is load_record][0]
    load_b = [e for e in solver.events if e.copy == "b" and e.record is load_record][0]
    # The active expression should not contain the original cas_old; it should
    # contain copy-specific renames.
    load_a_str = str(load_a.active)
    load_b_str = str(load_b.active)
    assert "__a" in load_a_str, f"expected _a-renamed CAS return: {load_a_str}"
    assert "__b" in load_b_str, f"expected _b-renamed CAS return: {load_b_str}"
    # And the rename should differ.
    assert load_a_str != load_b_str


def test_cas_old_value_alpha_renaming_distinct_decls():
    cas_old = Int("cas_old_distinct")
    cas_record = _cas_record(
        IntVal(0), IntVal(0), IntVal(1), cas_old, event_id=0, program_seq=0
    )
    solver = _solve([cas_record], grid=(2, 1, 1))
    cas_a = [e for e in solver.events if e.copy == "a"][0]
    cas_b = [e for e in solver.events if e.copy == "b"][0]
    assert cas_a.old_value is not cas_b.old_value
    assert str(cas_a.old_value) != str(cas_b.old_value)


def test_cas_read_from_uses_exact_addr_not_byte_overlap():
    """Two CAS objects at byte addresses 100 (elem=4) and 101 (elem=4) byte-overlap
    but are not the same atomic location; solver must NOT manufacture a sync edge.
    """
    cas_old_a = Int("cas_old_a")
    cas_old_b = Int("cas_old_b")
    cas_a = _cas_record(
        IntVal(100),
        IntVal(0),
        IntVal(1),
        cas_old_a,
        event_id=0,
        program_seq=0,
        sem="release",
        scope="gpu",
    )
    cas_b = _cas_record(
        IntVal(101),
        IntVal(0),
        IntVal(1),
        cas_old_b,
        event_id=1,
        program_seq=0,
        sem="acquire",
        scope="gpu",
    )
    # data writes around the CAS objects to give the solver actual race candidates.
    store_data = _scalar_store(IntVal(200), event_id=2, program_seq=0, elem_size=4)
    store_data.program_seq = 0
    cas_a.program_seq = 1
    cas_b.program_seq = 1
    load_data = _scalar_load(IntVal(200), event_id=3, program_seq=2, elem_size=4)

    # Producer copy A: store_data (po=0) → cas_a release (po=1)
    # Consumer copy B: cas_b acquire (po=1) → load_data (po=2)
    # Since CAS addrs differ, sync should NOT establish; race remains.
    records = [store_data, cas_a, load_data]
    solver = _solve(records, grid=(2, 1, 1))
    reports = solver.find_races()
    # Race candidates between store_data^A and load_data^B should still surface.
    pairs = {(r.first.event_id, r.second.event_id) for r in reports}
    assert any(
        2 in p and 3 in p for p in pairs
    ), "store/load race should not be suppressed when CAS addresses differ"


def test_unknown_initial_value_does_not_fabricate_sync():
    """Tensor above _MAX_INITIAL_ATOMIC_ELEMENTS: initial source falls back to
    rf_unknown → solver must not invent a sync edge that suppresses the data
    race. Patch 2 lifted the supported size to ``_MAX_INITIAL_ATOMIC_ELEMENTS``
    contiguous elements; this test stays meaningful by sitting strictly above
    that bound.
    """
    big_numel = TwoCopySymbolicHBSolver._MAX_INITIAL_ATOMIC_ELEMENTS + 1
    multi = torch.zeros(big_numel, dtype=torch.int32)
    cas_old = Int("cas_old_unknown_initial")
    cas = _cas_record(
        IntVal(int(multi.data_ptr())),
        IntVal(0),
        IntVal(1),
        cas_old,
        event_id=10,
        program_seq=1,
        tensor=multi,
    )
    # Place store/load on a slot well past every modeled flag offset so they
    # cannot be confused with the CAS address itself.
    data_offset = (big_numel + 1) * 4
    store_data = _scalar_store(
        IntVal(int(multi.data_ptr()) + data_offset),
        event_id=11,
        program_seq=0,
        elem_size=4,
    )
    load_data = _scalar_load(
        IntVal(int(multi.data_ptr()) + data_offset),
        event_id=12,
        program_seq=2,
        elem_size=4,
    )

    records = [store_data, cas, load_data]
    solver = _solve(records, grid=(2, 1, 1))
    # Sanity: this CAS reader must hit the unknown-source branch.
    cas_reader_a = next(e for e in solver.events if e.copy == "a" and e.record is cas)
    assert cas_reader_a.idx in solver.rf_unknown_source
    assert cas_reader_a.idx not in solver.rf_init_source

    reports = solver.find_races()
    pairs = {(r.first.event_id, r.second.event_id) for r in reports}
    assert any(
        11 in p and 12 in p for p in pairs
    ), "unknown initial CAS source must not suppress store/load race"


def test_atomic_only_competing_cas_no_race():
    cas_old_a = Int("cas_only_a")
    cas_old_b = Int("cas_only_b")
    cas_a = _cas_record(
        IntVal(500), IntVal(0), IntVal(1), cas_old_a, event_id=0, program_seq=0
    )
    cas_b = _cas_record(
        IntVal(500), IntVal(0), IntVal(1), cas_old_b, event_id=1, program_seq=0
    )
    assert _solve([cas_a, cas_b], grid=(2, 1, 1)).find_races() == []


def test_atomic_rmw_does_not_participate_in_cas_synchronization():
    """RMW vs CAS must not establish a synchronizes-with edge."""
    rmw = _rmw_record(IntVal(700), event_id=0, program_seq=1)
    cas_old = Int("cas_after_rmw")
    cas = _cas_record(
        IntVal(700),
        IntVal(0),
        IntVal(1),
        cas_old,
        event_id=1,
        program_seq=2,
    )
    store_data = _scalar_store(IntVal(800), event_id=2, program_seq=0, elem_size=4)
    load_data = _scalar_load(IntVal(800), event_id=3, program_seq=3, elem_size=4)
    reports = _solve([store_data, rmw, cas, load_data], grid=(2, 1, 1)).find_races()
    pairs = {(r.first.event_id, r.second.event_id) for r in reports}
    # RMW cannot bridge the data-race between store and load.
    assert any(2 in p and 3 in p for p in pairs)


def test_atomic_only_competing_rmw_no_data_race():
    """Two RMWs at the same address: atomic vs atomic, no race."""
    rmw_a = _rmw_record(IntVal(900), event_id=0, program_seq=0)
    rmw_b = _rmw_record(IntVal(900), event_id=1, program_seq=0)
    assert _solve([rmw_a, rmw_b], grid=(2, 1, 1)).find_races() == []


def test_inactive_atomic_event_does_not_create_hb_bridge():
    """Inactive release CAS in copy A must not synchronize with acquire CAS in copy B,
    so a producer/consumer pair around it remains unordered.
    """
    base_data = IntVal(1100)
    base_flag = IntVal(1200)

    # Inactive release CAS via mask = False on local_constraints.
    cas_old_inactive = Int("cas_inactive")
    cas_inactive = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        cas_old_inactive,
        event_id=10,
        program_seq=1,
        sem="release",
    )
    # Force inactive via local constraints.
    cas_inactive.local_constraints = (BoolVal(False),)

    cas_old_acquire = Int("cas_acquire")
    cas_acquire = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        cas_old_acquire,
        event_id=11,
        program_seq=1,
        sem="acquire",
    )

    store_data = _scalar_store(base_data, event_id=12, program_seq=0, elem_size=4)
    load_data = _scalar_load(base_data, event_id=13, program_seq=2, elem_size=4)

    records = [store_data, cas_inactive, cas_acquire, load_data]
    reports = _solve(records, grid=(2, 1, 1)).find_races()
    pairs = {(r.first.event_id, r.second.event_id) for r in reports}
    assert any(12 in p and 13 in p for p in pairs)


def test_rf_choice_gated_by_reader_active():
    """An inactive CAS reader must contribute no synchronizes-with edge."""
    cas_old_inactive = Int("cas_inactive_reader")
    cas_inactive = _cas_record(
        IntVal(1300),
        IntVal(0),
        IntVal(1),
        cas_old_inactive,
        event_id=20,
        program_seq=0,
        sem="acquire",
    )
    cas_inactive.local_constraints = (BoolVal(False),)
    # Producer release CAS:
    cas_old_release = Int("cas_release_reader")
    cas_release = _cas_record(
        IntVal(1300),
        IntVal(0),
        IntVal(1),
        cas_old_release,
        event_id=21,
        program_seq=0,
        sem="release",
    )
    solver = _solve([cas_inactive, cas_release], grid=(2, 1, 1))
    # The inactive reader's RF source must NOT be selectable.
    inactive_evs = [e for e in solver.events if e.record is cas_inactive]
    for inactive in inactive_evs:
        # gather rf vars referencing this reader
        rf_vars = [
            v for (w_idx, r_idx), v in solver.rf_source.items() if r_idx == inactive.idx
        ]
        # there must be a constraint forcing all rf to False when reader is inactive
        s = Solver()
        for c in solver.rf_constraints:
            s.add(c)
        # Inactive should imply all rf == False
        for rf in rf_vars:
            s.push()
            s.add(rf)
            assert s.check() != sat, "inactive reader must have no active rf source"
            s.pop()


def test_per_lane_cas_lowering_vector():
    """Vector CAS: list-shaped old/cmp/new lowers to per-lane events."""
    old0 = Int("cas_old_lane0")
    old1 = Int("cas_old_lane1")
    addrs = [IntVal(2000), IntVal(2004)]
    cas = _cas_record(
        addrs,
        [IntVal(0), IntVal(0)],
        [IntVal(1), IntVal(1)],
        [old0, old1],
        event_id=0,
        program_seq=0,
    )
    cas.copy_local_vars = (old0, old1)
    solver = _solve([cas], grid=(2, 1, 1))
    cas_events_a = [e for e in solver.events if e.copy == "a"]
    assert len(cas_events_a) == 2
    assert {e.lane for e in cas_events_a} == {0, 1}


def test_witness_addr_max_for_byte_overlap():
    """elem_size>1 byte-overlap witness should be max(addr_a, addr_b).

    Single record, byte-stride 1, elem_size 4 — adjacent pids overlap by 3
    bytes. The model picks witnesses where addr_a < addr_b (or equal); the
    witness address must be the larger one (the start of the overlap).
    """
    addr = IntVal(100) + 1 * SymbolicExpr.PID0
    rec = _scalar_store(addr, event_id=0, program_seq=0, elem_size=4)
    reports = _solve([rec], grid=(2, 1, 1)).find_races()
    assert len(reports) == 1
    # The report's witness_addr is max(addr_a, addr_b). Whichever pid Z3 picks,
    # both addresses are within {100..102} and the witness is the larger one.
    assert reports[0].witness_addr in {101, 102, 103}


def test_finalize_uses_arange_dict_snapshot():
    """A solver instance keeps using the arange_dict it was given even when
    SymbolicExpr.ARANGE_DICT changes afterwards.
    """
    arange_var = Int("arange_iso")
    snapshot = {(0, 4): (arange_var, And(arange_var >= 0, arange_var < 4))}
    base = 3_500_000
    # stride=12 < arange_max*elem_size=16 → adjacent pids overlap.
    addr = IntVal(base) + 4 * arange_var + 12 * SymbolicExpr.PID0
    rec = _scalar_store(addr, event_id=0, program_seq=0, elem_size=4)

    solver = _solve([rec], grid=(2, 1, 1), arange_dict=snapshot)

    # Mutate the live ARANGE_DICT — the solver should not be affected.
    SymbolicExpr.ARANGE_DICT.clear()
    SymbolicExpr.ARANGE_DICT[(0, 999)] = (Int("garbage"), BoolVal(True))

    # Re-running find_races on the same solver returns the same race count.
    assert len(solver.find_races()) == 1


def test_rf_constraints_visible_before_hb_closure():
    """Construction must populate rf_source before build_transitive_hb runs;
    a CAS-CAS sync edge should be reachable in self.hb.
    """
    base_data = IntVal(4100)
    base_flag = IntVal(4200)

    store_data = _scalar_store(base_data, event_id=0, program_seq=0, elem_size=4)
    cas_old_release = Int("cas_release_hb")
    cas_release = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        cas_old_release,
        event_id=1,
        program_seq=1,
        sem="release",
    )

    cas_old_acquire = Int("cas_acquire_hb")
    cas_acquire = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        cas_old_acquire,
        event_id=2,
        program_seq=0,
        sem="acquire",
    )
    load_data = _scalar_load(base_data, event_id=3, program_seq=1, elem_size=4)

    records = [store_data, cas_release, cas_acquire, load_data]
    solver = _solve(records, grid=(2, 1, 1))

    # Find store_data^A and load_data^B events.
    store_a = next(e for e in solver.events if e.copy == "a" and e.record is store_data)
    load_b = next(e for e in solver.events if e.copy == "b" and e.record is load_data)
    # The HB closure entry from store_a to load_b should be at least
    # POSSIBLY satisfiable — i.e., not a hard False.
    s = Solver()
    s.add(solver.grid_constraints)
    s.add(solver.different_blocks)
    for c in solver.rf_constraints:
        s.add(c)
    s.add(solver.hb[store_a.idx][load_b.idx])
    assert s.check() == sat


def test_initial_source_addr_matches_data_ptr_for_scalar_flag():
    """Address-domain invariant: scalar-flag CAS's R.addr equals tensor.data_ptr()."""
    flag = torch.zeros(1, dtype=torch.int32)
    cas_old = Int("cas_addr_check")
    cas = _cas_record(
        IntVal(int(flag.data_ptr())),
        IntVal(0),
        IntVal(1),
        cas_old,
        event_id=0,
        program_seq=0,
        tensor=flag,
    )
    solver = _solve([cas], grid=(2, 1, 1))
    cas_a = next(e for e in solver.events if e.copy == "a")
    s = Solver()
    s.add(solver.grid_constraints)
    s.add(solver.different_blocks)
    s.add(cas_a.addr == IntVal(int(flag.data_ptr())))
    assert s.check() == sat


def test_same_copy_backward_read_from_excluded():
    """A writer that is later-in-program-order in the same copy cannot supply
    a read-from source for an earlier reader in that copy.
    """
    base_flag = IntVal(5100)
    earlier_old = Int("rf_earlier")
    earlier = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        earlier_old,
        event_id=0,
        program_seq=0,
    )
    later_old = Int("rf_later")
    later = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        later_old,
        event_id=1,
        program_seq=1,
    )
    solver = _solve([earlier, later], grid=(2, 1, 1))

    # earlier_a (program_seq=0, copy=a) should NOT have later_a as an rf candidate.
    earlier_a = next(e for e in solver.events if e.copy == "a" and e.record is earlier)
    later_a = next(e for e in solver.events if e.copy == "a" and e.record is later)
    assert (later_a.idx, earlier_a.idx) not in solver.rf_source


def test_program_seq_zero_records_get_only_initial_source():
    """No same-copy backward writers means rf_source has only cross-copy entries."""
    base_flag = IntVal(6100)
    cas_old = Int("po_zero")
    rec = _cas_record(
        base_flag,
        IntVal(0),
        IntVal(1),
        cas_old,
        event_id=0,
        program_seq=0,
    )
    solver = _solve([rec], grid=(2, 1, 1))
    # Two CAS events (a, b). rf_source should have 2 cross-copy entries.
    a = next(e for e in solver.events if e.copy == "a")
    b = next(e for e in solver.events if e.copy == "b")
    assert (a.idx, b.idx) in solver.rf_source
    assert (b.idx, a.idx) in solver.rf_source
    assert len(solver.rf_source) == 2


def test_cas_trylock_single_winner_suppresses_guarded_waw():
    """Two CAS(0 -> 1) operations on the same scalar flag cannot both read 0.

    Without per-location atomic order (Patch 1), the two-copy solver can pick
    old_a == 0 and old_b == 0 simultaneously, activate both guarded stores,
    and report a false WAW between them. With atomic_order coherence at
    most one modeled CAS can read the initial 0 and successfully write 1.
    """
    flag = torch.zeros(1, dtype=torch.int32)
    flag_addr = IntVal(int(flag.data_ptr()))
    data_addr = IntVal(9_900_000)

    old = Int("trylock_old_z3")
    cas = _cas_record(
        flag_addr,
        IntVal(0),
        IntVal(1),
        old,
        event_id=0,
        program_seq=0,
        sem="acq_rel",
        scope="gpu",
        tensor=flag,
    )
    # Guarded scalar store at the same data slot: race candidate is the
    # cross-copy WAW between the two guarded stores.
    guarded_store = _scalar_store(
        data_addr,
        event_id=1,
        program_seq=1,
        elem_size=4,
        mask=(old == 0),
    )

    reports = _solve([cas, guarded_store], grid=(2, 1, 1)).find_races()
    assert reports == []


# ──────────────────────── Atomic mutual-atomicity boundaries ────────────────


def test_mixed_width_atomics_at_same_address_race():
    """A 4-byte and an 8-byte atomic RMW at the same base address overlap
    but are not mutually atomic (torn access) — must race."""
    base = 1_000_000
    narrow = _rmw_record(IntVal(base), event_id=0, program_seq=0, elem_size=4)
    wide = _rmw_record(IntVal(base), event_id=1, program_seq=1, elem_size=8)

    reports = _solve([narrow, wide], grid=(2, 1, 1)).find_races()
    assert reports, "expected torn mixed-width atomic pair to race"


def test_same_width_atomics_at_same_address_no_race():
    """Identical-width device-scope atomics at the same address are
    mutually atomic — race-free (the pre-fix rule, still intact)."""
    base = 1_000_000
    a = _rmw_record(IntVal(base), event_id=0, program_seq=0, elem_size=4)
    b = _rmw_record(IntVal(base), event_id=1, program_seq=1, elem_size=4)

    reports = _solve([a, b], grid=(2, 1, 1)).find_races()
    assert reports == []


def test_partially_overlapping_same_width_atomics_race():
    """Same width but byte-overlapping at DIFFERENT addresses (e.g. a
    misaligned pair) is torn, not mutually atomic — must race."""
    base = 1_000_000
    a = _rmw_record(IntVal(base), event_id=0, program_seq=0, elem_size=4)
    b = _rmw_record(IntVal(base + 2), event_id=1, program_seq=1, elem_size=4)

    reports = _solve([a, b], grid=(2, 1, 1)).find_races()
    assert reports, "expected partially overlapping atomics to race"
