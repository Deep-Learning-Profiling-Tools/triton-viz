"""Solver-level tests for RMW-return modeling (spec part B).

Records are built by hand (same style as test_two_copy_symbolic_hb_solver)
so the axioms are exercised in isolation from the capture pipelines:

  * value-modeled RMW rf + reads-through sw (release chains, B.1.4)
  * atomicity/immediacy generalized to RMW (B.1.3)
  * the counting axiom and its guards (B.1.5): last-block-done,
    work-queue disjointness, single-winner
  * the rf_chain source (real races at non-adjacent ranks stay SAT)
  * the observation-in-address gate (B.5 boundary with the B.1.5 carve-out)

Every proof here has a mutation twin that must flip to SAT — a vacuously
UNSAT encoding would fail those twins.
"""

from __future__ import annotations

import pytest
import torch
from z3 import Int, IntVal

from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
    _rmw_written_value,
)
from triton_viz.clients.symbolic_engine import SymbolicExpr
from triton_viz.core.data import AtomicRMW, Load, Store


PARTIAL_BASE = 1 << 20
DATA_BASE = 1 << 21
BUF_BASE = 1 << 22


def _store(addr, *, event_id, program_seq, elem_size=4, active=True, mask=None):
    return AccessEventRecord(
        op_type=Store,
        access_mode="write",
        addr_expr=addr,
        local_constraints=() if mask is None else (mask,),
        active=active,
        reads=False,
        writes=True,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def _load(addr, *, event_id, program_seq, elem_size=4, active=True, mask=None):
    return AccessEventRecord(
        op_type=Load,
        access_mode="read",
        addr_expr=addr,
        local_constraints=() if mask is None else (mask,),
        active=active,
        reads=True,
        writes=False,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def _rmw(
    addr,
    old,
    *,
    op,
    operand,
    event_id,
    program_seq,
    sem="acq_rel",
    scope="gpu",
    elem_size=4,
    tensor=None,
    active=True,
):
    """A VALUE-MODELED RMW record: observation var in copy_local_vars,
    exactly like the CAS return."""
    return AccessEventRecord(
        op_type=AtomicRMW,
        access_mode="read",
        tensor=tensor,
        addr_expr=addr,
        active=active,
        reads=True,
        writes=True,
        is_atomic=True,
        atomic_kind="rmw",
        sem=sem,
        scope=scope,
        old_value=old,
        written_value=None,
        rmw_op=op,
        rmw_operand=operand,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
        copy_local_vars=(old,),
    )


def _counter_tensor(value=0):
    return torch.full((1,), value, dtype=torch.int32)


def _solve(records, *, grid=(4, 1, 1), arange_dict=None):
    return TwoCopySymbolicHBSolver(records, grid=grid, arange_dict=arange_dict or {})


# ──────────────────── written-value model ────────────────────


def test_rmw_written_value_table():
    old, v = Int("o"), Int("v")
    assert _rmw_written_value("add", old, v) is not None
    assert _rmw_written_value("max", old, v) is not None
    assert _rmw_written_value("min", old, v) is not None
    assert _rmw_written_value("xchg", old, v) is not None
    # Bitwise / unsigned / float ops have no Int-sort model.
    for op in ("and", "or", "xor", "umax", "umin", "fadd", None):
        assert _rmw_written_value(op, old, v) is None
    assert _rmw_written_value("add", None, v) is None
    assert _rmw_written_value("add", old, None) is None


# ──────────────────── last-block-done (counting + sw) ────────────────────


def _last_block_records(*, sem, counter):
    """partial[pid] = ...; old = atomic_add(counter, 1, sem); if old == N-1:
    read partial[0:N]. N = 4 (the grid size)."""
    counter_base = int(counter.data_ptr())
    old = Int("lbd_old")
    lane = Int("lbd_lane")
    pid = SymbolicExpr.PID0
    return (
        [
            _store(IntVal(PARTIAL_BASE) + pid * 4, event_id=0, program_seq=0),
            _rmw(
                IntVal(counter_base),
                old,
                op="add",
                operand=IntVal(1),
                event_id=1,
                program_seq=1,
                sem=sem,
                tensor=counter,
            ),
            _load(
                IntVal(PARTIAL_BASE) + lane * 4,
                event_id=2,
                program_seq=2,
                mask=old == IntVal(3),
            ),
        ],
        {(0, 4): (lane, None)},
    )


def test_last_block_done_acq_rel_is_proved():
    counter = _counter_tensor(0)
    records, arange_reg = _last_block_records(sem="acq_rel", counter=counter)
    solver = _solve(records, arange_dict=arange_reg)
    assert solver._counting, "counting axiom must fire for the counter"
    assert solver.find_races() == []


def test_last_block_done_relaxed_flips_to_race():
    """Mutation twin: dropping the release/acquire kills the sw edge."""
    counter = _counter_tensor(0)
    records, arange_reg = _last_block_records(sem="relaxed", counter=counter)
    reports = _solve(records, arange_dict=arange_reg).find_races()
    assert reports, "relaxed counter must expose the partial-store/read race"
    kinds = {(rep.first_record.op_type, rep.second_record.op_type) for rep in reports}
    assert (Store, Load) in kinds or (Load, Store) in kinds


def test_last_block_done_unknown_init_omits_axiom_and_over_reports():
    """Guard (d) failure: no tensor → no counting → the epilogue gate cannot
    be pinned, and the pair must be REPORTED (over-report direction), never
    silently proved."""
    counter = _counter_tensor(0)
    records, arange_reg = _last_block_records(sem="acq_rel", counter=counter)
    records[1].tensor = None
    solver = _solve(records, arange_dict=arange_reg)
    assert not solver._counting
    assert solver.find_races(), "omitted axiom must fall back to reporting"


# ──────────────────── single winner (counting distinctness) ────────────────


def _single_winner_records(*, counter):
    old = Int("sw_old")
    return [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=counter,
        ),
        _store(IntVal(DATA_BASE), event_id=1, program_seq=1, mask=old == IntVal(0)),
    ]


def test_single_winner_store_is_proved():
    """Only the rank-0 arriver stores: distinct observations make the
    guarded WAW unsatisfiable — no release/acquire needed for mutual
    exclusion by counting."""
    counter = _counter_tensor(0)
    solver = _solve(_single_winner_records(counter=counter))
    assert solver._counting
    assert solver.find_races() == []


def test_single_winner_without_init_flips_to_race():
    """Mutation twin: unknown initial value → axiom omitted → both copies
    may observe 0 via rf_unknown → the WAW must be reported."""
    counter = _counter_tensor(0)
    records = _single_winner_records(counter=counter)
    records[0].tensor = None
    solver = _solve(records)
    assert not solver._counting
    assert solver.find_races()


# ──────────────────── work queue (observation in address) ──────────────────


def test_work_queue_single_fetch_is_proved():
    """idx = atomic_add(head, 1); store buf[idx]: distinct ranks → distinct
    slots. The observation feeds the ADDRESS — allowed exactly because the
    counting axiom fired (B.1.5 carve-out)."""
    head = _counter_tensor(0)
    old = Int("wq_old")
    records = [
        _rmw(
            IntVal(int(head.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=head,
        ),
        _store(IntVal(BUF_BASE) + old * 4, event_id=1, program_seq=1),
    ]
    solver = _solve(records)
    assert solver._counting
    assert solver.find_races() == []


def test_work_queue_narrow_stride_flips_to_race():
    """Mutation twin: slot stride 2 < elem 4 → adjacent ranks overlap."""
    head = _counter_tensor(0)
    old = Int("wqn_old")
    records = [
        _rmw(
            IntVal(int(head.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=head,
        ),
        _store(IntVal(BUF_BASE) + old * 2, event_id=1, program_seq=1),
    ]
    assert _solve(records).find_races()


def test_observation_address_without_counting_is_unsupported():
    """B.5 boundary: an xchg observation (no counting) feeding an address
    must raise — not silently widen, not silently prove."""
    head = _counter_tensor(0)
    old = Int("gate_old")
    records = [
        _rmw(
            IntVal(int(head.data_ptr())),
            old,
            op="xchg",
            operand=IntVal(7),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=head,
        ),
        _store(IntVal(BUF_BASE) + old * 4, event_id=1, program_seq=1),
    ]
    with pytest.raises(UnsupportedSymbolicRaceQuery):
        _solve(records)


# ──────────────────── rf_chain: non-adjacent ranks stay SAT ─────────────────


def test_rf_chain_keeps_non_adjacent_rank_race_sat():
    """rank-0 stores X, rank-2 reads X, relaxed: a real race. Without the
    rf_chain source the closed world (init + the peer copy's write) cannot
    justify o == 2 next to o == 0 and the query would go UNSAT — the exact
    false-proof this choice exists to prevent."""
    counter = _counter_tensor(0)
    old = Int("chain_old")
    records = [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=counter,
        ),
        _store(IntVal(DATA_BASE), event_id=1, program_seq=1, mask=old == IntVal(0)),
        _load(IntVal(DATA_BASE), event_id=2, program_seq=2, mask=old == IntVal(2)),
    ]
    solver = _solve(records)
    assert solver._counting
    assert solver.find_races(), "the rank-0-store vs rank-2-read race is real"


# ──────────────────── release sequence through an RMW chain ────────────────


def _release_chain_records(*, producer_sem, modeled_middle=True):
    """Producer (pid 0): store data; xchg(flag, 10, producer_sem).
    Consumer (pid != 0): mid = cas(flag, 10, 15, relaxed) (a C11-RMW link
    with a modeled, wrap-free write); fetch2 = add(flag, 0, acquire);
    if fetch2 == 15: read data. The sw edge to fetch2 exists only THROUGH
    the relaxed middle RMW link. (The middle hop is a CAS rather than a
    nonzero add: an uncertified add's write is wrap-capable and now opens
    the rf_unknown escape by design — see the wraparound regression.)"""
    flag = _counter_tensor(0)
    flag_base = int(flag.data_ptr())
    pid = SymbolicExpr.PID0
    is_producer = pid == 0
    is_consumer = pid != 0
    old_mid = Int("rc_old_mid")
    old_fetch = Int("rc_old_fetch")
    if modeled_middle:
        middle = AccessEventRecord(
            op_type=AtomicRMW,
            access_mode="read",
            tensor=flag,
            addr_expr=IntVal(flag_base),
            active=is_consumer,
            reads=True,
            writes=None,
            is_atomic=True,
            atomic_kind="cas",
            sem="relaxed",
            scope="gpu",
            old_value=old_mid,
            cas_cmp_value=IntVal(10),
            cas_new_value=IntVal(15),
            event_id=2,
            program_seq=2,
            elem_size=4,
            copy_local_vars=(old_mid,),
        )
    else:
        middle = _rmw(
            IntVal(flag_base),
            old_mid,
            op="or",
            operand=IntVal(5),
            event_id=2,
            program_seq=2,
            sem="relaxed",
            tensor=flag,
            active=is_consumer,
        )
    records = [
        _store(IntVal(DATA_BASE), event_id=0, program_seq=0, active=is_producer),
        _rmw(
            IntVal(flag_base),
            Int("rc_old_x"),
            op="xchg",
            operand=IntVal(10),
            event_id=1,
            program_seq=1,
            sem=producer_sem,
            tensor=flag,
            active=is_producer,
        ),
        middle,
        _rmw(
            IntVal(flag_base),
            old_fetch,
            op="add",
            operand=IntVal(0),
            event_id=3,
            program_seq=3,
            sem="acquire",
            tensor=flag,
            active=is_consumer,
        ),
        _load(
            IntVal(DATA_BASE),
            event_id=4,
            program_seq=4,
            active=is_consumer,
            mask=old_fetch == IntVal(15),
        ),
    ]
    return records


def test_release_sequence_through_rmw_chain_is_proved():
    solver = _solve(_release_chain_records(producer_sem="release"), grid=(2, 1, 1))
    assert solver.find_races() == []


def test_release_sequence_relaxed_producer_flips_to_race():
    """Mutation twin: a relaxed publisher heads no release sequence."""
    reports = _solve(
        _release_chain_records(producer_sem="relaxed"), grid=(2, 1, 1)
    ).find_races()
    assert reports


def test_unmodeled_middle_rmw_breaks_the_chain_conservatively():
    """The middle hop as a bitwise `or` has no modeled write part: it opens
    the rf_unknown escape instead of extending the chain, so the pair is
    REPORTED (over-report direction), never proved through an unmodeled
    link."""
    reports = _solve(
        _release_chain_records(producer_sem="release", modeled_middle=False),
        grid=(2, 1, 1),
    ).find_races()
    assert reports


# ──────────────────── guard failures omit the axiom ────────────────────


def test_counting_omitted_for_masked_rmw():
    """Guard (b): a pid-gated RMW is not always-active."""
    counter = _counter_tensor(0)
    old = Int("masked_old")
    records = [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            tensor=counter,
            active=SymbolicExpr.PID0 == 0,
        ),
    ]
    assert not _solve(records)._counting


def test_counting_omitted_when_plain_store_overlaps_counter():
    """Guard (e): any other possible writer to L kills the axiom."""
    counter = _counter_tensor(0)
    base = int(counter.data_ptr())
    old = Int("ovl_old")
    records = [
        _rmw(
            IntVal(base),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            tensor=counter,
        ),
        _store(IntVal(base), event_id=1, program_seq=1),
    ]
    assert not _solve(records)._counting


def test_counting_omitted_for_nonconstant_increment():
    """Guard (c): a pid-dependent increment has no single c."""
    counter = _counter_tensor(0)
    old = Int("nc_old")
    records = [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=SymbolicExpr.PID0 + 1,
            event_id=0,
            program_seq=0,
            tensor=counter,
        ),
    ]
    assert not _solve(records)._counting


def test_unmodeled_rmw_still_opens_escape_for_modeled_reader():
    """A value-modeled reader next to an UNMODELED writer at the same
    location must keep the rf_unknown escape (closed-world honesty)."""
    flag = _counter_tensor(0)
    base = int(flag.data_ptr())
    records = [
        _rmw(
            IntVal(base),
            Int("esc_old_r"),
            op="add",
            operand=IntVal(0),
            event_id=0,
            program_seq=0,
            sem="acquire",
            tensor=flag,
        ),
        # bitwise or: observation modeled? No — written unmodeled and we
        # give it NO observation var either (old=None path): plain record.
        AccessEventRecord(
            op_type=AtomicRMW,
            access_mode="read",
            addr_expr=IntVal(base),
            active=True,
            reads=True,
            writes=True,
            is_atomic=True,
            atomic_kind="rmw",
            sem="relaxed",
            scope="gpu",
            event_id=1,
            program_seq=1,
            elem_size=4,
        ),
    ]
    solver = _solve(records)
    assert (
        0 in {e.idx for e in solver.events if e.record is records[0]}
        or solver.rf_unknown_source
    ), "escape bookkeeping missing"
    assert solver.rf_unknown_source, "unmodeled overlapping writer must open rf_unknown"


# ──────────── adversarial regressions (S6 verification round) ────────────


def test_rank0_winner_duplicate_lane_waw_reported():
    """Adversarial finding 1: the single-winner (o == 0) block's
    duplicate-lane store WAW was structurally UNSAT — the intra-instance
    query pinned copy-locals equal while coherence still demanded two
    distinct positions and rf sources for the ONE dynamic RMW. The twins
    must be identified as one op when pids coincide."""
    counter = _counter_tensor(0)
    old = Int("r0_old")
    records = [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=counter,
        ),
        # TWO lanes to ONE address, gated on the rank-0 observation: the
        # winner's own lanes collide in every real execution.
        _store(
            [IntVal(DATA_BASE), IntVal(DATA_BASE)],
            event_id=1,
            program_seq=1,
            mask=old == IntVal(0),
        ),
    ]
    solver = _solve(records)
    assert solver._counting
    assert solver.find_races(), "the rank-0 winner's duplicate-lane WAW is real"


def test_rank0_cas_winner_duplicate_lane_waw_reported():
    """Same identification for the CAS analogue (the try-lock winner)."""
    lock = _counter_tensor(0)
    old = Int("r0c_old")
    records = [
        AccessEventRecord(
            op_type=AtomicRMW,  # op_type is informational; kind drives logic
            access_mode="read",
            tensor=lock,
            addr_expr=IntVal(int(lock.data_ptr())),
            active=True,
            reads=True,
            writes=None,
            is_atomic=True,
            atomic_kind="cas",
            sem="acq_rel",
            scope="gpu",
            old_value=old,
            cas_cmp_value=IntVal(0),
            cas_new_value=IntVal(1),
            event_id=0,
            program_seq=0,
            elem_size=4,
            copy_local_vars=(old,),
        ),
        _store(
            [IntVal(DATA_BASE), IntVal(DATA_BASE)],
            event_id=1,
            program_seq=1,
            mask=old == IntVal(0),
        ),
    ]
    assert _solve(records).find_races(), "the lock winner's lane WAW is real"


def test_torn_width_modeled_writer_opens_escape():
    """Adversarial finding 2/3: an 8-byte MODELED xchg over a 4-byte
    modeled reader is rf-incompatible (width mismatch) — it must count as
    an unmodeled overlapping writer and open the rf_unknown escape instead
    of pinning the reader to the initial value."""
    flag = _counter_tensor(0)
    base = int(flag.data_ptr())
    old_r = Int("torn_old_r")
    old_w = Int("torn_old_w")
    records = [
        _rmw(
            IntVal(base),
            old_w,
            op="xchg",
            operand=IntVal(10),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            elem_size=8,  # 8-byte writer
        ),
        _rmw(
            IntVal(base),
            old_r,
            op="add",
            operand=IntVal(0),
            event_id=1,
            program_seq=1,
            sem="acquire",
            elem_size=4,  # 4-byte reader
            tensor=flag,
        ),
        _store(IntVal(DATA_BASE), event_id=2, program_seq=2, mask=old_r == IntVal(10)),
        _store(IntVal(DATA_BASE), event_id=3, program_seq=3),
    ]
    solver = _solve(records)
    reader_events = [e for e in solver.events if e.record is records[1]]
    assert any(
        e.idx in solver.rf_unknown_source for e in reader_events
    ), "torn overlap must open rf_unknown"
    assert solver.find_races(), "the gated store can really execute (torn read)"


def test_counting_omitted_on_possible_wraparound():
    """Adversarial finding 5: an INT32_MAX-initialized counter really
    wraps on hardware while the unbounded-Int model does not — guard (f)
    must omit the axiom (over-report direction)."""
    counter = _counter_tensor(0)
    counter[0] = torch.iinfo(torch.int32).max
    old = Int("wrap_old")
    records = [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=counter,
        ),
        _store(IntVal(DATA_BASE), event_id=1, program_seq=1),
        _store(
            IntVal(DATA_BASE),
            event_id=2,
            program_seq=2,
            mask=old == IntVal(-(1 << 31)),
        ),
    ]
    solver = _solve(records)
    assert not solver._counting, "guard (f): possible wrap must omit the axiom"
    assert solver.find_races(), "the wrap-gated store races with the plain store"


def test_counting_survives_wraparound_guard_for_sane_counter():
    """Guard (f) must NOT kill the ordinary zero-initialized counter under
    a symbolic grid (the CUDA launch caps bound the reachable sum)."""
    counter = _counter_tensor(0)
    old = Int("sane_old")
    records = [
        _rmw(
            IntVal(int(counter.data_ptr())),
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            sem="relaxed",
            tensor=counter,
        ),
    ]
    solver = TwoCopySymbolicHBSolver(
        records, grid=(Int("grid_0"), 1, 1), arange_dict={}
    )
    assert solver._counting, "int32 counter with init 0 cannot wrap within caps"


def test_multilane_modeled_rmw_constructs_and_skips_counting():
    """A vector RMW (two lanes) is outside the counting guard but must
    still lower cleanly with a broadcast observation var."""
    counter = torch.zeros(2, dtype=torch.int32)
    base = int(counter.data_ptr())
    old = Int("ml_old")
    records = [
        _rmw(
            [IntVal(base), IntVal(base + 4)],
            old,
            op="add",
            operand=IntVal(1),
            event_id=0,
            program_seq=0,
            tensor=counter,
        ),
    ]
    solver = _solve(records)
    assert not solver._counting
    solver.find_races()  # must not raise
