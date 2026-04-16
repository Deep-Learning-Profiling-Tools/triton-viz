"""Step 5 (HBSolver) + Step 6 (epoch partitioning) invariants.

Hand-built AccessEventRecords exercise each edge class in isolation so
the po / sw / reads-from / ambiguous-writer / phase-advance rules can be
independently regressed. Kernel-level coverage is in
``tests/end_to_end/test_race_detector_sync.py``.
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl

import triton_viz

from triton_viz.clients.race_detector.data import (
    AccessEventRecord,
    RaceCandidate,
    RaceType,
)
from triton_viz.core.config import config as cfg
from triton_viz.clients.race_detector.hb_solver import HBSolver
from triton_viz.clients.race_detector.race_detector import (
    SymbolicRaceDetector,
    _is_phase_advancing_write,
)
from triton_viz.core.data import AtomicCas, AtomicRMW, Load, Store


def _make_plain(
    *,
    event_id: int,
    grid_idx: tuple[int, ...],
    program_seq: int,
    lane_addrs: list[int],
    access_mode: str,
    elem_size: int = 4,
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


def _make_atomic(
    *,
    event_id: int,
    grid_idx: tuple[int, ...],
    program_seq: int,
    lane_addrs: list[int],
    atomic_op: str,
    atomic_old: list[int],
    written_value: list[int],
    atomic_sem: str,
    atomic_scope: str,
    elem_size: int = 4,
    epoch: int = 0,
    op_type: type = AtomicRMW,
    atomic_val: list[int] | None = None,
) -> AccessEventRecord:
    lane_arr = np.array(lane_addrs, dtype=np.int64)
    active = np.ones(len(lane_addrs), dtype=bool)
    return AccessEventRecord(
        event_id=event_id,
        op_type=op_type,
        grid_idx=grid_idx,
        program_seq=program_seq,
        lane_addrs=lane_arr,
        active_mask=active,
        elem_size=elem_size,
        read_mask=active.copy(),
        write_mask=active.copy(),
        atomic_op=atomic_op,
        atomic_sem=atomic_sem,
        atomic_scope=atomic_scope,
        atomic_old=np.array(atomic_old, dtype=np.int32),
        atomic_val=np.array(atomic_val or written_value, dtype=np.int32),
        written_value=np.array(written_value, dtype=np.int32),
        epoch=epoch,
    )


# ── po construction ───────────────────────────────────────────────────────


def test_hb_solver_builds_po_from_program_seq():
    """po must key on ``program_seq``, not ``event_id`` — under multi-SM
    execution ``event_id`` is append order, not program order (upstream #346)."""
    a = _make_plain(
        event_id=42,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[100],
        access_mode="write",
    )
    b = _make_plain(
        event_id=5,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[100],
        access_mode="read",
    )
    solver = HBSolver([a, b])
    cand = RaceCandidate(
        first=a,
        second=b,
        race_type=RaceType.RAW,
        witness_addr=100,
        epoch=0,
    )
    result = solver.check_candidate(cand)
    assert result.ordered, "po must order same-grid events by program_seq"
    assert result.reason == "po_path"


# ── sw: reads-from gate ───────────────────────────────────────────────────


def test_release_acquire_requires_reads_from():
    """Matching sem / scope alone does not create sw. The reader must
    observe the writer's written value (#345)."""
    # Producer: plain store THEN release. Consumer: acquire THEN plain load.
    # Program-order: store@seq0 -> release@seq1 (block 0)
    #                acquire@seq0 -> load@seq1   (block 1)
    plain_store = _make_plain(
        event_id=3,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[500],
        access_mode="write",
    )
    writer = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[0],
        written_value=[1],
        atomic_sem="release",
        atomic_scope="gpu",
    )
    # Reader sees a DIFFERENT value (not 1) -> no sw edge.
    bad_reader = _make_atomic(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[7],
        written_value=[2],
        atomic_sem="acquire",
        atomic_scope="gpu",
    )
    plain_load = _make_plain(
        event_id=4,
        grid_idx=(1, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="read",
    )
    solver = HBSolver([plain_store, writer, bad_reader, plain_load])
    cand = RaceCandidate(
        first=plain_store,
        second=plain_load,
        race_type=RaceType.RAW,
        witness_addr=500,
        epoch=0,
    )
    assert not solver.check_candidate(
        cand
    ).ordered, "reads-from must fail when reader.atomic_old != writer.written_value"

    # Reader now observes 1 (writer's written_value) -> sw edge formed.
    good_reader = _make_atomic(
        event_id=5,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[1],
        written_value=[2],
        atomic_sem="acquire",
        atomic_scope="gpu",
    )
    solver2 = HBSolver([plain_store, writer, good_reader, plain_load])
    result = solver2.check_candidate(cand)
    assert result.ordered
    assert result.reason == "release_acquire_reads_from"


def test_relaxed_atomic_does_not_create_sw():
    writer = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[0],
        written_value=[1],
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )
    reader = _make_atomic(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[1],
        written_value=[2],
        atomic_sem="relaxed",
        atomic_scope="gpu",
    )
    plain_store = _make_plain(
        event_id=3,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="write",
    )
    plain_load = _make_plain(
        event_id=4,
        grid_idx=(1, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="read",
    )
    solver = HBSolver([plain_store, writer, reader, plain_load])
    cand = RaceCandidate(
        first=plain_store,
        second=plain_load,
        race_type=RaceType.RAW,
        witness_addr=500,
        epoch=0,
    )
    assert not solver.check_candidate(cand).ordered


def test_cta_scope_does_not_create_sw_or_barrier():
    """CTA scope is block-local and not valid as cross-block sync (#345)."""
    writer = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[0],
        written_value=[1],
        atomic_sem="release",
        atomic_scope="cta",
    )
    reader = _make_atomic(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[1],
        written_value=[2],
        atomic_sem="acquire",
        atomic_scope="cta",
    )
    plain_store = _make_plain(
        event_id=3,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="write",
    )
    plain_load = _make_plain(
        event_id=4,
        grid_idx=(1, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="read",
    )
    solver = HBSolver([plain_store, writer, reader, plain_load])
    cand = RaceCandidate(
        first=plain_store,
        second=plain_load,
        race_type=RaceType.RAW,
        witness_addr=500,
        epoch=0,
    )
    assert not solver.check_candidate(cand).ordered


# ── Same-value ABA ambiguity ──────────────────────────────────────────────


def test_ambiguous_same_value_writer_blocks_sw():
    """If a third-party writer also wrote the same value, the reader's
    atomic_old can't uniquely attribute back to ``wi`` (#350)."""
    wi = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[0],
        written_value=[1],
        atomic_sem="release",
        atomic_scope="gpu",
    )
    # Third block ALSO writes value 1 at addr 200 -> ambiguity.
    w_ambig = _make_atomic(
        event_id=2,
        grid_idx=(2, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[0],
        written_value=[1],
        atomic_sem="release",
        atomic_scope="gpu",
    )
    ri = _make_atomic(
        event_id=3,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[1],
        written_value=[2],
        atomic_sem="acquire",
        atomic_scope="gpu",
    )
    plain_store = _make_plain(
        event_id=4,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="write",
    )
    plain_load = _make_plain(
        event_id=5,
        grid_idx=(1, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="read",
    )
    solver = HBSolver([plain_store, wi, w_ambig, ri, plain_load])
    cand = RaceCandidate(
        first=plain_store,
        second=plain_load,
        race_type=RaceType.RAW,
        witness_addr=500,
        epoch=0,
    )
    assert not solver.check_candidate(
        cand
    ).ordered, "same-value third-party writer must block sw (#350)"


def test_reader_self_same_value_cas_writeback_is_not_ambiguous():
    """``cas(1, 1)`` polling loops have the reader itself write back the
    same value it saw (#347). The ambiguity scan must skip ``ri`` so this
    doesn't falsely block sw from the real writer.
    """
    plain_store = _make_plain(
        event_id=3,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[500],
        access_mode="write",
    )
    wi = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=1,
        lane_addrs=[200],
        atomic_op="xchg",
        atomic_old=[0],
        written_value=[1],
        atomic_sem="release",
        atomic_scope="gpu",
    )
    # Reader is a CAS that writes back 1 on success.
    ri = _make_atomic(
        event_id=2,
        grid_idx=(1, 0, 0),
        program_seq=0,
        lane_addrs=[200],
        atomic_op="cas",
        atomic_old=[1],
        written_value=[1],
        atomic_sem="acquire",
        atomic_scope="gpu",
        op_type=AtomicCas,
    )
    plain_load = _make_plain(
        event_id=4,
        grid_idx=(1, 0, 0),
        program_seq=1,
        lane_addrs=[500],
        access_mode="read",
    )
    solver = HBSolver([plain_store, wi, ri, plain_load])
    cand = RaceCandidate(
        first=plain_store,
        second=plain_load,
        race_type=RaceType.RAW,
        witness_addr=500,
        epoch=0,
    )
    result = solver.check_candidate(cand)
    assert (
        result.ordered
    ), "reader's own writeback must not trip the ambiguity check (#347)"
    assert result.reason == "release_acquire_reads_from"


# ── Phase-advance epoch bump ──────────────────────────────────────────────


def test_epoch_does_not_advance_on_failed_cas():
    """Failed CAS writes the pre-op value back (written_value == old).
    Step 6 must NOT count this as a phase advance."""
    failed_cas = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[300],
        atomic_op="cas",
        atomic_old=[7],
        written_value=[7],
        atomic_sem="acq_rel",
        atomic_scope="gpu",
        op_type=AtomicCas,
    )
    assert not _is_phase_advancing_write(failed_cas, 300)


def test_epoch_does_not_advance_on_same_value_rmw_add_zero():
    """``atomic_add(0)`` touches the barrier address but leaves the value
    unchanged. Conservative epoch partitioning must keep it in-phase."""
    rmw_add_zero = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[300],
        atomic_op="add",
        atomic_old=[5],
        written_value=[5],
        atomic_sem="acq_rel",
        atomic_scope="gpu",
    )
    assert not _is_phase_advancing_write(rmw_add_zero, 300)


def test_epoch_advances_on_real_phase_bump():
    """Sanity counterpart: a true value change DOES advance the phase."""
    inc = _make_atomic(
        event_id=1,
        grid_idx=(0, 0, 0),
        program_seq=0,
        lane_addrs=[300],
        atomic_op="add",
        atomic_old=[5],
        written_value=[6],
        atomic_sem="acq_rel",
        atomic_scope="gpu",
    )
    assert _is_phase_advancing_write(inc, 300)


# ── program_seq propagation through atomics ───────────────────────────────


@triton.jit
def _mix_for_program_seq(x_ptr, flag_ptr):
    tl.store(x_ptr, 1)
    tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")
    tl.atomic_cas(flag_ptr, 1, 2, sem="acquire", scope="gpu")


def test_atomic_events_carry_program_seq():
    """Step 2 invariant: both atomic and plain paths must stamp
    program_seq and launch_id — the HB solver's po, the epoch
    partitioner, and the ambiguity scan all depend on it."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = True
        detector = SymbolicRaceDetector(abort_on_error=False)
        traced = triton_viz.trace(client=detector)(_mix_for_program_seq)
        x = torch.zeros(1, dtype=torch.int32)
        flag = torch.zeros(1, dtype=torch.int32)
        traced[(1,)](x, flag)
        # Each block's concrete events must have consecutive program_seq.
        assert [e.program_seq for e in detector.concrete_events] == [0, 1, 2]
        assert {e.launch_id for e in detector.concrete_events} == {detector._launch_id}
    finally:
        cfg.enable_race_detector = saved
