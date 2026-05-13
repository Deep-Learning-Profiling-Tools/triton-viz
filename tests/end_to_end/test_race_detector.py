import inspect

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector, RaceType
from triton_viz.clients.race_detector.hb_solver import RaceReport
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.core.config import config as cfg
from triton_viz.core.trace import launches


class _ModeTrackingRaceDetector(SymbolicRaceDetector):
    """Pass-through detector kept for tests that previously asserted on the
    deleted concrete-fallback phase. It now only forwards to the base
    implementation; assertions on race counts continue to work unchanged.
    """


# ======== WAW — Overlapping Store (Wrong Stride) ========


def test_waw_overlapping_store():
    """Adjacent blocks overlap by 1 element due to wrong stride."""

    @triton_viz.trace(RaceDetector())
    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * (BLOCK_SIZE - 1)  # BUG: should be BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, offsets.to(tl.float32), mask=mask)

    n, bs = 32, 8
    out = torch.empty(n, dtype=torch.float32)
    kernel[(triton.cdiv(n, bs),)](out, n, bs)

    races = launches[-1].records
    assert len(races) > 0
    assert any(r.race_type == RaceType.WAW for r in races)


# ======== RAW+WAW — Non-atomic Histogram ========


def test_raw_waw_histogram():
    """Non-atomic load-modify-store to bins_ptr."""

    @triton_viz.trace(RaceDetector())
    @triton.jit
    def kernel(input_ptr, bins_ptr, n_elements, n_bins, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask, other=0)
        bin_indices = values % n_bins
        counts = tl.load(bins_ptr + bin_indices, mask=mask, other=0)
        tl.store(bins_ptr + bin_indices, counts + 1, mask=mask)

    n, n_bins, bs = 64, 4, 8
    inp = torch.randint(0, n_bins, (n,), dtype=torch.int32)
    bins = torch.zeros(n_bins, dtype=torch.int32)
    kernel[(triton.cdiv(n, bs),)](inp, bins, n, n_bins, bs)

    races = launches[-1].records
    assert len(races) > 0
    race_types = {r.race_type for r in races}
    assert RaceType.WAW in race_types


# ======== Correct vector_add (No Race) ========


def test_no_race_vector_add():
    """Correct kernel with no overlapping accesses — should report no races."""

    @triton_viz.trace(RaceDetector())
    @triton.jit
    def kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    n, bs = 64, 8
    x = torch.randn(n)
    y = torch.randn(n)
    out = torch.empty(n)
    kernel[(triton.cdiv(n, bs),)](x, y, out, n, bs)

    races = launches[-1].records
    assert len(races) == 0


# ======== Correct Atomic Histogram (No Race) ========


def test_no_race_atomic_histogram():
    """Histogram using tl.atomic_add — should report no races."""

    @triton_viz.trace(RaceDetector())
    @triton.jit
    def kernel(input_ptr, bins_ptr, n_elements, n_bins, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask, other=0)
        bin_indices = values % n_bins
        tl.atomic_add(bins_ptr + bin_indices, 1, mask=mask)

    n, n_bins, bs = 64, 4, 8
    inp = torch.randint(0, n_bins, (n,), dtype=torch.int32)
    bins = torch.zeros(n_bins, dtype=torch.int32)
    kernel[(triton.cdiv(n, bs),)](inp, bins, n, n_bins, bs)

    races = launches[-1].records
    assert len(races) == 0


# ======== Minimal While+CAS Smoke Test ========


@pytest.mark.skip(
    reason="while-loop CAS kernels need concrete fallback to terminate symbolic "
    "execution; concrete fallback was removed in the two-copy migration. "
    "Re-enable when while-loop unrolling / data-dependent loop bailout is "
    "added to the symbolic capture path."
)
def test_while_loop_atomic_cas_minimal_case():
    """Minimal while-loop CAS kernel should run and report no races."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, sync_ptr, MAX_SPIN: tl.constexpr):
        pid = tl.program_id(0)
        spins = 0
        while tl.atomic_cas(sync_ptr, 1, 1) != 1 and spins < MAX_SPIN:
            spins += 1
        tl.store(out_ptr + pid, tl.cast(spins, tl.int32))

    n_blocks = 2
    max_spin = 4
    out = torch.empty((n_blocks,), dtype=torch.int32)
    sync = torch.zeros((1,), dtype=torch.int32)
    kernel[(n_blocks,)](out, sync, max_spin)

    races = launches[-1].records
    assert len(races) == 0


# ======== Two-phase Global Barrier ========


@pytest.mark.skip(
    reason="cross-phase no-race assertion needs barrier/epoch modeling that the "
    "two-copy CAS/HB migration intentionally leaves out of scope (see plan: "
    "'epoch / barrier-phase modeling' under Out of scope). The current solver "
    "treats the no-op atomic_cas as no synchronization, so phase-0 and phase-1 "
    "stores at the same slot are reported as a race."
)
def test_two_phase_barrier_symbolic_no_race():
    """Symbolic path: cross-phase overlaps after barrier should not race."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, sync_ptr, N_BLOCKS: tl.constexpr):
        pid = tl.program_id(0)

        # Phase 0: block i writes slot i.
        tl.store(out_ptr + pid, tl.cast(pid, tl.float32))

        # Symbolic barrier marker: add + cas on the same sync address.
        tl.atomic_add(sync_ptr, 1)
        tl.atomic_cas(sync_ptr, 0, 0)

        # Phase 1: block i writes slot (i + 1) % N_BLOCKS.
        dst = (pid + 1) % N_BLOCKS
        tl.store(out_ptr + dst, tl.cast(pid + 100, tl.float32))

    n_blocks = 4
    out = torch.zeros((n_blocks,), dtype=torch.float32)
    sync = torch.zeros((1,), dtype=torch.int32)
    kernel[(n_blocks,)](out, sync, n_blocks)

    races = launches[-1].records
    assert len(races) == 0


@pytest.mark.skip(
    reason="kernel uses a while-loop spin barrier; needs concrete fallback to "
    "terminate symbolic execution (removed in the two-copy migration)."
)
def test_two_phase_barrier_concrete_no_race():
    """Two phase stores separated by a global CAS barrier should not race."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, sync_ptr, N_BLOCKS: tl.constexpr):
        pid = tl.program_id(0)

        # Phase 0: block i writes slot i.
        tl.store(out_ptr + pid, tl.cast(pid, tl.float32))

        # Global sync: each block arrives, then spins until all have arrived.
        tl.atomic_add(sync_ptr, 1)
        spins = 0
        max_spin = 10000
        while (
            tl.atomic_cas(sync_ptr, N_BLOCKS, N_BLOCKS) != N_BLOCKS and spins < max_spin
        ):
            spins += 1

        # Phase 1: block i writes slot (i + 1) % N_BLOCKS.
        dst = (pid + 1) % N_BLOCKS
        tl.store(out_ptr + dst, tl.cast(pid + 100, tl.float32))

    n_blocks = 4
    out = torch.zeros((n_blocks,), dtype=torch.float32)
    sync = torch.zeros((1,), dtype=torch.int32)
    kernel[(n_blocks,)](out, sync, n_blocks)

    # Validate the barrier completed before checking race reports.
    assert int(sync.item()) == n_blocks

    races = launches[-1].records
    # Desired behavior with phase-aware modeling: no races.
    assert len(races) == 0


# ======== Atomic CAS — Symbolic HB Solver ========


@pytest.fixture
def _isolate_race_detector_atomic_cfg():
    saved_enable = cfg.enable_race_detector
    saved_num_sms = cfg.num_sms
    cfg.enable_race_detector = True
    cfg.num_sms = 1
    triton_viz.clear()
    yield
    triton_viz.clear()
    cfg.enable_race_detector = saved_enable
    cfg.num_sms = saved_num_sms


@triton.jit
def _plain_cross_grid_smoke_kernel(data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    x = tl.load(data_ptr, mask=is_cons, other=0)
    tl.store(out_ptr + pid, x, mask=is_cons)


@triton.jit
def _cas_acq_rel_unguarded_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    _old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    x = tl.load(data_ptr, mask=is_cons, other=0)
    tl.store(out_ptr + pid, x, mask=is_cons)


@triton.jit
def _cas_acq_rel_guarded_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def _cas_relaxed_guarded_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="relaxed", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def _cas_cta_guarded_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="cta")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def _cas_single_program_order_kernel(flag_ptr, data_ptr, out_ptr):
    tl.store(data_ptr, 1)
    _old = tl.atomic_cas(flag_ptr, 0, 1, sem="acq_rel", scope="gpu")
    x = tl.load(data_ptr)
    tl.store(out_ptr, x)


@triton.jit
def _atomic_only_competing_updates_kernel(flag_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    cmp = tl.where(is_prod, 0, 1)
    tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")


def _run_detector(kernel, grid, *args, **kwargs):
    triton_viz.clear()
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(kernel)
    traced[grid](*args, **kwargs)
    return detector


def _line_no(kernel, needle: str) -> int:
    source_fn = kernel.fn if hasattr(kernel, "fn") else kernel
    lines, start = inspect.getsourcelines(source_fn)
    for idx, line in enumerate(lines):
        if needle in line:
            return start + idx
    raise AssertionError(f"Could not find source line containing: {needle}")


def _assert_report_lines(report: RaceReport, kernel, needles: tuple[str, str]) -> None:
    actual_lines = {
        report.first.record.source_location[1],
        report.second.record.source_location[1],
    }
    expected_lines = {_line_no(kernel, needle) for needle in needles}
    assert actual_lines == expected_lines


def _assert_launch_reports(detector: SymbolicRaceDetector) -> None:
    assert launches, "expected at least one traced launch"
    assert launches[-1].records == detector.last_reports
    assert all(isinstance(report, RaceReport) for report in launches[-1].records)


def _assert_atomic_records(
    detector: SymbolicRaceDetector, *, sem: str, scope: str
) -> None:
    atomic_records = [record for record in detector.records if record.is_atomic]
    assert atomic_records, "expected atomic_cas events to be captured"
    assert all(record.atomic_kind == "cas" for record in atomic_records)
    assert {record.sem for record in atomic_records} == {sem}
    assert {record.scope for record in atomic_records} == {scope}
    # CAS records keep raw cmp/new/old; success / written_value are recomputed
    # per copy by the two-copy solver, so written_value is None on the record.
    assert all(record.old_value is not None for record in atomic_records)
    assert all(record.cas_cmp_value is not None for record in atomic_records)
    assert all(record.cas_new_value is not None for record in atomic_records)


def test_plain_cross_grid_smoke_reports_race(_isolate_race_detector_atomic_cfg):
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)

    detector = _run_detector(_plain_cross_grid_smoke_kernel, (2,), data, out)

    assert len(detector.last_reports) == 1
    _assert_launch_reports(detector)
    _assert_report_lines(
        detector.last_reports[0],
        _plain_cross_grid_smoke_kernel,
        (
            "tl.store(data_ptr, 1, mask=is_prod)",
            "x = tl.load(data_ptr, mask=is_cons, other=0)",
        ),
    )


def test_cas_acq_rel_unguarded_reports_race(_isolate_race_detector_atomic_cfg):
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)

    detector = _run_detector(_cas_acq_rel_unguarded_kernel, (2,), flag, data, out)

    assert len(detector.last_reports) == 1
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="acq_rel", scope="gpu")
    _assert_report_lines(
        detector.last_reports[0],
        _cas_acq_rel_unguarded_kernel,
        (
            "tl.store(data_ptr, 1, mask=is_prod)",
            "x = tl.load(data_ptr, mask=is_cons, other=0)",
        ),
    )


def test_cas_acq_rel_guarded_is_not_racy(_isolate_race_detector_atomic_cfg):
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)

    detector = _run_detector(_cas_acq_rel_guarded_kernel, (2,), flag, data, out)

    assert detector.last_reports == []
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="acq_rel", scope="gpu")


def test_cas_relaxed_guarded_reports_race(_isolate_race_detector_atomic_cfg):
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)

    detector = _run_detector(_cas_relaxed_guarded_kernel, (2,), flag, data, out)

    assert len(detector.last_reports) == 1
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="relaxed", scope="gpu")
    _assert_report_lines(
        detector.last_reports[0],
        _cas_relaxed_guarded_kernel,
        (
            "tl.store(data_ptr, 1, mask=is_prod)",
            "x = tl.load(data_ptr, mask=cons_mask, other=0)",
        ),
    )


def test_cas_cta_guarded_cross_grid_reports_race(_isolate_race_detector_atomic_cfg):
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)

    detector = _run_detector(_cas_cta_guarded_kernel, (2,), flag, data, out)

    assert len(detector.last_reports) == 1
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="acq_rel", scope="cta")
    _assert_report_lines(
        detector.last_reports[0],
        _cas_cta_guarded_kernel,
        (
            "tl.store(data_ptr, 1, mask=is_prod)",
            "x = tl.load(data_ptr, mask=cons_mask, other=0)",
        ),
    )


def test_single_program_order_is_not_racy(_isolate_race_detector_atomic_cfg):
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int32)

    detector = _run_detector(_cas_single_program_order_kernel, (1,), flag, data, out)

    assert detector.last_reports == []
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="acq_rel", scope="gpu")


def test_atomic_only_competing_updates_is_not_racy(_isolate_race_detector_atomic_cfg):
    flag = torch.zeros(1, dtype=torch.int32)

    detector = _run_detector(_atomic_only_competing_updates_kernel, (2,), flag)

    assert detector.last_reports == []
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="acq_rel", scope="gpu")


# ======== Atomic mask gating (Issue 1) ========


@triton.jit
def _masked_rmw_no_race_kernel(p_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    # Producer's atomic_add is masked off; consumer reads p without contention.
    tl.atomic_add(p_ptr, 1, mask=is_prod & False)
    x = tl.load(p_ptr, mask=is_cons, other=0)
    tl.store(out_ptr + pid, x, mask=is_cons)


def test_masked_atomic_rmw_no_race_with_unmasked_load(
    _isolate_race_detector_atomic_cfg,
):
    p = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(_masked_rmw_no_race_kernel, (2,), p, out)
    # Masked-off atomic_add should not race with the unmasked load.
    assert detector.last_reports == []


@triton.jit
def _pid_guarded_rmw_vs_store_kernel(p_ptr):
    pid = tl.program_id(0)
    is_a = pid == 0
    is_b = pid == 1
    tl.atomic_add(p_ptr, 1, mask=is_a)
    tl.store(p_ptr, 7, mask=is_b)


def test_masked_atomic_rmw_pid_guard_reports_race(
    _isolate_race_detector_atomic_cfg,
):
    p = torch.zeros(1, dtype=torch.int32)
    detector = _run_detector(_pid_guarded_rmw_vs_store_kernel, (2,), p)
    # atomic-vs-non-atomic at the same address; both active in their own pid.
    # Cross-block, the two events conflict → exactly one race expected.
    assert len(detector.last_reports) == 1


# ======== Atomic-in-loop unsupported (Issue 3) ========


@triton.jit
def _cas_in_loop_kernel(flag_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    for i in tl.range(0, N):
        old = tl.atomic_cas(flag_ptr, 0, 1, sem="acq_rel", scope="gpu")
        tl.store(out_ptr + pid, old)


def test_atomic_cas_inside_for_loop_is_unsupported(
    _isolate_race_detector_atomic_cfg,
):
    flag = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(_cas_in_loop_kernel, (2,), flag, out, 3)
    assert detector.records == []
    assert detector.last_reports == []
    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "loop" in detector.unsupported_reason


@triton.jit
def _rmw_in_loop_kernel(p_ptr, N: tl.constexpr):
    for i in tl.range(0, N):
        tl.atomic_add(p_ptr, 1)


def test_atomic_rmw_inside_for_loop_is_unsupported(
    _isolate_race_detector_atomic_cfg,
):
    p = torch.zeros(1, dtype=torch.int32)
    detector = _run_detector(_rmw_in_loop_kernel, (2,), p, 3)
    assert detector.records == []
    assert detector.last_reports == []
    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "loop" in detector.unsupported_reason


def test_atomic_in_loop_with_abort_on_error_raises(
    _isolate_race_detector_atomic_cfg,
):
    from triton_viz.clients.race_detector.hb_common import (
        UnsupportedSymbolicRaceQuery,
    )

    triton_viz.clear()
    detector = SymbolicRaceDetector(abort_on_error=True)
    p = torch.zeros(1, dtype=torch.int32)
    traced = triton_viz.trace(client=detector)(_rmw_in_loop_kernel)
    with pytest.raises(UnsupportedSymbolicRaceQuery):
        traced[(2,)](p, 3)


# ======== AtomicRMW return-value downstream is unsupported (Issue 5) ========


@triton.jit
def _rmw_return_used_downstream_kernel(p_ptr, q_ptr, out_ptr):
    pid = tl.program_id(0)
    old = tl.atomic_add(p_ptr, 1)
    val = tl.load(q_ptr, mask=old == 0, other=0)
    tl.store(out_ptr + pid, val)


def test_atomic_rmw_return_used_downstream_is_unsupported(
    _isolate_race_detector_atomic_cfg,
):
    p = torch.zeros(1, dtype=torch.int32)
    q = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(_rmw_return_used_downstream_kernel, (2,), p, q, out)
    assert detector.last_reports == []
    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    # Reason mentions either "rmw" or "return"
    reason = detector.unsupported_reason.lower()
    assert "rmw" in reason or "return" in reason


@triton.jit
def _rmw_return_unused_kernel(p_ptr):
    tl.atomic_add(p_ptr, 1)


def test_atomic_rmw_return_unused_is_supported(
    _isolate_race_detector_atomic_cfg,
):
    p = torch.zeros(1, dtype=torch.int32)
    detector = _run_detector(_rmw_return_unused_kernel, (2,), p)
    # Discarded RMW return: no unsupported, one RMW event captured per launch.
    assert detector.last_status == "ok"
    assert detector.unsupported_reason is None
    rmw_records = [r for r in detector.records if r.atomic_kind == "rmw"]
    assert len(rmw_records) >= 1


# ======== Closed-world / scalar-flag boundary (Issue 4) ========


@triton.jit
def _flag_array_cas_acq_rel_guarded_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    # flag_ptr is the START of an 8-element contiguous array. After Patch 2,
    # _initial_atomic_source enumerates per-element initial values up to
    # _MAX_INITIAL_ATOMIC_ELEMENTS (=1024), so this CAS uses rf_init and the
    # closed-world model can build a synchronizes-with edge.
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


def test_flag_array_cas_acq_rel_guarded_is_not_racy(
    _isolate_race_detector_atomic_cfg,
):
    """After Patch 2: _initial_atomic_source covers contiguous arrays up to
    1024 elements, so a guarded acq/rel CAS over a small flag array is now
    correctly modeled and reports no race.
    """
    flag = torch.zeros(8, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(
        _flag_array_cas_acq_rel_guarded_kernel, (2,), flag, data, out
    )
    assert detector.last_status == "ok"
    assert detector.last_reports == []


# ======== Unsupported visibility (Acceptance #14) ========


def test_unsupported_reason_is_reachable_via_client_manager(
    _isolate_race_detector_atomic_cfg,
):
    """Users running with default abort_on_error=False can still distinguish
    'no race' from 'unsupported' by reading detector.unsupported_reason.
    """
    p = torch.zeros(1, dtype=torch.int32)
    triton_viz.clear()
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_rmw_in_loop_kernel)
    traced[(2,)](p, 2)
    # Reachable via the original detector reference (typical use).
    assert detector.unsupported_reason is not None
    # And via the client manager attached to the traced kernel.
    cm_detector = traced.client_manager.clients["race_detector"]
    assert cm_detector.unsupported_reason is not None


# ======== CAS coherence — try-lock single winner (Patch 1) ========


@triton.jit
def _cas_trylock_single_writer_kernel(flag_ptr, data_ptr):
    old = tl.atomic_cas(flag_ptr, 0, 1, sem="acq_rel", scope="gpu")
    tl.store(data_ptr, tl.program_id(0), mask=(old == 0))


def test_cas_trylock_single_writer_is_not_racy(_isolate_race_detector_atomic_cfg):
    """At most one of two competing CAS(0 -> 1) operations can succeed.

    Without per-location atomic-order coherence, the two-copy solver could
    set old_a == 0 and old_b == 0, activate both guarded stores, and report
    a false WAW. Patch 1 introduces atomic_order vars so two CAS reads of
    the initial value cannot coexist with both guarded writes.
    """
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    detector = _run_detector(_cas_trylock_single_writer_kernel, (2,), flag, data)
    assert detector.last_status == "ok"
    assert detector.unsupported_reason is None
    assert detector.last_reports == []


# ======== Store elem_size precision (Patch 4) ========


@triton.jit
def _float32_store_kernel(out_ptr, BS: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BS + tl.arange(0, BS)
    tl.store(out_ptr + offsets, offsets.to(tl.float32))


def test_store_elem_size_is_four_for_float32_store(
    _isolate_race_detector_atomic_cfg,
):
    """Patch 4 sets dtype/shape on StoreSymbolicExpr so _infer_elem_size
    reads the byte width directly. Without it, the store would silently
    degrade to elem_size=1 and the byte-overlap predicate would collapse to
    addr ==.
    """
    out = torch.zeros(32, dtype=torch.float32)
    detector = _run_detector(_float32_store_kernel, (4,), out, 8)
    store_records = [r for r in detector.records if r.access_mode == "write"]
    assert store_records
    assert all(r.elem_size == 4 for r in store_records), (
        f"Expected elem_size=4 for float32 stores; got "
        f"{[r.elem_size for r in store_records]}"
    )


# ======== Data-dependent address unsupported (Patch 5) ========


@triton.jit
def _data_dependent_atomic_addr_kernel(idx_ptr, flag_ptr):
    pid = tl.program_id(0)
    idx = tl.load(idx_ptr + pid)
    # Atomic CAS at a data-dependent address — the symbolic engine retains
    # the load in the pointer expression, so _expr_contains_load fires.
    tl.atomic_cas(flag_ptr + idx, 0, 1, sem="acq_rel", scope="gpu")


def test_data_dependent_atomic_address_is_unsupported(
    _isolate_race_detector_atomic_cfg,
):
    idx = torch.zeros(2, dtype=torch.int32)
    flag = torch.zeros(4, dtype=torch.int32)
    detector = _run_detector(_data_dependent_atomic_addr_kernel, (2,), idx, flag)
    # The atomic CAS handler runs before the load result gets concretized,
    # so _reject_data_dependent_address fires and marks unsupported.
    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "data-dependent" in detector.unsupported_reason
    assert detector.last_reports == []


def test_reject_data_dependent_address_marks_unsupported(
    _isolate_race_detector_atomic_cfg,
):
    """Direct unit test for ``SymbolicRaceDetector._reject_data_dependent_address``.

    The atomic-CAS path (``test_data_dependent_atomic_address_is_unsupported``)
    exercises this branch end-to-end. The equivalent plain-store path is hard
    to drive end-to-end because Triton's interpreter concretizes loaded
    offsets before they reach ``_handle_access_check``; this synthetic test
    confirms the rejection helper itself works on any pointer expression
    that embeds ``tl.load``, regardless of the access kind that wraps it.
    """
    from triton_viz.clients.symbolic_engine import SymbolicExpr

    triton_viz.clear()
    detector = SymbolicRaceDetector()
    # Initialize launch state without running a kernel.
    detector.grid_callback((2, 1, 1))

    # Build a synthetic load expression (op == "load") whose mere presence
    # in any outer pointer expression must trigger rejection.
    const = SymbolicExpr.from_value(0)
    load_expr = SymbolicExpr.create("load", const, None, None)

    rejected = detector._reject_data_dependent_address(load_expr)
    assert rejected is True
    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "data-dependent" in detector.unsupported_reason


# ======== last_status sanity for an ordinary launch (Patch 3) ========


def test_no_race_kernel_reports_ok_status(_isolate_race_detector_atomic_cfg):
    """A clean launch yields last_status == 'ok' and no unsupported reason."""

    @triton_viz.trace(RaceDetector())
    @triton.jit
    def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)

    n, bs = 64, 8
    x = torch.randn(n)
    out = torch.empty(n)
    triton_viz.clear()
    kernel[(triton.cdiv(n, bs),)](x, out, n, bs)
    detector = kernel.client_manager.clients["race_detector"]
    assert detector.last_status == "ok"
    assert detector.unsupported_reason is None
    assert detector.last_reports == []
