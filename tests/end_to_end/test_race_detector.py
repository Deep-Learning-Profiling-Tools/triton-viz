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
    # flag_ptr is the START of an 8-element array (numel != 1) — _initial_atomic_source
    # cannot identify the scalar initial value, so the closed-world model
    # falls back to rf_unknown which does NOT enable synchronizes-with.
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@pytest.mark.xfail(
    strict=True,
    reason="multi-element flag arrays are not modeled by _initial_atomic_source; "
    "the closed-world source model conservatively reports races for guarded "
    "acq/rel CAS over flag arrays. Track in a follow-up issue.",
)
def test_flag_array_cas_acq_rel_guarded_is_not_racy_xfail(
    _isolate_race_detector_atomic_cfg,
):
    flag = torch.zeros(8, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(
        _flag_array_cas_acq_rel_guarded_kernel, (2,), flag, data, out
    )
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
