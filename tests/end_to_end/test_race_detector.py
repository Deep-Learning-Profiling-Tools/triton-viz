import inspect

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector, RaceType, Sanitizer
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from triton_viz.clients.race_detector.hb_solver import RaceReport
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.clients.symbolic_engine import SymbolicExpr
from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client
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


# ======== Relaunch — Same Traced Kernel Twice ========


def test_second_launch_of_same_kernel_detects_again():
    """The trace decorator holds one client instance across launches, so the
    second launch must recapture and re-solve. Regression test: finalize's
    _clear_launch_runtime used to null addr_sym, which SymbolicClient's
    grid_callback dereferences but never recreates — the second launch of any
    traced kernel crashed with an AssertionError before this was fixed.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * (BLOCK_SIZE - 1)  # BUG: should be BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, offsets.to(tl.float32), mask=mask)

    n, bs = 32, 8
    out = torch.empty(n, dtype=torch.float32)
    for _ in range(2):
        kernel[(triton.cdiv(n, bs),)](out, n, bs)
        assert detector.last_status == "ok"
        assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


def test_second_launch_of_loop_kernel_does_not_crash():
    """Loop kernels exercise _loop_hook_after, which also asserts on the
    launch-runtime symbolic state; relaunching must not trip it either.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        for _i in range(2):
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))

    out = torch.zeros(16, dtype=torch.float32)
    for _ in range(2):
        kernel[(2,)](out, 8)
        assert detector.last_status == "ok"


# ======== Loops — Cross-Iteration Cross-Block Races ========


def test_loop_cross_iteration_race_detected():
    """Block 0 at iteration 1 and block 1 at iteration 0 both write out[1].
    Regression test: the flushed loop's iterator var was omitted from
    copy_local_vars (the loop is popped off loop_stack before
    _process_pending_check runs), so the two-copy solver never alpha-renamed
    it — both program copies were pinned to the same iteration and every
    cross-iteration cross-block race came back unsat.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        for i in range(2):
            tl.store(out_ptr + pid + i, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


def test_loop_disjoint_blocks_no_race():
    """Per-copy iterator renaming must keep the iterator's range constraint:
    addr = pid*2 + i with i in [0, 2) gives disjoint blocks for ANY pair of
    iterations, so an unbounded renamed iterator would be a false positive.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        for i in range(2):
            tl.store(out_ptr + pid * 2 + i, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_nested_loop_cross_iteration_race_and_no_race():
    """Nested loops: the inner flush must carry BOTH iterator vars (outer from
    the still-active loop_stack, inner from the flushed LoopContext)."""

    racy = SymbolicRaceDetector()

    @triton_viz.trace(racy)
    @triton.jit
    def racy_kernel(out_ptr):
        pid = tl.program_id(0)
        for i in range(2):
            for j in range(2):
                tl.store(out_ptr + pid + i * 2 + j, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    racy_kernel[(2,)](out)
    assert racy.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in racy.last_reports)

    clean = SymbolicRaceDetector()

    @triton_viz.trace(clean)
    @triton.jit
    def clean_kernel(out_ptr):
        pid = tl.program_id(0)
        for i in range(2):
            for j in range(2):
                tl.store(out_ptr + pid * 4 + i * 2 + j, 1.0)

    clean_kernel[(2,)](out)
    assert clean.last_status == "ok"
    assert clean.last_reports == []


def test_post_loop_leftover_iterator_no_false_positive():
    """A store after the loop that reuses the leftover Python loop variable
    must be modeled with the iterator's concrete final value (identical in
    every block), not a symbolic var. Regression test: per-copy renaming is
    launch-wide per var, so without concretization the post-loop record's
    iterator was renamed but carried no range premise — an unbounded var
    that produced cross-tensor false positives on this race-free kernel.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(a_ptr, out_ptr):
        pid = tl.program_id(0)
        for i in range(2):
            tl.store(a_ptr + pid * 2 + i, 1.0)
        tl.store(out_ptr + pid + i, 1.0)

    a = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](a, out)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_post_loop_leftover_iterator_true_race_detected():
    """Leftover-iterator concretization must not hide real races: with
    i == 1 after the loop, every block stores to out[1] — a WAW race."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        acc = 0.0
        for i in range(2):
            acc += 1.0
        tl.store(out_ptr + i, acc + pid)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


def test_sibling_loop_leftover_iterator_no_false_positive():
    """A second loop whose body reuses the first loop's leftover iterator:
    the finished iterator concretizes to its final value while the active
    loop's own iterator stays symbolic and per-copy renamed."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        for i in range(2):
            pass
        for j in range(2):
            tl.store(out_ptr + pid * 4 + i * 2 + j, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_inner_loop_final_value_varying_with_outer_is_unsupported():
    """An inner loop whose trip count depends on the still-active outer
    iterator has no single final value for its leftover variable (it is
    1 - outer here), and the deferred store dedupes across outer
    iterations — no constant substitution is correct. The launch must be
    marked unsupported rather than reporting a phantom race (this kernel
    is race-free: block0 writes out[1], block1 writes out[2])."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        acc = 0.0
        for outer in range(2):
            for i in range(2 - outer):
                acc += 1.0
            tl.store(out_ptr + pid + outer + i, acc)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "finished loop iterator" in detector.unsupported_reason
    assert detector.last_reports == []


def test_zero_iteration_reentry_keeps_leftover_iterator_value():
    """A zero-trip re-activation of an inner loop leaves the leftover
    Python variable at the previous activation's final value; the detector
    must restore that substitution instead of leaving the iterator var
    unbounded. Race-free: a gets {0} / {2}, out gets {0} / {1}."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(a_ptr, out_ptr):
        pid = tl.program_id(0)
        for outer in range(2):
            for i in range(1 - outer):
                tl.store(a_ptr + pid * 2 + i, 1.0)
        tl.store(out_ptr + pid + i, 1.0)

    a = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](a, out)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_tl_range_load_store_cross_iteration_race():
    """tl.range loop with a load+store body: the WAW on out_ptr is detected
    across iterations and the read-read overlap on x_ptr stays race-free."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(x_ptr, out_ptr):
        pid = tl.program_id(0)
        for i in tl.range(0, 2):
            v = tl.load(x_ptr + pid + i)
            tl.store(out_ptr + pid + i, v)

    x = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](x, out)

    assert detector.last_status == "ok"
    assert detector.last_reports
    assert {r.race_type for r in detector.last_reports} == {RaceType.WAW}


# ======== Loop Lifecycle — nested re-entry ========


def test_nested_loop_records_do_not_scale_with_outer_trip_count():
    """Each outer iteration re-enters the inner loop with a fresh
    LoopContext (empty signature_cache), so the inner flush used to append
    one structurally identical record per outer iteration — K duplicate
    records, K*(K+1)/2 duplicate reports for one source-level race, and a
    quadratic solver blowup. The launch-level flush dedup must collapse
    them to one."""

    clean = SymbolicRaceDetector()

    @triton_viz.trace(clean)
    @triton.jit
    def clean_kernel(out_ptr, K: tl.constexpr):
        pid = tl.program_id(0)
        for i in range(K):
            for j in range(2):
                tl.store(out_ptr + pid * 2 * K + i * 2 + j, 1.0)

    out = torch.zeros(32, dtype=torch.float32)
    clean_kernel[(2,)](out, 8)

    assert clean.last_status == "ok"
    assert clean.last_reports == []
    assert len(clean.records) == 1

    racy = SymbolicRaceDetector()

    @triton_viz.trace(racy)
    @triton.jit
    def racy_kernel(out_ptr, K: tl.constexpr):
        pid = tl.program_id(0)
        for i in range(K):
            for j in range(2):
                tl.store(out_ptr + pid + i * 2 + j, 1.0)

    racy_kernel[(2,)](out, 8)

    assert racy.last_status == "ok"
    assert len(racy.records) == 1
    # One source-level race must yield one report, not K*(K+1)/2 duplicates.
    assert len(racy.last_reports) == 1
    assert racy.last_reports[0].race_type == RaceType.WAW


# ======== Host-Side Control Flow on Per-Instance Values ========


def test_pid_dependent_branch_is_unsupported_not_false_positive():
    """`if pid == 0:` is resolved by the interpreter with the capture
    block's concrete pid; the branch condition is not modeled, so the
    guarded store would be recorded unconditionally for every PID and this
    race-free kernel (only block 0 writes) reported a phantom WAW. The
    launch must be marked unsupported instead."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        if pid == 0:
            offs = tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out, 8)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "varies per program instance" in detector.unsupported_reason
    assert detector.last_reports == []


def test_pid_dependent_branch_is_unsupported_not_false_negative():
    """Blocks 1 and 2 both write out[0] — a real WAW — but the capture
    block (0) does not take the branch, so nothing was recorded and the
    launch finished "ok" with zero reports. Unsupported is the only honest
    verdict without path-condition modeling."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        if pid > 0:
            tl.store(out_ptr, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(3,)](out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "varies per program instance" in detector.unsupported_reason
    assert detector.last_reports == []


def test_pid_dependent_loop_bound_is_unsupported():
    """A loop bound containing pid is concretized to the capture block's
    trip count (need_full_grid is the sanitizer's compensation; one-shot
    capture has none), silently truncating every other block's iteration
    space."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        for i in tl.range(0, pid + 1):
            tl.store(out_ptr + pid + i, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "concretized" in detector.unsupported_reason
    assert detector.last_reports == []


# ======== 2D Tiles — Independent Arange Instances ========


def test_2d_tile_independent_arange_instances_race_detected():
    """Row and column index vectors are independent tl.arange instances with
    equal (start, end). Regression test: ARANGE_DICT used to intern the
    summary var by (start, end) only, pinning row == col — the modeled
    footprint collapsed to the tile diagonal and this real cross-block
    overlap (block0 writes {0..3}, block1 writes {1..4}) came back unsat.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        r = tl.arange(0, 2)
        c = tl.arange(0, 2)
        offs = r[:, None] * 2 + c[None, :]
        tl.store(out_ptr + pid + offs, 1.0)

    out = torch.zeros(16, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


def test_2d_tile_disjoint_blocks_no_race():
    """Independent per-arange vars must stay range-bounded per copy: with
    addrs = pid*4 + 2r + c the full rectangular footprints of the two blocks
    are disjoint, so any unbounded or conflated index var would show up as a
    false positive here."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        r = tl.arange(0, 2)
        c = tl.arange(0, 2)
        offs = r[:, None] * 2 + c[None, :]
        tl.store(out_ptr + pid * 4 + offs, 1.0)

    out = torch.zeros(16, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


# ======== RAW+WAW — Non-atomic Histogram ========


def test_raw_waw_histogram():
    """Non-atomic load-modify-store on ``bins_ptr + bin_indices`` — the
    target address depends on a loaded value. Since the address-position
    lifting the detector DECIDES this launch: bin_indices lower to selects
    over the read-only input snapshot (per-launch + contents-snapshot
    scope), and with more inputs than bins two instances must collide —
    RAW/WAW reports. (The pre-lifting behavior was an honest abstention;
    the ancient first-block concretisation this test once guarded against
    stays gone — the snapshot semantics is per-instance-exact, not a
    first-block template.)
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
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

    assert detector.last_status == "ok"
    assert detector.last_premises == ("contents-snapshot",)
    assert any(
        r.race_type in (RaceType.RAW, RaceType.WAW) for r in detector.last_reports
    )


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


# ======== Intra-Instance Duplicate Lanes / Single-Block Launches ========


def test_single_block_duplicate_lane_store_reports_race():
    """grid=(1,) makes the solver's different_blocks constraint UNSAT, so
    every cross-copy query is vacuous. Regression test: lanes 0/2 (and 1/3)
    of one store write the same byte with no defined lane order — a real
    WAW that used to come back as a silent 'ok' with zero reports."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs % 2, offs)

    out = torch.zeros(2, dtype=torch.int32)
    kernel[(1,)](out, 4)

    assert detector.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)
    # The witness is a single program instance, not a cross-block pair.
    for r in detector.last_reports:
        assert r.witness_grid_a == r.witness_grid_b


def test_single_block_disjoint_store_reports_ok():
    """The intra-instance query must not turn lane-injective single-block
    stores into false positives."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, offs)

    out = torch.zeros(4, dtype=torch.int32)
    kernel[(1,)](out, 4)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_duplicate_lane_store_detected_in_multi_block_launch():
    """Duplicate lanes inside each program instance with block-disjoint
    footprints: invisible to the cross-copy (different_blocks) queries even
    on a multi-block grid."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + pid * BLOCK + offs % 2, offs)

    out = torch.zeros(8, dtype=torch.int32)
    kernel[(2,)](out, 4)

    assert detector.last_status == "ok"
    assert any(
        r.witness_grid_a == r.witness_grid_b for r in detector.last_reports
    ), "expected an intra-instance duplicate-lane report"


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
    """cta-scoped CAS establishes no cross-CTA synchronization, so the
    data store/load pair races (RAW) — and the two CAS operations
    themselves, being block-local atomics from different CTAs at the same
    address, are not mutually atomic either (WAW)."""

    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)

    detector = _run_detector(_cas_cta_guarded_kernel, (2,), flag, data, out)

    assert len(detector.last_reports) == 2
    _assert_launch_reports(detector)
    _assert_atomic_records(detector, sem="acq_rel", scope="cta")
    data_reports = [
        r
        for r in detector.last_reports
        if not (r.first.record.is_atomic or r.second.record.is_atomic)
    ]
    cas_reports = [
        r
        for r in detector.last_reports
        if r.first.record.is_atomic and r.second.record.is_atomic
    ]
    assert len(data_reports) == 1 and len(cas_reports) == 1
    _assert_report_lines(
        data_reports[0],
        _cas_cta_guarded_kernel,
        (
            "tl.store(data_ptr, 1, mask=is_prod)",
            "x = tl.load(data_ptr, mask=cons_mask, other=0)",
        ),
    )
    _assert_report_lines(
        cas_reports[0],
        _cas_cta_guarded_kernel,
        (
            'old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="cta")',
            'old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="cta")',
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


# ======== AtomicRMW return-value modeling (spec part B) ========
# Integer RMW returns are value-modeled since part B: the downstream use
# below is analyzed (winner-gated read, disjoint per-pid stores → clean)
# instead of aborting to unsupported. The float-typed return keeps the
# sentinel boundary — covered in test_race_detector_rmw.py.


@triton.jit
def _rmw_return_used_downstream_kernel(p_ptr, q_ptr, out_ptr):
    pid = tl.program_id(0)
    old = tl.atomic_add(p_ptr, 1)
    val = tl.load(q_ptr, mask=old == 0, other=0)
    tl.store(out_ptr + pid, val)


def test_atomic_rmw_return_used_downstream_is_modeled(
    _isolate_race_detector_atomic_cfg,
):
    p = torch.zeros(1, dtype=torch.int32)
    q = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(_rmw_return_used_downstream_kernel, (2,), p, q, out)
    assert detector.last_status == "ok"
    assert detector.unsupported_reason is None
    assert detector.last_reports == []
    rmw_records = [r for r in detector.records if r.atomic_kind == "rmw"]
    assert len(rmw_records) == 1
    assert rmw_records[0].rmw_op == "add"
    assert rmw_records[0].old_value is not None


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
    # Atomic CAS at a plain-loaded address — lifted to a snapshot select
    # (address-position spec §1, the co-admitted atomic surface).
    tl.atomic_cas(flag_ptr + idx, 0, 1, sem="acq_rel", scope="gpu")


def test_data_dependent_atomic_address_is_decided(
    _isolate_race_detector_atomic_cfg,
):
    """The co-admitted atomic surface of the address-position lifting:
    a CAS whose address is a plain-loaded index is DECIDED — the all-zero
    index sends both pids to flag[0], where gpu-scope CASes are mutually
    atomic (inclusive scopes, same width, same address) — clean, with the
    contents-snapshot premise recorded."""
    idx = torch.zeros(2, dtype=torch.int32)
    flag = torch.zeros(4, dtype=torch.int32)
    detector = _run_detector(_data_dependent_atomic_addr_kernel, (2,), idx, flag)
    assert detector.last_status == "ok"
    assert detector.last_reports == []
    assert detector.last_premises == ("contents-snapshot",)


# ======== tl.static_range — concrete unrolling ========


def test_static_range_unrolls_concretely_and_detects_race():
    """tl.static_range is compile-time unrolled, so each iteration records
    with a concrete index (no symbolic iterator). Regression test: wrapping
    it symbolically broke host-side consumers of the index (tuple
    indexing's __index__) and mismodeled the unrolled semantics."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        for i in tl.static_range(2):
            tl.store(out_ptr + pid + i, 1.0)

    out = torch.zeros(8, dtype=torch.float32)
    kernel[(2,)](out)

    assert detector.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


# ======== tl.clamp — symbolic ternary support ========


def test_clamp_on_symbolic_values_is_supported():
    """tl.clamp lowers through the ternary-op overrider as
    minimum(maximum(x, lo), hi). Regression test: np.clip used to raise
    NotImplementedError('Unsupported ternary operation'), crashing any
    kernel that clamps a symbolic value."""

    clean = SymbolicRaceDetector()

    @triton_viz.trace(clean)
    @triton.jit
    def clean_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, tl.clamp(x, 0.0, 1.0))

    x = torch.ones(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    clean_kernel[(2,)](x, out, 4)
    assert clean.last_status == "ok"
    assert clean.last_reports == []

    racy = SymbolicRaceDetector()

    @triton_viz.trace(racy)
    @triton.jit
    def racy_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * (BLOCK - 1) + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, tl.clamp(x, 0.0, 1.0))

    racy_kernel[(2,)](x, out, 4)
    assert racy.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in racy.last_reports)


# ======== Atomic scope — cta atomics are not cross-CTA atomic ========


@triton.jit
def _cta_scoped_atomic_add_kernel(out_ptr):
    tl.atomic_add(out_ptr, 1, scope="cta")


def test_cta_scoped_atomics_from_different_blocks_race(
    _isolate_race_detector_atomic_cfg,
):
    """PTX .cta scope guarantees atomicity within one CTA only; two
    cta-scoped atomics from different blocks at the same address are not
    mutually atomic. Regression test: atomic-vs-atomic pairs used to be
    unconditionally race-free regardless of scope."""

    out = torch.zeros(1, dtype=torch.int32)
    detector = _run_detector(_cta_scoped_atomic_add_kernel, (2,), out)

    assert detector.last_status == "ok"
    assert len(detector.last_reports) == 1


@triton.jit
def _gpu_scoped_atomic_add_kernel(out_ptr):
    tl.atomic_add(out_ptr, 1, scope="gpu")


def test_gpu_scoped_atomics_at_same_address_no_race(
    _isolate_race_detector_atomic_cfg,
):
    """Device-scope same-width atomics at the same address stay mutually
    atomic and race-free."""

    out = torch.zeros(1, dtype=torch.int32)
    detector = _run_detector(_gpu_scoped_atomic_add_kernel, (2,), out)

    assert detector.last_status == "ok"
    assert detector.last_reports == []


# ======== CAS read-from with unmodeled (non-CAS) writers ========


@triton.jit
def _rmw_published_guard_kernel(flag_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    tl.atomic_xchg(flag_ptr, 1, mask=is_prod)
    old = tl.atomic_cas(flag_ptr, 1, 1, sem="relaxed", scope="gpu")
    offs = tl.arange(0, BLOCK)
    m = (old == 1) & (offs < 1)
    tl.store(out_ptr + offs, 1.0, mask=m)


def test_cas_guard_published_by_rmw_reports_race(_isolate_race_detector_atomic_cfg):
    """flag starts at 0 and is published to 1 via atomic_xchg — a writer the
    closed-world rf model does not include. Both blocks can then observe 1
    through the relaxed CAS and store out[0]: a real WAW. Regression test:
    the reader's old value used to be hard-pinned to {initial} + {CAS-written
    values} = {0}, making the guard infeasible and silencing the race with
    last_status ok. An overlapping non-CAS writer must open the rf_unknown
    escape (without fabricating synchronizes-with)."""

    flag = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(4, dtype=torch.float32)
    detector = _run_detector(_rmw_published_guard_kernel, (2,), flag, out, 2)

    assert detector.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


@triton.jit
def _rmw_other_tensor_guard_kernel(flag_ptr, other_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    tl.atomic_xchg(other_ptr, 1, mask=is_prod)
    old = tl.atomic_cas(flag_ptr, 1, 1, sem="relaxed", scope="gpu")
    offs = tl.arange(0, BLOCK)
    m = (old == 1) & (offs < 1)
    tl.store(out_ptr + offs, 1.0, mask=m)


def test_cas_closed_world_holds_without_overlapping_writer(
    _isolate_race_detector_atomic_cfg,
):
    """The xchg targets a different tensor, so the flag is only ever written
    by modeled CAS: the closed world must hold and the old == 1 guard stays
    infeasible (initial 0; CAS(cmp=1, new=1) can only republish 0) — the
    rf_unknown escape must not weaken no-race verdicts when no overlapping
    unmodeled writer exists."""

    flag = torch.zeros(1, dtype=torch.int32)
    other = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(4, dtype=torch.float32)
    detector = _run_detector(_rmw_other_tensor_guard_kernel, (2,), flag, other, out, 2)

    assert detector.last_status == "ok"
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

    const = SymbolicExpr.from_value(0)

    # Plain tl.load in a pointer is LIFTED now (address-position spec):
    # the gate must NOT fire on it.
    load_expr = SymbolicExpr.create("load", const, None, None)
    assert detector._reject_data_dependent_address(load_expr) is False

    # Snapshot-UNSTABLE ops keep rejecting (kernel-computed permutations
    # of runtime data have no snapshot semantics).
    cumsum_expr = SymbolicExpr.create("cumsum", load_expr, 0, False, None)
    rejected = detector._reject_data_dependent_address(cumsum_expr)
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


# ======== Counterexample: load-dependent mask must not be generalized ========


def test_load_dependent_mask_must_not_be_generalized_from_first_block(
    _isolate_race_detector_atomic_cfg,
):
    """False negative: loaded input value in a store mask is not modeled per PID.

    Ground truth for ``flag = [1, 0, 0]`` over ``grid = (3,)``:
      pid=0: v = flag[0] = 1 -> mask = False, inactive
      pid=1: v = flag[1] = 0 -> mask = True,  writes out[0]
      pid=2: v = flag[2] = 0 -> mask = True,  writes out[0]

    pid=1 and pid=2 are two distinct program instances, both write the same
    scalar address ``out_ptr`` without atomics or synchronisation — a
    standard WAW data race.

    A sound input-dependent analyzer must produce a witness like:
        pid_a = 1, pid_b = 2
        flag[pid_a] = 0, flag[pid_b] = 0
        addr_a = addr_b = out.data_ptr()

    The current branch sees ``tl.load(flag_ptr + pid)`` through the symbolic
    capture path, which (a) returns a pointer expression rather than the
    loaded value, and (b) triggers ``replace_subtree("load")`` concretisation
    of the mask using the first sampled block's value. Combined with the
    one-shot ``pre_run_callback`` lifecycle, this generalises ``mask = False``
    (the value seen by pid=0) to every symbolic PID, so the detector reports
    zero races. This test asserts the correct semantics; it should fail on
    the current branch and pass after a fix that models loaded values per
    PID (e.g. via a finite-map / Z3 ``Select`` over the input tensor).
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(flag_ptr, out_ptr):
        pid = tl.program_id(0)
        v = tl.load(flag_ptr + pid)
        tl.store(out_ptr, pid, mask=(v == 0))

    # First captured block (pid=0) sees flag[0] == 1; later blocks are the
    # racing writers. The asymmetry is what exposes the over-generalisation.
    flag = torch.tensor([1, 0, 0], dtype=torch.int32)
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel[(3,)](flag, out)

    # The launch should NOT be flagged unsupported — the kernel is plain
    # load + masked store with no atomics-in-loop or RMW-return downstream.
    assert detector.last_status == "ok", (
        f"unexpected status {detector.last_status!r}; "
        f"reason={detector.unsupported_reason!r}"
    )
    assert detector.unsupported_reason is None

    # Sound analyzer behavior: at least one race report, and at least one
    # of them must be a WAW (pid=1 vs pid=2 both writing out[0]) with the
    # witness address equal to the scalar ``out`` pointer.
    assert len(detector.last_reports) >= 1, (
        "expected at least one race report between pid=1 and pid=2 "
        "writing the same scalar out[0]"
    )
    assert any(
        r.race_type == RaceType.WAW
        and {r.witness_grid_a[0], r.witness_grid_b[0]} == {1, 2}
        and r.witness_addr == out.data_ptr()
        for r in detector.last_reports
    ), (
        "expected a WAW race with pids {1, 2} writing out[0]; "
        f"got reports: {[(r.race_type, r.witness_grid_a, r.witness_grid_b, r.witness_addr) for r in detector.last_reports]}"
    )


# ======== Load-value semantics regression coverage ========


def test_masked_load_other_value_drives_store_mask(
    _isolate_race_detector_atomic_cfg,
):
    """Exercises masked tl.load with explicit ``other``.

    For ``flag = [99, 99, 99]`` over ``grid = (3,)``:
      pid=0: load active, v = flag[0] = 99 -> store mask = (99 == 42) -> inactive
      pid=1: load masked out, v = other = 42 -> store mask = (42 == 42) -> active
      pid=2: load masked out, v = other = 42 -> store mask = (42 == 42) -> active

    Soundly detects a WAW between pid=1 and pid=2 writing out[0]. Tests that
    the load-value provider builds ``If(mask, Select(arr, addr), other)`` and
    that the domain constraint is conditional (``Implies(mask, ...)``) so
    masked-out lanes do not over-constrain the model.

    Uses ``other=42`` rather than ``other=0`` because Triton's semantic
    layer drops zero-valued ``other`` via a ``bool(constexpr)`` check
    before the override is invoked — we need a value that passes that
    truthy check so ``other`` actually reaches the provider.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(flag_ptr, out_ptr):
        pid = tl.program_id(0)
        v = tl.load(flag_ptr + pid, mask=(pid == 0), other=42)
        tl.store(out_ptr, pid, mask=(v == 42))

    flag = torch.tensor([99, 99, 99], dtype=torch.int32)
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel[(3,)](flag, out)

    assert detector.last_status == "ok", (
        f"unexpected status {detector.last_status!r}; "
        f"reason={detector.unsupported_reason!r}"
    )
    assert any(
        r.race_type == RaceType.WAW
        and {r.witness_grid_a[0], r.witness_grid_b[0]} == {1, 2}
        and r.witness_addr == out.data_ptr()
        for r in detector.last_reports
    ), "expected a WAW race with pids {1, 2} writing out[0] under masked-load"


def test_float_load_source_is_unsupported(_isolate_race_detector_atomic_cfg):
    """Float input tensor used as a load value source is unsupported in v1.

    The provider rejects non-integer dtypes since the Z3 model is integer-
    only — silently truncating would mask real value-dependent behaviour.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(flag_ptr, out_ptr):
        pid = tl.program_id(0)
        v = tl.load(flag_ptr + pid)
        tl.store(out_ptr, pid, mask=(v == 0.0))

    flag = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel[(3,)](flag, out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "dtype" in detector.unsupported_reason
    assert detector.last_reports == []


def test_non_contiguous_load_source_is_unsupported(
    _isolate_race_detector_atomic_cfg,
):
    """Non-contiguous input tensor is unsupported in v1 — directly exercises
    the provider's snapshot helper.

    End-to-end tracing rejects most non-contiguous arg tensors earlier in
    ``arg_callback``, so this unit-style test drives the snapshot helper
    with a hand-rolled view to confirm the provider itself raises
    :class:`UnsupportedSymbolicRaceQuery` with a ``contiguous`` reason.
    """
    from triton_viz.clients.race_detector.hb_common import (
        UnsupportedSymbolicRaceQuery,
    )

    detector = SymbolicRaceDetector()
    base = torch.tensor([[1, 9], [0, 9], [0, 9]], dtype=torch.int32)
    view = base[:, 0]
    assert not view.is_contiguous()

    with pytest.raises(UnsupportedSymbolicRaceQuery, match="contiguous"):
        detector._snapshot_array_for_tensor(view)


def test_oversized_load_source_is_unsupported(_isolate_race_detector_atomic_cfg):
    """Source tensor exceeding ``_MAX_LOAD_SOURCE_ELEMENTS`` is unsupported.

    The Z3 array snapshot unrolls element-by-element via ``Store``; capping
    keeps the per-launch snapshot from blowing up.
    """

    detector = SymbolicRaceDetector()
    cap = detector._MAX_LOAD_SOURCE_ELEMENTS

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(flag_ptr, out_ptr):
        pid = tl.program_id(0)
        v = tl.load(flag_ptr + pid)
        tl.store(out_ptr, pid, mask=(v == 0))

    flag = torch.zeros(cap + 1, dtype=torch.int32)
    flag[0] = 1  # force at least one PID to deactivate to expose the path
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel[(3,)](flag, out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert (
        "size cap" in detector.unsupported_reason
        or "exceeds" in detector.unsupported_reason
    )
    assert detector.last_reports == []


def test_self_write_then_load_is_unsupported(_isolate_race_detector_atomic_cfg):
    """Write-then-load order: kernel writes ``buf`` before reading from it.

    The write-side region tracking registers the buffer; the subsequent
    load-value provider sees the overlap and raises unsupported.
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(buf_ptr, out_ptr):
        pid = tl.program_id(0)
        tl.store(buf_ptr + pid, 0)
        v = tl.load(buf_ptr + pid)
        tl.store(out_ptr, pid, mask=(v == 0))

    buf = torch.zeros(3, dtype=torch.int32)
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel[(3,)](buf, out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert "written by this kernel" in detector.unsupported_reason
    assert detector.last_reports == []


def test_load_then_self_write_is_unsupported(_isolate_race_detector_atomic_cfg):
    """Load-then-write order: kernel reads ``buf`` as a load value source,
    then writes back to the same region.

    The load-value provider registers ``buf`` as a load source first; the
    subsequent store's write-side check sees the overlap and raises
    unsupported (or the post-capture sweep catches it).
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(buf_ptr, out_ptr):
        pid = tl.program_id(0)
        v = tl.load(buf_ptr + pid)
        tl.store(out_ptr, pid, mask=(v == 0))
        tl.store(buf_ptr + pid, 0)

    buf = torch.tensor([1, 0, 0], dtype=torch.int32)
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel[(3,)](buf, out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert (
        "written by this kernel" in detector.unsupported_reason
        or "previously read" in detector.unsupported_reason
        or "overlaps" in detector.unsupported_reason
    )
    assert detector.last_reports == []


# ======== External reproducer: ROCm/aiter#3091 ========


def test_aiter_3091_redundant_histogram_writes(
    _isolate_race_detector_atomic_cfg,
):
    """Minimal reproducer for ROCm/aiter#3091.

    Bug pattern: ``_combined_routing_fused`` calls ``_sum_bitmatrix_rows_fused``
    on every program instance with no ``tl.program_id``-based partitioning of
    the output buffer (``ExpertHist``). Every pid concurrently stores the
    same values to the same global addresses — a non-atomic WAW race even
    when the written values agree. Subsequent ``tl.load(ExpertHist + pid)``
    reads then race against those in-flight writes (RAW).

    This reproducer collapses both phases into a single kernel:
      - Phase 1: every pid writes the full histogram unconditionally
        (mimics ``_sum_bitmatrix_rows_fused``).
      - Phase 2: each pid reads back its own slot (mimics the
        ``n_tokens = tl.load(ExpertHist + pid)`` check in the caller).

    Expected: detector reports both a WAW (between phase-1 writes of two
    distinct pids) and a RAW (between a phase-1 write and a phase-2 load
    from a different pid).
    """

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(hist_ptr, out_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, N)
        # Phase 1: redundant cross-PID write to the full histogram.
        tl.store(hist_ptr + offs, offs)
        # Phase 2: read this pid's slot; phase-1 writes from other pids
        # touch the same address.
        n = tl.load(hist_ptr + pid)
        tl.store(out_ptr + pid, n)

    N = 8
    hist = torch.zeros(N, dtype=torch.int32)
    out = torch.zeros(N, dtype=torch.int32)
    kernel[(N,)](hist, out, N)

    assert detector.last_status == "ok", (
        f"unexpected status {detector.last_status!r}; "
        f"reason={detector.unsupported_reason!r}"
    )

    race_types = {r.race_type for r in detector.last_reports}
    assert RaceType.WAW in race_types, (
        f"expected a WAW race on the phase-1 histogram write across pids; "
        f"got {race_types}"
    )
    assert RaceType.RAW in race_types, (
        f"expected a RAW race between phase-1 store and phase-2 load on "
        f"hist_ptr + pid; got {race_types}"
    )

    hist_base = hist.data_ptr()
    hist_end = hist_base + N * hist.element_size()
    for r in detector.last_reports:
        assert r.witness_grid_a[0] != r.witness_grid_b[0], (
            "race witnesses must be two distinct program instances; "
            f"got a={r.witness_grid_a}, b={r.witness_grid_b}"
        )
        assert hist_base <= r.witness_addr < hist_end, (
            "race witness address must fall inside the histogram tensor; "
            f"got {r.witness_addr} not in [{hist_base}, {hist_end})"
        )


# ======== Launch lifecycle — capture slot, finalize cleanup, status ========


class _BlockCountingClient(Client):
    """Minimal co-attached client that records which blocks executed."""

    NAME = "block_counter"

    def __init__(self):
        super().__init__()
        self.blocks_run: list[tuple[int, ...] | None] = []
        self._last_grid_idx: tuple[int, ...] | None = None

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        # post_run only fires for blocks that actually executed the kernel.
        self.blocks_run.append(self._last_grid_idx)
        return False

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        pass

    def grid_idx_callback(self, grid_idx):
        self._last_grid_idx = grid_idx

    def register_op_callback(self, op_type, *args, **kwargs):
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return []


def test_co_attached_client_sees_full_grid_when_engine_needs_it():
    """A data-dependent loop bound flips the engine's need_full_grid, so
    every block must keep running for a co-attached client. Regression
    test: the race detector's pre_run returned False once its one-shot
    capture finished, and ClientManager's all() aggregation starved the
    co-attached client of blocks 1..3."""

    counter = _BlockCountingClient()
    detector = SymbolicRaceDetector()

    # Detector registered LAST: patch_op rebuilds each op from the original,
    # so the last client's overriders win (the counter registers none).
    @triton_viz.trace(detector)
    @triton_viz.trace(counter)
    @triton.jit
    def kernel(n_ptr, out_ptr):
        pid = tl.program_id(0)
        n = tl.load(n_ptr)
        for i in range(n):
            tl.store(out_ptr + pid * 4 + i, 1.0)

    n = torch.tensor([2], dtype=torch.int32)
    out = torch.zeros(16, dtype=torch.float32)
    kernel[(4,)](n, out)

    assert sorted(counter.blocks_run) == [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
    ], f"co-attached client starved: only saw {counter.blocks_run}"
    # One-shot capture cannot model the load-dependent bound: never 'ok'.
    assert detector.last_status == "unsupported"


def test_concurrent_workers_capture_exactly_once():
    """Under TRITON_VIZ_NUM_SMS >= 2 several worker threads pass pre_run
    before any post_run. Regression test: the capture gate was only set in
    post_run, so multiple workers captured concurrently into shared
    per-launch state — duplicated records, interleaved program_seq, and
    K*(K+1)/2 duplicate reports for one source-level race."""

    saved_num_sms = cfg.num_sms
    cfg.num_sms = 4
    try:
        detector = SymbolicRaceDetector()

        @triton_viz.trace(detector)
        @triton.jit
        def kernel(out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * (BLOCK - 1) + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(out_ptr + offs, offs.to(tl.float32), mask=mask)

        out = torch.empty(64, dtype=torch.float32)
        kernel[(8,)](out, 64, 8)

        assert detector.last_status == "ok"
        # One-shot capture: the kernel's single store -> exactly one record
        # and one WAW report, regardless of how many workers ran blocks.
        assert len(detector.records) == 1
        assert len(detector.last_reports) == 1
        assert detector.last_reports[0].race_type == RaceType.WAW

        # Loop bookkeeping is part of the shared capture state too: a
        # sibling worker pushing/popping contexts corrupted it pre-fix.
        det_loop = SymbolicRaceDetector()

        @triton_viz.trace(det_loop)
        @triton.jit
        def loop_kernel(out_ptr):
            pid = tl.program_id(0)
            for i in range(2):
                for j in range(2):
                    tl.store(out_ptr + pid + 2 * i + j, 1.0)

        out2 = torch.zeros(16, dtype=torch.float32)
        loop_kernel[(8,)](out2)
        assert det_loop.last_status == "ok"
        assert len(det_loop.records) == 1
    finally:
        cfg.num_sms = saved_num_sms


def test_mid_capture_exception_reports_aborted_not_ok():
    """An exception escaping the kernel mid-capture must never be readable
    as a clean verdict, even when the harness has no finalize-on-error
    routing: grid_callback arms last_status = 'aborted' at launch start and
    only finalize() upgrades it. Regression test: the status was pre-set to
    'ok' at launch start, so a mid-kernel crash left 'ok' with empty
    reports — indistinguishable from a clean no-race verdict."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, 1.0)
        x = 1 // 0  # noqa: F841 — host-side crash mid-capture

    out = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ZeroDivisionError):
        kernel[(2,)](out)

    assert detector.last_status == "aborted"
    assert detector.last_reports == []


def test_pre_grid_exception_does_not_read_previous_ok():
    """A launch that dies BEFORE grid_callback (user grid lambda, arg
    conversion) must not leave the previous launch's clean verdict readable
    as this launch's: arg_callback re-arms the pessimistic 'aborted' first.
    Also covers the never-launched detector, which reads 'aborted' too."""

    detector = SymbolicRaceDetector()
    assert detector.last_status == "aborted"

    @triton_viz.trace(detector)
    @triton.jit
    def racy_kernel(out_ptr):
        tl.store(out_ptr, 1.0)

    out = torch.zeros(4, dtype=torch.float32)
    racy_kernel[(2,)](out)
    assert detector.last_status == "ok"
    assert len(detector.last_reports) >= 1

    with pytest.raises(ZeroDivisionError):
        racy_kernel[lambda META: (1 // 0,)](out)

    assert detector.last_status == "aborted"
    assert detector.last_reports == []


def test_mid_capture_exception_does_not_leak_observer_into_next_client():
    """A launch that dies before finalize() cannot uninstall its class-level
    scalar-concretize observer; the next launch's SymbolicClient
    grid_callback must reclaim the slot. Regression test: the stale detector
    observer fired inside a later Sanitizer launch and crashed it with
    UnsupportedSymbolicRaceQuery on a pid-dependent branch the sanitizer
    handles fine on its own."""

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def crashing_kernel(out_ptr):
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, 1.0)
        x = 1 // 0  # noqa: F841 — host-side crash mid-capture

    out = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ZeroDivisionError):
        crashing_kernel[(2,)](out)

    # finalize() never ran, so the dead launch's observer may still occupy
    # the class-level slot here; the next launch must self-heal regardless.
    sanitizer = Sanitizer(abort_on_error=True)

    @triton_viz.trace(sanitizer)
    @triton.jit
    def branching_kernel(out_ptr):
        pid = tl.program_id(0)
        if pid == 0:
            tl.store(out_ptr, 1.0)

    out2 = torch.zeros(4, dtype=torch.float32)
    branching_kernel[(2,)](out2)

    assert SymbolicExpr._scalar_concretize_observer is None
    assert SymbolicExpr._scalar_concretize_observer_owner is None

    # The detector itself must also recover on its next launch.
    @triton_viz.trace(detector)
    @triton.jit
    def racy_kernel(out_ptr):
        tl.store(out_ptr, 1.0)

    out3 = torch.zeros(4, dtype=torch.float32)
    racy_kernel[(2,)](out3)
    assert detector.last_status == "ok"
    assert len(detector.last_reports) >= 1


def test_abort_on_error_status_survives_raise():
    """abort_on_error raises out of the launch; the detector must still
    record the unsupported verdict. Regression test: _raise_or_mark raised
    without setting the unsupported state, so the exception-path finalize
    reported a clean 'ok'."""

    detector = SymbolicRaceDetector(abort_on_error=True)

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        if pid == 0:
            tl.store(out_ptr, 1.0)

    out = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(UnsupportedSymbolicRaceQuery):
        kernel[(2,)](out)

    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason is not None
    assert detector.last_reports == []


def test_solver_crash_clears_hooks_and_never_reports_ok(monkeypatch):
    """An exception escaping finalize's solver phase must still release the
    launch runtime — the class-level hooks would otherwise leak into later
    launches of other clients — and must not leave the launch-start 'ok'
    behind."""
    import triton_viz.clients.race_detector.race_detector as rd_module

    class _BoomSolver:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated solver failure")

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def racy_kernel(out_ptr):
        tl.store(out_ptr, 1.0)

    out = torch.zeros(4, dtype=torch.float32)
    monkeypatch.setattr(rd_module, "TwoCopySymbolicHBSolver", _BoomSolver)
    with pytest.raises(RuntimeError, match="simulated solver failure"):
        racy_kernel[(2,)](out)

    assert detector.last_status == "aborted"
    assert detector.last_reports == []
    assert SymbolicExpr._load_value_provider is None
    assert SymbolicExpr._scalar_concretize_observer is None

    # A later sanitizer launch must be clean — pre-fix, the leaked
    # load-value provider hijacked the sanitizer's masked-load eval and
    # crashed it with UnsupportedSymbolicRaceQuery.
    sanitizer = Sanitizer(abort_on_error=True)

    @triton_viz.trace(sanitizer)
    @triton.jit
    def sanitizer_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)  # masked load WITHOUT other
        tl.store(out_ptr + offs, v, mask=mask)

    x = torch.ones(32)
    out2 = torch.zeros(32)
    sanitizer_kernel[(4,)](x, out2, 32, 8)


def test_z3_unknown_reports_unsupported_not_ok(monkeypatch):
    """A race query Z3 cannot decide (timeout, nonlinear give-up) must not
    collapse into a clean 'ok' verdict. Regression test: find_races only
    appended candidates on '== sat', so an unknown silently became
    last_status == 'ok' with zero reports."""
    import triton_viz.clients.race_detector.two_copy_symbolic_hb_solver as tc_module
    from z3 import unknown as z3_unknown

    class _UnknownSolver:
        def __init__(self, *args, **kwargs):
            pass

        def add(self, *args, **kwargs):
            pass

        def push(self):
            pass

        def pop(self):
            pass

        def check(self):
            return z3_unknown

        def reason_unknown(self):
            return "stubbed unknown"

        def model(self):
            raise AssertionError("model() must not be read on unknown")

    detector = SymbolicRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr):
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, 1.0)

    out = torch.zeros(4, dtype=torch.float32)
    monkeypatch.setattr(tc_module, "Solver", _UnknownSolver)
    kernel[(2,)](out)

    assert detector.last_status == "unsupported"
    assert detector.last_reports == []
    assert "could not decide" in (detector.unsupported_reason or "")


# ======== Block pointers — tile footprint modeling ========


@triton.jit
def _block_ptr_store_kernel(out_ptr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(64,),
        strides=(1,),
        offsets=(pid * STRIDE,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(bp, tl.full((BLOCK,), 1.0, tl.float32), boundary_check=(0,))


def test_block_ptr_overlapping_tiles_report_race():
    """tl.make_block_ptr accesses must lower the tile footprint (the access
    expr, not its descriptor ptr). Regression test: _handle_access_check
    evaluated expr.ptr, whose make_block_ptr lowering raises
    NotImplementedError — every block-pointer kernel crashed."""
    out = torch.zeros(64, dtype=torch.float32)
    # Tiles [0, 16) and [8, 24) overlap.
    detector = _run_detector(_block_ptr_store_kernel, (2,), out, 8, 16)
    assert detector.last_status == "ok", detector.unsupported_reason
    assert any(r.race_type == RaceType.WAW for r in detector.last_reports)


def test_block_ptr_disjoint_tiles_report_ok():
    """Disjoint tiles must stay clean: the tile index vars are copy-local,
    so the solver quantifies each program copy's tile element independently
    without fabricating overlap."""
    out = torch.zeros(64, dtype=torch.float32)
    detector = _run_detector(_block_ptr_store_kernel, (2,), out, 16, 16)
    assert detector.last_status == "ok", detector.unsupported_reason
    assert detector.last_reports == []


@triton.jit
def _block_ptr_advance_loop_kernel(out_ptr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(64,),
        strides=(1,),
        offsets=(pid * STRIDE,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    for _i in range(2):
        tl.store(bp, tl.full((BLOCK,), 1.0, tl.float32), boundary_check=(0,))
        bp = tl.advance(bp, (BLOCK,))


def test_block_ptr_advance_loop_race_and_no_race():
    """tl.advance chains flushed through the loop-deferred path must keep
    their tile index vars copy-local (PendingEvent carries them to
    _process_pending_check)."""
    out = torch.zeros(64, dtype=torch.float32)
    # pid 0 covers [0, 16), pid 1 covers [8, 24): overlap -> race.
    racy = _run_detector(_block_ptr_advance_loop_kernel, (2,), out, 8, 8)
    assert racy.last_status == "ok", racy.unsupported_reason
    assert any(r.race_type == RaceType.WAW for r in racy.last_reports)

    # pid 0 covers [0, 16), pid 1 covers [16, 32): disjoint -> clean.
    clean = _run_detector(_block_ptr_advance_loop_kernel, (2,), out, 16, 8)
    assert clean.last_status == "ok", clean.unsupported_reason
    assert clean.last_reports == []


# ======== Value-dependent addresses — cumsum / sort ========


@triton.jit
def _cumsum_ptr_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.cumsum(tl.full((BLOCK,), 1, tl.int32), 0)
    tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))


def test_cumsum_derived_pointer_is_unsupported_not_crash():
    """Regression test: the address guard only checked has_op("load"), so a
    cumsum-derived pointer reached eval and CumsumSymbolicExpr's
    NotImplementedError crashed the launch instead of yielding an
    unsupported verdict."""
    out = torch.zeros(64, dtype=torch.float32)
    detector = _run_detector(_cumsum_ptr_kernel, (2,), out, 8)
    assert detector.last_status == "unsupported"
    assert "cumsum" in (detector.unsupported_reason or "")
    assert detector.last_reports == []


@triton.jit
def _sort_ptr_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    sorted_offs = tl.sort(BLOCK - 1 - offs)
    tl.store(out_ptr + pid * BLOCK + sorted_offs, offs.to(tl.float32))


def test_sort_derived_pointer_is_unsupported_not_silent_ok():
    """Regression test: SortSymbolicExpr lowers as the identity of its
    input, so a sort-derived pointer recorded the unsorted lanes — a wrong
    per-lane footprint under a clean 'ok' verdict."""
    out = torch.zeros(64, dtype=torch.float32)
    detector = _run_detector(_sort_ptr_kernel, (2,), out, 8)
    assert detector.last_status == "unsupported"
    assert "sort" in (detector.unsupported_reason or "")
    assert detector.last_reports == []


# ======== Acquire-on-failure: rf-val is reader-success-independent ========


@triton.jit
def _cas_acquire_on_failure_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    # The consumer's cmp=7 can never match (flag stays in {0,1}): its CAS
    # always FAILS — but a failed acquire-CAS still READS the location,
    # and reading the released value is what establishes the sw edge.
    cmp = tl.where(is_prod, 0, 7)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def _cas_acquire_on_failure_relaxed_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1

    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 7)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="relaxed", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


def test_cas_acquire_on_failure_synchronizes(_isolate_race_detector_atomic_cfg):
    """A consumer CAS that always fails still acquires through its READ:
    rf-val (writer.written == reader.old) is independent of the reader's
    own success, so the guarded load is ordered — no race."""
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(_cas_acquire_on_failure_kernel, (2,), flag, data, out)
    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_cas_acquire_on_failure_relaxed_races(_isolate_race_detector_atomic_cfg):
    """Mutation twin: the same failed-CAS gate at relaxed carries the
    value but no ordering — the guarded load must race."""
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    detector = _run_detector(
        _cas_acquire_on_failure_relaxed_kernel, (2,), flag, data, out
    )
    assert len(detector.last_reports) == 1
    _assert_report_lines(
        detector.last_reports[0],
        _cas_acquire_on_failure_relaxed_kernel,
        (
            "tl.store(data_ptr, 1, mask=is_prod)",
            "x = tl.load(data_ptr, mask=cons_mask, other=0)",
        ),
    )
