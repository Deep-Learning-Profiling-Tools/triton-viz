import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.core.trace import launches
from triton_viz.clients import RaceDetector, RaceType


class _ModeTrackingRaceDetector(RaceDetector):
    """Expose which path produced launch records for test assertions."""

    def __init__(self):
        super().__init__()
        self.finalize_phase = None

    def finalize(self) -> list:
        self.finalize_phase = self._phase
        return super().finalize()


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


def test_while_loop_atomic_cas_minimal_case():
    """Minimal while-loop CAS kernel should run and report no races."""
    previous_num_sms = triton_viz.config.num_sms
    triton_viz.config.num_sms = 2

    try:
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

        assert detector.finalize_phase == "concrete"
        assert torch.all(out == max_spin)
        races = launches[-1].records
        assert len(races) == 0
    finally:
        triton_viz.config.num_sms = previous_num_sms


# ======== Two-phase Global Barrier ========


def test_two_phase_barrier_symbolic_no_race():
    """Symbolic path: cross-phase overlaps after barrier should not race."""

    previous_num_sms = triton_viz.config.num_sms
    triton_viz.config.num_sms = 1

    try:
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
        assert detector.finalize_phase == "z3_done"
        assert len(races) == 0
    finally:
        triton_viz.config.num_sms = previous_num_sms


def test_two_phase_barrier_concrete_no_race():
    """Two phase stores separated by a global CAS barrier should not race."""

    previous_num_sms = triton_viz.config.num_sms
    triton_viz.config.num_sms = 4

    try:
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
                tl.atomic_cas(sync_ptr, N_BLOCKS, N_BLOCKS) != N_BLOCKS
                and spins < max_spin
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
        assert detector.finalize_phase == "concrete"
        # Desired behavior with phase-aware modeling: no races.
        assert len(races) == 0
    finally:
        triton_viz.config.num_sms = previous_num_sms
