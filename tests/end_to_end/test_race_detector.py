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


# ======== RW+WAW — Non-atomic Histogram ========


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
    assert RaceType.RW in race_types


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


# ======== Two-phase Global Barrier ========


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
    assert detector.finalize_phase == "z3_done"
    assert len(races) == 0


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
    assert detector.finalize_phase == "concrete"
    # Desired behavior with phase-aware modeling: no races.
    assert len(races) == 0


# ======== Data-dependent pointer triggers concrete restart ========


def test_data_dependent_ptr_concrete_fallback():
    """Data-dependent pointer must trigger full concrete restart with real side effects."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # Load values, then use them as store indices (data-dependent pointer)
        indices = tl.load(input_ptr + offsets, mask=mask, other=0)
        tl.store(output_ptr + indices, (offsets + 1).to(tl.float32), mask=mask)

    n, bs = 16, 8
    inp = torch.arange(n, dtype=torch.int32)  # identity mapping = no overlap
    out = torch.full((n,), -1.0, dtype=torch.float32)  # sentinel
    kernel[(triton.cdiv(n, bs),)](inp, out, n, bs)

    assert detector.finalize_phase == "concrete"
    # Verify block 0's concrete side effects: out[i] == i + 1 for all i
    expected = torch.arange(1, n + 1, dtype=torch.float32)
    assert torch.equal(
        out, expected
    ), f"Block 0 side effects missing — got {out}, expected {expected}"
    races = launches[-1].records
    assert len(races) == 0  # identity mapping means no overlap


# ======== Masked non-overlap must not false-positive ========


def test_no_false_positive_masked_nonoverlap():
    """Masked-off lanes aliasing must not produce a false positive.

    Both blocks compute address ranges that overlap in their masked-off
    lanes, but their active lanes are disjoint. A solver that doesn't
    bind mask+overlap to the same lane would produce a false positive.
    """
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK_SIZE: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        addrs = pid * HALF + offsets
        mask = offsets < HALF
        tl.store(out_ptr + addrs, offsets.to(tl.float32), mask=mask)

    out = torch.empty(8, dtype=torch.float32)
    kernel[(2,)](out, 8, 4)

    assert detector.finalize_phase == "z3_done"
    races = launches[-1].records
    assert len(races) == 0


# ======== Witness address correctness ========


def test_witness_address_in_overlap_range():
    """The witness address must fall in the actual overlapping region."""

    @triton_viz.trace(RaceDetector())
    @triton.jit
    def kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * (BLOCK_SIZE - 1)  # BUG: overlap by 1
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out_ptr + offsets, offsets.to(tl.float32), mask=mask)

    n, bs = 32, 8
    out = torch.empty(n, dtype=torch.float32)
    kernel[(triton.cdiv(n, bs),)](out, n, bs)

    races = launches[-1].records
    assert len(races) > 0
    base = out.data_ptr()
    elem_size = out.element_size()
    for r in races:
        ba = r.access_a.grid_idx[0]
        bb = r.access_b.grid_idx[0]
        range_a = set(
            base + i * elem_size
            for i in range(ba * (bs - 1), min(ba * (bs - 1) + bs, n))
        )
        range_b = set(
            base + i * elem_size
            for i in range(bb * (bs - 1), min(bb * (bs - 1) + bs, n))
        )
        assert (
            r.address_offset in range_a
        ), f"witness {r.address_offset} not in block {ba} range"
        assert (
            r.address_offset in range_b
        ), f"witness {r.address_offset} not in block {bb} range"


# ======== Tensor pointer fallback records races ========


def test_tensor_pointer_fallback_records_races():
    """Block-pointer ops bail to concrete and overlapping accesses are still detected."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        # Overlap: stride is BLOCK_SIZE-1 instead of BLOCK_SIZE
        block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(N,),
            strides=(1,),
            offsets=(pid * (BLOCK_SIZE - 1),),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
        tl.store(block_ptr, val, boundary_check=(0,))

    n, bs = 32, 8
    out = torch.zeros(n, dtype=torch.float32)
    kernel[(triton.cdiv(n, bs),)](out, N=n, BLOCK_SIZE=bs)

    assert detector.finalize_phase == "concrete"
    races = launches[-1].records
    assert len(races) > 0, (
        "Tensor pointer ops lowered to concrete but no races detected — "
        "concrete before-callbacks may not cover TensorPointerStore"
    )


# ======== Analysis-time bailout triggers grid restart (Patch 1) ========


class _AnalysisBailDetector(RaceDetector):
    """Forces analysis-time bailout to test grid restart."""

    def _check_symbolic_races(self):
        self._need_concrete_fallback = True
        return []

    def finalize(self):
        self.finalize_phase = self._phase
        return super().finalize()


def test_analysis_time_bail_restarts_grid():
    """When _check_symbolic_races triggers bail, block 0 must re-run concretely."""
    detector = _AnalysisBailDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offsets, (pid + 1).to(tl.float32))

    bs = 4
    n_blocks = 2
    out = torch.full((n_blocks * bs,), -1.0, dtype=torch.float32)
    kernel[(n_blocks,)](out, bs)

    assert detector.finalize_phase == "concrete"
    # Both blocks must have written: block 0 writes 1.0, block 1 writes 2.0
    expected = torch.cat([torch.full((bs,), float(pid + 1)) for pid in range(n_blocks)])
    assert torch.equal(
        out, expected
    ), f"Block 0 concrete side effects missing — got {out}, expected {expected}"


# ======== Tensor pointer load fallback (Patch 2) ========


def test_tensor_pointer_load_fallback():
    """Block-pointer load bails to concrete; output matches input."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(N,),
            strides=(1,),
            offsets=(pid * BLOCK_SIZE,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        vals = tl.load(block_ptr, boundary_check=(0,))
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        tl.store(out_ptr + offsets, vals, mask=mask)

    n, bs = 16, 8
    inp = torch.arange(n, dtype=torch.float32)
    out = torch.full((n,), -1.0, dtype=torch.float32)
    kernel[(triton.cdiv(n, bs),)](inp, out, N=n, BLOCK_SIZE=bs)

    assert detector.finalize_phase == "concrete"
    assert torch.equal(out, inp), f"Load path mismatch — got {out}, expected {inp}"


# ======== Two distinct CAS sites remain symbolic (Patch 3) ========


def test_two_distinct_cas_sites_remain_symbolic():
    """Two CAS ops on different sync pointers should stay on the symbolic path."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, sync_a_ptr, sync_b_ptr):
        pid = tl.program_id(0)
        # One CAS on each of two distinct sync addresses
        tl.atomic_cas(sync_a_ptr, 0, 0)
        tl.atomic_cas(sync_b_ptr, 0, 0)
        tl.store(out_ptr + pid, tl.cast(pid, tl.float32))

    n_blocks = 2
    out = torch.zeros(n_blocks, dtype=torch.float32)
    sync_a = torch.zeros(1, dtype=torch.int32)
    sync_b = torch.zeros(1, dtype=torch.int32)
    kernel[(n_blocks,)](out, sync_a, sync_b)

    assert (
        detector.finalize_phase == "z3_done"
    ), f"Expected symbolic path but got phase={detector.finalize_phase}"
    races = launches[-1].records
    assert len(races) == 0


# ======== Masked barrier is NOT trusted (Fix 1) ========


def test_masked_barrier_not_trusted_symbolic():
    """Masked atomic_add means not all blocks participate — barrier invalid."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, sync_ptr, N_BLOCKS: tl.constexpr):
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, tl.cast(pid, tl.float32))
        # Barrier add only for pid == 0 — not all blocks participate
        barrier_mask = pid == 0
        tl.atomic_add(sync_ptr, 1, mask=barrier_mask)
        tl.atomic_cas(sync_ptr, 0, 0)
        dst = (pid + 1) % N_BLOCKS
        tl.store(out_ptr + dst, tl.cast(pid + 100, tl.float32))

    n_blocks = 4
    out = torch.zeros(n_blocks, dtype=torch.float32)
    sync = torch.zeros(1, dtype=torch.int32)
    kernel[(n_blocks,)](out, sync, n_blocks)

    races = launches[-1].records
    assert detector.finalize_phase == "z3_done"
    assert len(races) > 0, "Masked barrier should NOT be trusted — races expected"


# ======== PID-dependent barrier pointer NOT trusted (Fix 1) ========


def test_pid_dependent_barrier_ptr_not_trusted():
    """Each block targets a different sync address — not a global barrier."""
    detector = _ModeTrackingRaceDetector()

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(out_ptr, sync_ptr, N_BLOCKS: tl.constexpr):
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, tl.cast(pid, tl.float32))
        tl.atomic_add(sync_ptr + pid, 1)
        tl.atomic_cas(sync_ptr + pid, 0, 0)
        dst = (pid + 1) % N_BLOCKS
        tl.store(out_ptr + dst, tl.cast(pid + 100, tl.float32))

    n_blocks = 4
    out = torch.zeros(n_blocks, dtype=torch.float32)
    sync = torch.zeros(n_blocks, dtype=torch.int32)
    kernel[(n_blocks,)](out, sync, n_blocks)

    races = launches[-1].records
    assert detector.finalize_phase == "z3_done"
    assert (
        len(races) > 0
    ), "PID-dependent barrier ptr should NOT be trusted — races expected"
