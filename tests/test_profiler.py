import torch

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Profiler


# ======== Case 3: Check masked element percentage for tuning BLOCK_SIZE ========
@triton_viz.trace(clients=(profiler := Profiler(disable_buffer_load_check=True)))
@triton.jit
def mask_percentage_test_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test kernel with a known mix of masked and unmasked load/store operations.

    Expected operations per block:
    - 2 masked loads (with mask parameter)
    - 1 unmasked load (without mask parameter)
    - 2 masked stores (with mask parameter)
    - 1 unmasked store (without mask parameter)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Masked load #1: load with mask
    a = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Masked load #2: another load with mask
    b = tl.load(in_ptr + offsets, mask=mask, other=1.0)

    # Unmasked load: load without mask
    c = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))  # No mask - this is RawLoad

    # Compute result
    result = a + b + c

    # Masked store #1: store with mask
    tl.store(out_ptr + offsets, result, mask=mask)

    # Masked store #2: another store with mask
    tl.store(out_ptr + offsets + N, result * 2, mask=mask)

    # Unmasked store: store without mask
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)  # No mask - this is RawStore


def test_mask_percentage():
    """
    Test that the profiler correctly reports the mask element statistics.

    The profiler now tracks:
    - Total number of mask elements across all operations
    - Number of False elements in all masks
    """
    N = 100
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)  # Should be 4 blocks

    # Create input/output tensors
    x = torch.randn(N, dtype=torch.float32)
    y = torch.empty(N * 2, dtype=torch.float32)

    # Run the kernel
    grid = (num_blocks,)
    mask_percentage_test_kernel[grid](x, y, N, BLOCK_SIZE)

    # Expected mask statistics:
    # Note: In the triton interpreter, load/store operations without an explicit mask
    # are interpreted as having a mask where all elements are True.
    #
    # Block 0: offsets=[0..31], mask=[True]*32 (0 False)
    # Block 1: offsets=[32..63], mask=[True]*32 (0 False)
    # Block 2: offsets=[64..95], mask=[True]*32 (0 False)
    # Block 3: offsets=[96..127], mask=[True]*4 + [False]*28 (28 False)
    #
    # Each block has 3 loads total (2 with explicit masks + 1 without explicit mask):
    # - 2 loads with explicit mask: offsets < N
    # - 1 load without explicit mask: always all-True
    # - Total load mask elements = 3 loads * 32 elements * 4 blocks = 384
    # - False elements in loads = 2 masked loads * 28 False (only in block 3) = 56
    #
    # Similarly for stores (2 with explicit masks + 1 without):
    # - Total store mask elements = 3 stores * 32 elements * 4 blocks = 384
    # - False elements in stores = 2 masked stores * 28 False (only in block 3) = 56

    expected_load_mask_total = 384
    expected_load_mask_false = 56
    expected_store_mask_total = 384
    expected_store_mask_false = 56

    # Verify the statistics from profiler
    assert (
        profiler.load_mask_total_count == expected_load_mask_total
    ), f"Expected {expected_load_mask_total} total load mask elements, got {profiler.load_mask_total_count}"
    assert (
        profiler.load_mask_false_count == expected_load_mask_false
    ), f"Expected {expected_load_mask_false} false load mask elements, got {profiler.load_mask_false_count}"
    assert (
        profiler.store_mask_total_count == expected_store_mask_total
    ), f"Expected {expected_store_mask_total} total store mask elements, got {profiler.store_mask_total_count}"
    assert (
        profiler.store_mask_false_count == expected_store_mask_false
    ), f"Expected {expected_store_mask_false} false store mask elements, got {profiler.store_mask_false_count}"

    # Verify the masked percentage calculation
    expected_load_masked_percentage = (56 / 384) * 100  # ~14.58%
    expected_store_masked_percentage = (56 / 384) * 100  # ~14.58%

    actual_load_masked_percentage = (
        profiler.load_mask_false_count / profiler.load_mask_total_count
    ) * 100
    actual_store_masked_percentage = (
        profiler.store_mask_false_count / profiler.store_mask_total_count
    ) * 100

    assert (
        abs(actual_load_masked_percentage - expected_load_masked_percentage) < 0.01
    ), f"Expected load masked percentage {expected_load_masked_percentage:.2f}%, got {actual_load_masked_percentage:.2f}%"
    assert (
        abs(actual_store_masked_percentage - expected_store_masked_percentage) < 0.01
    ), f"Expected store masked percentage {expected_store_masked_percentage:.2f}%, got {actual_store_masked_percentage:.2f}%"
