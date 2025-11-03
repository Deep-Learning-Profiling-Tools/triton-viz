import torch
import pytest

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Profiler


# ======== Case 2: Check if for loop can be unrolled ========
@triton_viz.trace(clients=(profiler := Profiler(disable_buffer_load_check=True)))
@triton.jit
def for_loop_test_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test kernel with for-loops to verify loop statistics tracking.

    This kernel contains 2 for-loops:
    - First loop: iterates 10 times
    - Second loop: iterates 5 times
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # First for-loop: 10 iterations
    result = x
    for i in range(10):
        result = result + 1.0

    # Second for-loop: 5 iterations
    for j in range(5):
        result = result * 2.0

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def test_for_loop_statistics():
    """
    Test that the profiler correctly tracks for-loop statistics.
    """
    N = 100
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    # Create input/output tensors
    x = torch.randn(N, dtype=torch.float32)
    y = torch.empty(N, dtype=torch.float32)

    # Run the kernel
    grid = (num_blocks,)
    for_loop_test_kernel[grid](x, y, N, BLOCK_SIZE)

    # Expected loop statistics:
    # The kernel has 2 for-loops, but they are executed once per grid
    # Each loop should be recorded only once (when first encountered)
    # Loop 1: range(10) -> 10 steps
    # Loop 2: range(5) -> 5 steps

    expected_num_loops = 2
    expected_loop_steps = [10, 5]

    # Verify the loop statistics from profiler
    assert (
        len(profiler.loop_info) == expected_num_loops
    ), f"Expected {expected_num_loops} loops, got {len(profiler.loop_info)}"

    for idx, (lineno, total_steps) in enumerate(profiler.loop_info):
        expected_steps = expected_loop_steps[idx]
        assert (
            total_steps == expected_steps
        ), f"Loop #{idx+1}: Expected {expected_steps} steps, got {total_steps}"
        assert isinstance(
            lineno, int
        ), f"Loop #{idx+1}: lineno should be int, got {type(lineno)}"
        assert lineno > 0, f"Loop #{idx+1}: lineno should be positive, got {lineno}"


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


# ======== Block Sampling ========
@triton.jit
def block_sampling_test_kernel(
    x_ptr,
    y_ptr,
    counter_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test kernel that increments a counter for each block executed.
    This allows us to verify that block sampling is working correctly.
    """
    pid = tl.program_id(0)

    # Atomically increment counter to count executions
    tl.atomic_add(counter_ptr, 1)

    # Normal kernel logic
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(y_ptr + offsets, y, mask=mask)


@pytest.mark.parametrize(
    "test_name,enable_sampling,k_value,expected_executions",
    [
        ("No block sampling", False, None, 8),
        ("Block sampling k=1", True, 1, 1),
        ("Block sampling k=3", True, 3, 3),
        ("Block sampling k=5", True, 5, 5),
        ("Block sampling k=8 (all blocks)", True, 8, 8),
        ("Block sampling k=20 (exceeds total)", True, 20, 8),
    ],
    ids=lambda x: x if isinstance(x, str) else str(x),
)
def test_block_sampling(test_name, enable_sampling, k_value, expected_executions):
    """
    Test that the block sampling feature correctly samples k blocks from the grid.

    This test verifies:
    - Block sampling can be enabled/disabled
    - The k parameter correctly limits the number of blocks executed
    - When k > total blocks, all blocks are executed
    - The sampling works with different k values
    """
    # Test configuration
    n_elements = 1024
    BLOCK_SIZE = 128
    grid = (n_elements // BLOCK_SIZE,)  # Creates 8 blocks

    # Create test data
    x = torch.randn(n_elements, dtype=torch.float32)
    y = torch.zeros_like(x)
    counter = torch.zeros(1, dtype=torch.int32)

    # Create profiler with block sampling configuration
    profiler = Profiler(
        block_sampling=enable_sampling,
        k=k_value,
        disable_buffer_load_check=True,
        disable_load_mask_percentage_check=True,
    )

    # Apply trace decorator and run kernel
    traced_kernel = triton_viz.trace(profiler)(block_sampling_test_kernel)
    traced_kernel[grid](x, y, counter, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Get actual execution count
    actual_executions = counter.item()

    # Verify the count matches expectation
    assert (
        actual_executions == expected_executions
    ), f"{test_name}: Expected {expected_executions} executions, got {actual_executions}"

    # Verify profiler settings
    if enable_sampling:
        assert (
            profiler.block_sampling is True
        ), f"{test_name}: block_sampling should be True"
        assert (
            profiler.k == k_value
        ), f"{test_name}: k should be {k_value}, got {profiler.k}"
