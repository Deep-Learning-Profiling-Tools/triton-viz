import torch

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
