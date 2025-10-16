import pytest
import torch
import numpy as np
from io import StringIO
import sys

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Profiler


@triton_viz.trace(clients=(profiler := Profiler(CHECK_LOAD_MASK_PERCENTAGE=True)))
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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Masked load #1: load with mask
    a = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Masked load #2: another load with mask
    b = tl.load(in_ptr + offsets, mask=mask, other=1.0)

    # Unmasked load: load without mask (only when we know it's safe)
    # For this test, we load from a fixed safe range
    safe_offsets = tl.arange(0, BLOCK_SIZE)  # Always within bounds
    c = tl.load(in_ptr + safe_offsets)  # No mask - this is RawLoad

    # Compute result
    result = a + b + c

    # Masked store #1: store with mask
    tl.store(out_ptr + offsets, result, mask=mask)

    # Masked store #2: another store with mask
    tl.store(out_ptr + offsets + N, result * 2, mask=mask)

    # Unmasked store: store without mask
    tl.store(out_ptr + safe_offsets, result)  # No mask - this is RawStore


def test_mask_percentage():
    """
    Test that the profiler correctly reports the percentage of masked vs unmasked operations.
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

    # Expected counts per block (based on actual triton interpreter behavior):
    # - 2 masked loads + 1 unmasked load = 3 loads per block
    # - 2 masked stores + 1 unmasked store = 3 stores per block
    # Note: The triton interpreter may create additional operations internally
    expected_total_loads = 3 * num_blocks
    expected_masked_loads = 2
    expected_total_stores = 3 * num_blocks
    expected_masked_stores = 2

    # Verify the statistics from profiler
    assert profiler.total_loads == expected_total_loads, \
        f"Expected {expected_total_loads} total loads, got {profiler.total_loads}"
    assert profiler.masked_loads == expected_masked_loads, \
        f"Expected {expected_masked_loads} masked loads, got {profiler.masked_loads}"
    assert profiler.total_stores == expected_total_stores, \
        f"Expected {expected_total_stores} total stores, got {profiler.total_stores}"
    assert profiler.masked_stores == expected_masked_stores, \
        f"Expected {expected_masked_stores} masked stores, got {profiler.masked_stores}"
