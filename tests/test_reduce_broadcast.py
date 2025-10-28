"""Test for reproducing the broadcast rank mismatch issue with reduce operations."""

import pytest
import torch
import numpy as np
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Sanitizer


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def reduce_sum_kernel(in_out_ptr0, XBLOCK: tl.constexpr):
    """Test kernel that performs a reduction and stores the result."""
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    # Load values
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    
    # Perform reduction (this is where the issue occurs)
    # The result is a scalar but needs to be broadcast for storage
    tmp1 = tl.sum(tmp0, axis=0)
    
    # Store the reduced value back WITH a mask to avoid out-of-bounds
    # This should broadcast tmp1 to match the shape of x0
    tl.store(in_out_ptr0 + x0, tmp1, xmask)


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def reduce_max_kernel(in_ptr, out_ptr, XBLOCK: tl.constexpr):
    """Test kernel that performs max reduction."""
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    tmp0 = tl.load(in_ptr + x0, xmask)
    tmp1 = tl.max(tmp0, axis=0)
    tl.store(out_ptr + x0, tmp1, xmask)


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def reduce_min_kernel(in_ptr, out_ptr, XBLOCK: tl.constexpr):
    """Test kernel that performs min reduction."""
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    
    tmp0 = tl.load(in_ptr + x0, xmask)
    tmp1 = tl.min(tmp0, axis=0)
    tl.store(out_ptr + x0, tmp1, xmask)


def test_reduce_sum_broadcast():
    """Test that reduce sum works with triton-viz sanitizer."""
    size = 32
    input_tensor = torch.randn(size, dtype=torch.float32, device='cpu')
    
    grid = lambda meta: (triton.cdiv(size, meta['XBLOCK']),)
    reduce_sum_kernel[grid](input_tensor, XBLOCK=128)


def test_reduce_max_broadcast():
    """Test that reduce max works with triton-viz sanitizer."""
    size = 32
    input_tensor = torch.randn(size, dtype=torch.float32, device='cpu')
    output_tensor = torch.zeros(size, dtype=torch.float32, device='cpu')
    
    grid = lambda meta: (triton.cdiv(size, meta['XBLOCK']),)
    reduce_max_kernel[grid](input_tensor, output_tensor, XBLOCK=128)


def test_reduce_min_broadcast():
    """Test that reduce min works with triton-viz sanitizer."""
    size = 32
    input_tensor = torch.randn(size, dtype=torch.float32, device='cpu')
    output_tensor = torch.zeros(size, dtype=torch.float32, device='cpu')
    
    grid = lambda meta: (triton.cdiv(size, meta['XBLOCK']),)
    reduce_min_kernel[grid](input_tensor, output_tensor, XBLOCK=128)


if __name__ == "__main__":
    # Run the tests
    try:
        test_reduce_sum_broadcast()
        print("✓ test_reduce_sum_broadcast passed")
    except Exception as e:
        print(f"✗ test_reduce_sum_broadcast failed: {e}")
    
    try:
        test_reduce_max_broadcast()
        print("✓ test_reduce_max_broadcast passed")
    except Exception as e:
        print(f"✗ test_reduce_max_broadcast failed: {e}")
    
    try:
        test_reduce_min_broadcast()
        print("✓ test_reduce_min_broadcast passed")
    except Exception as e:
        print(f"✗ test_reduce_min_broadcast failed: {e}")
