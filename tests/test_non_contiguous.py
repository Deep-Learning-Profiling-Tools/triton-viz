import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer

@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def add_kernel_no_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A Triton kernel that loads and stores values without boundary checks (mask).
    This can lead to out-of-bound access if n_elements exceeds the buffer size.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # No mask is applied here, so loading/storing beyond the valid range can occur.
    x_val = tl.load(x_ptr + offsets)
    y_val = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x_val + y_val)

def test_autotune_add_inrange_non_contiguous():
    """
    This test ensures that the Triton-viz can handle non-contiguous tensors correctly.
    """
    x = torch.randn(64, 2, device='cuda')
    y = torch.randn(64, 2, device='cuda')
    
    # Transpose x and y to make them non-contiguous
    x_nc = x.transpose(0, 1)  # x_nc.shape = (2, 64)
    y_nc = y.transpose(0, 1)  # y_nc.shape = (2, 64)
    
    out_nc = torch.empty_like(x_nc)

    # check if non-contiguous
    assert not x_nc.is_contiguous()
    assert not y_nc.is_contiguous()
    assert not out_nc.is_contiguous()
    
    n_elements = x_nc.numel()  # n_elements = 128
    
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    add_kernel_no_mask[grid](
        x_ptr=x_nc,
        y_ptr=y_nc,
        out_ptr=out_nc,
        n_elements=n_elements,
        BLOCK_SIZE=16
    )

    expected = x_nc + y_nc
    assert torch.allclose(out_nc, x_nc + y_nc)

def test_autotune_add_out_of_bound_non_contiguous():
    """
    This test reproduces out-of-bound access within a non-contiguous tensor.
    """
    x = torch.randn(64, 2, device='cuda')
    y = torch.randn(64, 2, device='cuda')

    x_nc = x.transpose(0, 1)  # x_nc.shape = (2, 64)
    y_nc = y.transpose(0, 1)  # y_nc.shape = (2, 64)

    out_nc = torch.empty_like(x_nc)

    # check if non-contiguous
    assert not x_nc.is_contiguous()
    assert not y_nc.is_contiguous()
    assert not out_nc.is_contiguous()

    # Intentionally using 256 elements while only 128 are actually available
    # This triggers out-of-bound access
    grid = lambda META: (triton.cdiv(256, META['BLOCK_SIZE']),)

    add_kernel_no_mask[grid](
        x_ptr=x_nc,
        y_ptr=y_nc,
        out_ptr=out_nc,
        n_elements=256,
        BLOCK_SIZE=16
    )

    print("test_autotune_add_out_of_bound_non_contiguous() completed: Potential out-of-bound access occurred.")

