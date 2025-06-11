import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz import config as cfg


try:
    torch.cuda.current_device()
except:
    pytest.skip("This test requires a CUDA-enabled environment.", allow_module_level=True)

cfg.sanitizer_backend = "symexec"

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    ],
    key=["n_elements"],
)
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


def test_autotune_add_inrange():
    """
    This test uses n_elements = 128, matching the size of the input tensors.
    It should NOT cause any out-of-bound access.
    """
    x = torch.randn(128)
    y = torch.randn(128)
    out = torch.empty_like(x)

    # The kernel launch uses n_elements=128, aligned with the tensor size.
    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    add_kernel_no_mask[grid](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=128)

    print("test_autotune_add_inrange() passed: No out-of-bound access.")


def test_autotune_add_out_of_bound():
    """
    This test deliberately sets n_elements = 256, exceeding the actual buffer size (128).
    It will likely cause out-of-bound reads/writes, which may trigger errors or warnings.
    """
    x = torch.randn(128)
    y = torch.randn(128)
    out = torch.empty_like(x)

    # The kernel launch uses n_elements=256, exceeding the valid tensor size.
    grid = lambda META: (triton.cdiv(256, META["BLOCK_SIZE"]),)
    add_kernel_no_mask[grid](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=256)

    # Depending on hardware/drivers, this may or may not raise an error immediately.
    print("test_autotune_add_oob() completed: Potential out-of-bound access occurred.")
