import torch
import triton
import triton.language as tl

# Example import of the Trace decorator with a sanitizer client
# Adjust according to your actual project structure
import triton_viz
from triton_viz.clients import Sanitizer


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def add_kernel_no_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A Triton kernel that loads and stores values without boundary checks (mask).
    This can lead to out-of-bound access if n_elements exceeds the buffer size.
    """
    pid = tl.program_id(0) # pid = [x, y]
    block_start = pid * BLOCK_SIZE # block_start = [x * BLOCK_SIZE, y * BLOCK_SIZE]
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # block_start = [x * BLOCK_SIZE, y * BLOCK_SIZE + BLOCK_SIZE]

    # No mask is applied here, so loading/storing beyond the valid range can occur.
    x_val = tl.load(x_ptr + offsets) # x_addr = [x_ptr + x * BLOCK_SIZE, x_ptr + y * BLOCK_SIZE + BLOCK_SIZE]
    y_val = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x_val + y_val)

def test_autotune_add_inrange():
    """
    This test uses n_elements = 128, matching the size of the input tensors.
    It should NOT cause any out-of-bound access.
    """
    x = torch.randn(128, device='cuda')
    y = torch.randn(128, device='cuda')
    out = torch.empty_like(x)

    # The kernel launch uses n_elements=128, aligned with the tensor size.
    grid = lambda META: (triton.cdiv(128, META['BLOCK_SIZE']),)
    add_kernel_no_mask[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=128,
        BLOCK_SIZE=16
    )

    print("test_autotune_add_inrange() passed: No out-of-bound access.")