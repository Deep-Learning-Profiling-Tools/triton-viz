import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz import config as cfg


cfg.sanitizer_backend = "symexec"


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    indices = tl.load(idx_ptr + offsets)
    out_val = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offsets, out_val)


def test_indirect_load_inrange():
    idx = torch.arange(128, dtype=torch.int32)
    src = torch.rand(128)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    indirect_load_kernel[grid](idx, src, dst, BLOCK_SIZE=32)


# def test_indirect_load_out_of_bound():
#     """
#     This test deliberately sets n_elements = 256, exceeding the actual buffer size (128).
#     It will likely cause out-of-bound reads/writes, which may trigger errors or warnings.
#     """
#     x = torch.arange(128, device="cuda", dtype=torch.int32)
#     out = torch.empty_like(x)

#     # The kernel launch uses n_elements=256, which exceeds the size of x.
#     grid = lambda META: (triton.cdiv(256, META["BLOCK_SIZE"]),)
#     indirect_load_kernel[grid](x_ptr=x, out_ptr=out, n_elements=256)

#     print("test_indirect_load_out_of_bound() passed: Out-of-bound access detected.")
