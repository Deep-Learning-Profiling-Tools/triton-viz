import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer


@triton_viz.trace(client=Tracer())
@triton.jit
def add_shift_bug_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Bug: y uses shifted index (+1), which misaligns all pairs.
    y = tl.load(y_ptr + offsets + 1, mask=(offsets + 1) < n_elements, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def run_demo():
    torch.manual_seed(0)
    n = 256
    x = torch.randn((n,), dtype=torch.float32)
    y = torch.randn((n,), dtype=torch.float32)
    out = torch.empty_like(x)

    block = 64
    grid = (triton.cdiv(n, block),)
    add_shift_bug_kernel[grid](x, y, out, n, BLOCK_SIZE=block)

    ref = x + y
    max_diff = (out - ref).abs().max().item()
    print(f"[buggy_vector_add_shift] max diff: {max_diff:.6f}")

    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()
