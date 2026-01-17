import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer


@triton_viz.trace(client=Tracer())
@triton.jit
def flip_1d_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    _ = tl.flip(x, dim=0)
    rev_offs = (n - 1) - offs
    tl.store(y_ptr + rev_offs, x, mask=mask)


@triton_viz.trace(client=Tracer())
@triton.jit
def flip_2d_kernel(
    x_ptr,
    y_ptr,
    H,
    W,
    stride_h,
    stride_w,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mm = offs_m[:, None]
    nn = offs_n[None, :]
    mask = (mm < H) & (nn < W)

    ptrs = x_ptr + mm * stride_h + nn * stride_w
    vals = tl.load(ptrs, mask=mask, other=0)

    _ = tl.flip(vals, dim=(0 if dim == 0 else 1))

    if dim == 0:
        rm = (H - 1) - mm
        rn = nn
    else:
        rm = mm
        rn = (W - 1) - nn

    out_ptrs = y_ptr + rm * stride_h + rn * stride_w
    tl.store(out_ptrs, vals, mask=mask)


def run_1d(n: int = 256):
    x = torch.arange(n, dtype=torch.int32)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 128),)
    flip_1d_kernel[grid](x, y, n, BLOCK=128)
    assert torch.equal(y, torch.flip(x, dims=[0]))


def run_2d(h: int = 16, w: int = 32, dim: int = 1):
    x = torch.arange(h * w, dtype=torch.int32).reshape(h, w)
    y = torch.empty_like(x)
    grid = (triton.cdiv(h, 16), triton.cdiv(w, 16))
    flip_2d_kernel[grid](
        x, y, h, w, x.stride(0), x.stride(1), dim, BLOCK_M=16, BLOCK_N=16
    )
    assert torch.equal(y, torch.flip(x, dims=[dim]))


if __name__ == "__main__":
    run_1d(256)
    run_2d(16, 32, dim=1)
    triton_viz.launch(share=True, port=8003)
