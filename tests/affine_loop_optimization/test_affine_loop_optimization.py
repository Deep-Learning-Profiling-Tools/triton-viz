import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz.core import config as cfg


cfg.sanitizer_backend = "symexec"


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def copy_row_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(N, TILE_N)
    for tile_idx in range(0, num_tiles):
        col_offsets = tile_idx * TILE_N + tl.arange(0, TILE_N)
        mask = col_offsets < N

        x = tl.load(in_ptr + pid * N + col_offsets, mask=mask, other=0.0)
        y = x  # Copy operation
        tl.store(out_ptr + pid * N + col_offsets, y, mask=mask)


def test_copy_kernel():
    torch.manual_seed(0)
    M, N = 32, 1000
    x = torch.randn((M, N), dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (M,)
    copy_row_kernel[grid](
        x,
        y,
        M,
        N,
        TILE_N=128,
    )
