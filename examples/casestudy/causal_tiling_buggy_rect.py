import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer


N_CTX = 64
BLOCK_M = 8
BLOCK_N = 16  # rectangular tile (BLOCK_M < BLOCK_N)


@triton_viz.trace(clients=Tracer())
@triton.jit
def causal_access_buggy_rect(
    scores_ptr,
    out_ptr,
    stride_scores,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = (offs_m < N_CTX)
    col_mask = (offs_n < N_CTX)
    causal_mask = offs_n[None, :] <= offs_m[:, None]
    mask = row_mask[:, None] & col_mask[None, :] & causal_mask

    is_diag = pid_m == pid_n
    mask = tl.where(is_diag, row_mask[:, None] & col_mask[None, :], mask)

    ptr = scores_ptr + offs_m[:, None] * stride_scores + offs_n[None, :]
    tile = tl.load(ptr, mask=mask, other=0.0)
    row_sums = tl.sum(tile, axis=1)
    tl.store(out_ptr + offs_m, row_sums, mask=row_mask)


def run_demo():
    scores = torch.arange(N_CTX * N_CTX, dtype=torch.float32).reshape(N_CTX, N_CTX)
    out = torch.zeros(N_CTX, dtype=torch.float32)

    grid = (triton.cdiv(N_CTX, BLOCK_M), triton.cdiv(N_CTX, BLOCK_N))
    causal_access_buggy_rect[grid](
        scores,
        out,
        scores.stride(0),
        N_CTX=N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    print("buggy rectangular diag sum:", out.sum().item())
    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()

