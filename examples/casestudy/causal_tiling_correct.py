import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer


N_CTX = 64
SUB_TILE = 4
BLOCK_M = SUB_TILE * 4  # 16
BLOCK_N = SUB_TILE * 4  # 16


@triton_viz.trace(clients=Tracer())
@triton.jit
def causal_access_correct(
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

    m_mask = offs_m < N_CTX
    n_mask = offs_n < N_CTX
    causal_mask = offs_n[None, :] <= offs_m[:, None]
    mask = m_mask[:, None] & n_mask[None, :] & causal_mask

    ptr = scores_ptr + offs_m[:, None] * stride_scores + offs_n[None, :]
    tile = tl.load(ptr, mask=mask, other=0.0)

    row_sums = tl.sum(tile, axis=1)
    tl.store(out_ptr + offs_m, row_sums, mask=m_mask)


def run_demo():
    scores = torch.arange(N_CTX * N_CTX, dtype=torch.float32).reshape(N_CTX, N_CTX)
    out = torch.zeros(N_CTX, dtype=torch.float32)

    grid = (triton.cdiv(N_CTX, BLOCK_M), triton.cdiv(N_CTX, BLOCK_N))
    causal_access_correct[grid](
        scores,
        out,
        scores.stride(0),
        N_CTX=N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    print("triangle sum:", out.sum().item())
    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()
