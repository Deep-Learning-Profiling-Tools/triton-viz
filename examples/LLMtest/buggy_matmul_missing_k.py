import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer

# Optional: visualizer LLM — set one or both before running.
_LL_CONFIG_PATH = ""
_LL_API_KEY = ""


@triton_viz.trace(client=Tracer())
@triton.jit
def matmul_missing_k_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Bug: skip the last K tile on purpose (range ends at K - BLOCK_K).
    for _k in range(0, K - BLOCK_K, BLOCK_K):
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (_k + offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(_k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_demo():
    torch.manual_seed(0)
    M, N, K = 64, 64, 64
    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    c = torch.empty((M, N), dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_missing_k_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    ref = a @ b
    max_diff = (c - ref).abs().max().item()
    print(f"[buggy_matmul_missing_k] max diff: {max_diff:.6f}")

    if _LL_CONFIG_PATH:
        triton_viz.setup_llm(config_path=_LL_CONFIG_PATH)
    if _LL_API_KEY:
        triton_viz.setup_llm(api_key=_LL_API_KEY)
    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()
