import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.data import Dot
from triton_viz.core.trace import launches


@triton_viz.trace(clients=Tracer())
@triton.jit
def matmul_acc_kernel(
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
    reset_acc: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if reset_acc:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    else:
        acc = tl.load(c_ptrs, mask=c_mask, other=0.0)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    tl.store(c_ptrs, acc, mask=c_mask)


def run_demo():
    torch.manual_seed(0)
    M = N = K = 32
    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)

    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))

    ref = a @ b
    print("=== Baseline (torch.matmul) ===")
    print("reference max:", ref.abs().max().item())

    # Warmup: 正常写入 C，但不保留 trace
    c_state = torch.zeros_like(ref)
    matmul_acc_kernel[grid](
        a,
        b,
        c_state,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_state.stride(0),
        c_state.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
        reset_acc=True,
    )
    print("after warmup diff:", (c_state - ref).abs().max().item())
    launches.clear()

    # Bug run: 忘记 reset，继续在已有 C 上累加（只记录这一轮）
    matmul_acc_kernel[grid](
        a,
        b,
        c_state,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_state.stride(0),
        c_state.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
        reset_acc=True,
    )
    diff = (c_state - ref).abs().max().item()
    print(f"[bug] max diff after forgetting reset: {diff:.4e}")

    if launches:
        last = launches[-1]
        for rec in getattr(last, "records", []):
            if isinstance(rec, Dot):
                rec.acc_reset = False

    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()
