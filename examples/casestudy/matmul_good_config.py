import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 8
NUM_WARPS = 8
NUM_STAGES = 2


@triton_viz.trace(clients=Tracer())
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_pid_n

    group_id = pid // group_size
    pid_in_group = pid % group_size
    pid_m = group_id * GROUP_M + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k
        mask_a = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        mask_b = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)


def run_demo():
    case_name = "Matmul Good Config"
    M = N = K = 512
    torch.manual_seed(0)
    device = torch.device("cpu")
    a = torch.randn((M, K), dtype=torch.float32, device=device)
    b = torch.randn((K, N), dtype=torch.float32, device=device)
    c = torch.empty((M, N), dtype=torch.float32, device=device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    matmul_kernel[grid](
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
        GROUP_M=GROUP_M,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    ref = a @ b
    diff = (c - ref).abs()
    print(f"[{case_name}] max diff: {diff.max().item():.3e}")
    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()

