
import torch
import triton
import triton.language as tl

# global quantize and transpose
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        # ...
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(
    A,
    absmax_inv_ptr,
    B,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)

    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))

    tl.store(B, output, mask=mask)

def quantize_global_transpose(input):
    absmax = input.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    M, N = input.shape
    out = torch.empty(N, M, device="cuda", dtype=torch.int8)

    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _quantize_global_transpose[grid](
        input,
        absmax_inv,
        out,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
    )
    return out, absmax




##################################################################################################################################################


import torch

# Test for quantize_global_transpose
def test_quantize_global_transpose():
    results = {}
    
    # Create a random 2D tensor on CUDA for first test case
    input_tensor_1 = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    # Call the quantize_global_transpose function for the first test case
    output_1, absmax_1 = quantize_global_transpose(input_tensor_1)
    results["test_case_1"] = (output_1, absmax_1)

    # Create a random 2D tensor on CUDA for second test case
    input_tensor_2 = torch.randn(256, 128, device='cuda', dtype=torch.float32)
    # Call the quantize_global_transpose function for the second test case
    output_2, absmax_2 = quantize_global_transpose(input_tensor_2)
    results["test_case_2"] = (output_2, absmax_2)

    # Create a random 2D tensor on CUDA for third test case
    input_tensor_3 = torch.randn(512, 256, device='cuda', dtype=torch.float32)
    # Call the quantize_global_transpose function for the third test case
    output_3, absmax_3 = quantize_global_transpose(input_tensor_3)
    results["test_case_3"] = (output_3, absmax_3)

    # Create a random 2D tensor on CUDA for fourth test case
    input_tensor_4 = torch.randn(256, 512, device='cuda', dtype=torch.float32)
    # Call the quantize_global_transpose function for the fourth test case
    output_4, absmax_4 = quantize_global_transpose(input_tensor_4)
    results["test_case_4"] = (output_4, absmax_4)
    
    return results

result_gold = test_quantize_global_transpose()
