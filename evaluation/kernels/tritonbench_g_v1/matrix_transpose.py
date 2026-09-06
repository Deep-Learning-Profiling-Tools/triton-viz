import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M,
    Out,
    matrix_stridex,
    matrix_stridey,
    out_stridex,
    out_stridey,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    # transpose
    matrix_ptr = M + d_head_arange[None, :] * matrix_stridey + size_m_arange[:, None] * matrix_stridex
    out_ptr = Out + d_head_arange[None, :] * out_stridex + size_m_arange[:, None] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, matrix)

def wrapper(size_m, d_head):
    matrix = torch.randn((size_m, d_head), dtype=torch.float16, device="cuda")
    out = torch.zeros((d_head, size_m), dtype=torch.float16, device="cuda")

    grid = (1,)
    kernel[grid](
        matrix,
        out,
        *matrix.stride(),
        *out.stride(),
        size_m,
        d_head,
    )
    return out



##################################################################################################################################################


import torch

def test_triton_vs_torch():
    results = {}

    # 测试用例 1: 基本矩阵转置 (小矩阵)
    size_m, d_head = 16, 16
    out = wrapper(size_m, d_head)
    results["test_case_1"] = out.clone()

    # 测试用例 2: 非方形矩阵
    size_m, d_head = 32, 64
    out = wrapper(size_m, d_head)
    results["test_case_2"] = out.clone()

    return results


# 运行测试
result_gold = test_triton_vs_torch()
print(result_gold)