import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


def max(inp):
    logging.debug("GEMS MAX")
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch.cuda.device(inp.device):
        max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        max_kernel[grid](inp, out_value, out_index, M, N, K)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out




##################################################################################################################################################


def test_max():
    # 测试1：1维Tensor，验证max函数
    # 使用随机生成的长度为1024的一维Tensor
    inp1d = torch.randn(1024, device="cuda")
    # 使用自定义max函数
    out1d_custom = max(inp1d)

    # 测试2：2维Tensor，验证max_dim函数
    # 使用随机生成的1024x1024的二维Tensor
    inp2d = torch.randn(1024, 1024, device="cuda")
    # 使用自定义max_dim函数，沿着dim=1计算最大值
    out2d_custom = max_dim(inp2d, dim=1)

    # 测试3：3维Tensor，验证max_dim函数
    # 使用随机生成的128x64x32的三维Tensor
    inp3d = torch.randn(128, 64, 32, device="cuda")
    # 使用自定义max_dim函数，沿着dim=2计算最大值
    out3d_custom = max_dim(inp3d, dim=2)

    # 测试4：保持维度的测试
    # 使用随机生成的512x256的二维Tensor
    inp2d_keepdim = torch.randn(512, 256, device="cuda")
    # 使用自定义max_dim函数，保持维度的情况下计算最大值
    out2d_custom_keepdim = max_dim(inp2d_keepdim, dim=1, keepdim=True)

    # 测试5：负维度测试
    # 使用随机生成的64x128x256的三维Tensor
    inp3d_neg_dim = torch.randn(64, 128, 256, device="cuda")
    # 使用自定义max_dim函数，沿着负的维度计算最大值（等价于dim=1）
    out3d_custom_neg_dim = max_dim(inp3d_neg_dim, dim=-2)

    # 记录每个测试用例的结果
    results = {
        "test_case_1": out1d_custom,
        "test_case_2": out2d_custom,
        "test_case_3": out3d_custom,
        "test_case_4": out2d_custom_keepdim,
        "test_case_5": out3d_custom_neg_dim,
    }

    return results

result_gold = test_max()
