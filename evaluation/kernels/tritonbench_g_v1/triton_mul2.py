import triton
from triton import language as tl
import torch


@triton.jit
def mul2_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = 2 * x
    tl.store(out_ptr + offsets, output, mask=mask)

@triton.jit
def mul2_inplace_kernel(
    ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask)
    output = 2 * x
    tl.store(ptr + offsets, output, mask=mask)


def triton_mul2(x, BLOCK_SIZE=16):
    output = torch.zeros_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output

def triton_mul2_inplace(x, BLOCK_SIZE=16):
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_inplace_kernel[grid](x, n_elements, BLOCK_SIZE)
    return x




##################################################################################################################################################


def test_mul():
    N = 1024 * 1024
    x = torch.randn(N, device='cuda')

    # 分支1: triton_mul2 with BLOCK_SIZE=1024
    triton_mul2_result = triton_mul2(x, BLOCK_SIZE=1024)

    # 分支2: triton_mul2_inplace with BLOCK_SIZE=1024
    triton_mul2_inplace_result = triton_mul2_inplace(x.clone(), BLOCK_SIZE=1024)

    # 分支3: triton_mul2 with a different BLOCK_SIZE
    triton_mul2_result_case2 = triton_mul2(x, BLOCK_SIZE=512)

    # 分支4: triton_mul2_inplace with a different BLOCK_SIZE
    triton_mul2_inplace_result_case2 = triton_mul2_inplace(x.clone(), BLOCK_SIZE=512)

    # 返回测试结果
    result_dict = {
        "test_case_1": triton_mul2_result,
        "test_case_2": triton_mul2_inplace_result,
        "test_case_3": triton_mul2_result_case2,
        "test_case_4": triton_mul2_inplace_result_case2,
    }
    
    return result_dict

# 执行测试函数
result_gold = test_mul()
