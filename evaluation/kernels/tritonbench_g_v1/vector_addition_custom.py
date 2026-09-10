
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, a + b, mask=offs < size)

def custom_add(a, b):
    """custom add one."""
    c = torch.empty_like(a)
    size = c.size(0)
    BLOCK = 16

    grid = (triton.cdiv(size, BLOCK), )
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c




##################################################################################################################################################


import torch

def test_add():
    # 测试用例 1：简单的两个向量加法
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.float32, device='cuda')
    b = torch.tensor([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    # 测试用例 2：不同值的加法
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32, device='cuda')
    b = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    # 测试用例 3：更大向量的加法
    a = torch.arange(32, dtype=torch.float32, device='cuda')
    b = torch.arange(32, 0, -1, dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    # 测试用例 4：空向量的边界情况
    a = torch.tensor([], dtype=torch.float32, device='cuda')
    b = torch.tensor([], dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    test_results = {
        "test_case_1": custom_add(a, b),
        "test_case_2": custom_add(a, b),
        "test_case_3": custom_add(a, b),
        "test_case_4": custom_add(a, b),
    }
    return test_results

result_gold = test_add()
