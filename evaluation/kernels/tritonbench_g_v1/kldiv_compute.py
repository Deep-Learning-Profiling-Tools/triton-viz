import torch
import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel(x_ptr,  # *Pointer* to first input vector.
                        y_ptr,  # *Pointer* to second input vector.
                        output_ptr,  # *Pointer* to output vector.
                        n_elements,  # Size of the vector.
                        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                        # NOTE: `constexpr` so it can be used as a shape value.
                        ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)

    tl.store(output_ptr + offsets, output, mask=mask)


def kldivergence(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    kldivergence_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output




##################################################################################################################################################


import torch

def test_kldivergence():
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # 使用 Triton 计算 KL 散度
    output_triton = kldivergence(x, y)

    # 分支覆盖率【1/4】

    # 补全所有分支调用
    results = {}
    
    # Test case 1
    x1 = torch.rand(1024, device='cuda')
    y1 = torch.rand(1024, device='cuda')
    results['test_case_1'] = kldivergence(x1, y1)

    # Test case 2
    x2 = torch.rand(2048, device='cuda')
    y2 = torch.rand(2048, device='cuda')
    results['test_case_2'] = kldivergence(x2, y2)

    # Test case 3
    x3 = torch.rand(4096, device='cuda')
    y3 = torch.rand(4096, device='cuda')
    results['test_case_3'] = kldivergence(x3, y3)

    # Test case 4
    x4 = torch.rand(8192, device='cuda')
    y4 = torch.rand(8192, device='cuda')
    results['test_case_4'] = kldivergence(x4, y4)

    return results

result_gold = test_kldivergence()
