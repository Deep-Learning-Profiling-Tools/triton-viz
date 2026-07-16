import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + value
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to call the kernel
def puzzle1(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    N = output.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    return output




##################################################################################################################################################


import torch

def test_puzzle():
    results = {}
    
    # Test case 1
    a1 = torch.Tensor([4, 5, 3, 2]).to(device=torch.device('cuda'))
    triton_output1 = puzzle1(a1)
    results['test_case_1'] = triton_output1
    
    # Test case 2
    a2 = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device=torch.device('cuda'))
    triton_output2 = puzzle1(a2)
    results['test_case_2'] = triton_output2
    
    # Test case 3
    a3 = torch.Tensor([10, 20, 30]).to(device=torch.device('cuda'))
    triton_output3 = puzzle1(a3)
    results['test_case_3'] = triton_output3
    
    # Test case 4
    a4 = torch.Tensor([0, -1, -2, -3]).to(device=torch.device('cuda'))
    triton_output4 = puzzle1(a4)
    results['test_case_4'] = triton_output4
    
    return results

result_gold = test_puzzle()
