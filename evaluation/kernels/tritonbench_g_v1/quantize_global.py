
import torch
import triton
import triton.language as tl

# global quantize
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
    tl.store(output_ptr + offsets, output, mask=mask)

def quantize_global(x: torch.Tensor):
    absmax = x.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    output = torch.empty(*x.shape, device="cuda", dtype=torch.int8)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _quantize_global[grid](x, absmax_inv, output, n_elements)
    return output, absmax




##################################################################################################################################################


import torch

# Test for quantize_global
def test_quantize_global():
    results = {}
    
    # Test case 1
    x1 = torch.randn(2048, device='cuda', dtype=torch.float32)
    output1, absmax1 = quantize_global(x1)
    results['test_case_1'] = (output1, absmax1)
    
    # Test case 2
    x2 = torch.randn(1024, device='cuda', dtype=torch.float32)
    output2, absmax2 = quantize_global(x2)
    results['test_case_2'] = (output2, absmax2)
    
    # Test case 3
    x3 = torch.randn(3072, device='cuda', dtype=torch.float32)
    output3, absmax3 = quantize_global(x3)
    results['test_case_3'] = (output3, absmax3)
    
    # Test case 4
    x4 = torch.randn(4096, device='cuda', dtype=torch.float32)
    output4, absmax4 = quantize_global(x4)
    results['test_case_4'] = (output4, absmax4)
    
    return results

# Run the tests
result_gold = test_quantize_global()
