import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(grad_ptr,
                      p_ptr,
                      p_mask_ptr,
                      n_elements,
                      alpha,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * alpha
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    '''
    equivalent to
    grad.add_(p.data * (1 - p.mask), alpha=decay)
    '''
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    assert grad.stride() == p_data.stride() == p_mask.stride()
    n_elements = grad.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)



##################################################################################################################################################


import torch

# 测试代码
def test_masked_add():
    # 设置随机种子以保证结果可复现
    torch.manual_seed(0)
    n = 10000  # 选择较大的张量大小

    # 生成随机张量
    grad = torch.randn(n, device='cuda')
    p_data = torch.randn(n, device='cuda')
    p_mask = torch.randint(0, 2, (n,), device='cuda')  # 生成0或1的掩码

    # Triton版本
    results = {}
    
    # Test case 1
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask, alpha=0.5)
    results['test_case_1'] = grad_triton.clone()

    # Test case 2: alpha = 0
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask, alpha=0)
    results['test_case_2'] = grad_triton.clone()

    # Test case 3: all mask values are 0
    p_mask_zero = torch.zeros(n, device='cuda', dtype=torch.int32)
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask_zero, alpha=0.5)
    results['test_case_3'] = grad_triton.clone()

    # Test case 4: all mask values are 1
    p_mask_one = torch.ones(n, device='cuda', dtype=torch.int32)
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask_one, alpha=0.5)
    results['test_case_4'] = grad_triton.clone()

    return results

# 运行测试
result_gold = test_masked_add()
