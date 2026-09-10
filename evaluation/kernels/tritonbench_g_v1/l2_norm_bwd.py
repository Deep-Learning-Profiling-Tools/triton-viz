
import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x) 
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)

def _l2_norm_bwd(
    x, dy, eps=1e-5,
):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _l2_norm_bwd_kernel[(M,)](
            x,
            dy,
            dx,
            x.stride(0),
            N,
            eps,
            BLOCK_N,
        )
    return dx.reshape(x_shape_og)




##################################################################################################################################################


import torch

# Test the backward L2 normalization
def test_l2_norm_bwd():
    results = {}
    
    # Test case 1: Default case
    x = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    dy = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_1'] = dx

    # Test case 2: Different shape
    x = torch.randn(2, 16, device='cuda', dtype=torch.float32)
    dy = torch.randn(2, 16, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_2'] = dx

    # Test case 3: Larger tensor
    x = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    dy = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_3'] = dx

    # Test case 4: Edge case with small tensor
    x = torch.randn(1, 8, device='cuda', dtype=torch.float32)
    dy = torch.randn(1, 8, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_4'] = dx

    return results

# Run the tests
result_gold = test_l2_norm_bwd()
