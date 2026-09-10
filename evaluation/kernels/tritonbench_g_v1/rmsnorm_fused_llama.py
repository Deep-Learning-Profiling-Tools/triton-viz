
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)

def rmsnorm_forward(x, weight, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    BLOCK_SIZE = 128 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    num_warps = 8
    # enqueue kernel
    _rms_norm_fwd_fused[(M,)](x_arg, y, weight,
                              x_arg.stride(0), N, eps,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y




##################################################################################################################################################


import torch

# Test function for rmsnorm_forward
def test_rmsnorm_forward():
    results = {}
    
    # Test case 1: Small input tensor
    x1 = torch.randn(2, 64, dtype=torch.float16).cuda()
    weight1 = torch.randn(64, dtype=torch.float16).cuda()
    eps1 = 1e-5
    y1 = rmsnorm_forward(x1, weight1, eps1)
    results['test_case_1'] = y1
    
    # Test case 2: Larger input tensor
    x2 = torch.randn(4, 128, dtype=torch.float16).cuda()
    weight2 = torch.randn(128, dtype=torch.float16).cuda()
    eps2 = 1e-5
    y2 = rmsnorm_forward(x2, weight2, eps2)
    results['test_case_2'] = y2
    
    # Test case 3: Edge case with maximum supported feature dimension
    x3 = torch.randn(1, 8192, dtype=torch.float16).cuda()  # 8192 * 2 bytes = 16384 bytes < 64KB
    weight3 = torch.randn(8192, dtype=torch.float16).cuda()
    eps3 = 1e-5
    y3 = rmsnorm_forward(x3, weight3, eps3)
    results['test_case_3'] = y3

    # Test case 4: Edge case with minimum supported feature dimension
    x4 = torch.randn(1, 1, dtype=torch.float16).cuda()
    weight4 = torch.randn(1, dtype=torch.float16).cuda()
    eps4 = 1e-5
    y4 = rmsnorm_forward(x4, weight4, eps4)
    results['test_case_4'] = y4

    return results

# Run the test function
result_gold = test_rmsnorm_forward()
