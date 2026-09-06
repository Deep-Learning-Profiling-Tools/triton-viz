
import torch
import triton
import triton.language as tl
import torch.nn as nn

# Kernel function for fused RMSNorm
@triton.jit
def rms_norm_fwd_fused(
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
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

# TritonLlamaRMSNorm class for integrating the kernel into a model
class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        rms_norm_fwd_fused[(M,)](
            x_arg,
            y,
            self.weight,
            x_arg.stride(0),
            N,
            self.variance_epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y




##################################################################################################################################################


# Define the test function
def test_triton_llama_rms_norm():
    results = {}
    
    # Test case 1: Small input size
    x1 = torch.randn(2, 16, dtype=torch.float32, device="cuda")
    weight1 = torch.ones(16, dtype=torch.float32, device="cuda")
    norm1 = TritonLlamaRMSNorm(weight1)
    y1 = norm1(x1)
    results['test_case_1'] = y1

    # Test case 2: Larger input size within 64KB limit
    x2 = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    weight2 = torch.ones(256, dtype=torch.float32, device="cuda")
    norm2 = TritonLlamaRMSNorm(weight2)
    y2 = norm2(x2)
    results['test_case_2'] = y2

    # Test case 3: Input size at the edge of 64KB limit
    x3 = torch.randn(1, 65536 // 4, dtype=torch.float32, device="cuda")  # 65536 bytes / 4 bytes per float
    weight3 = torch.ones(65536 // 4, dtype=torch.float32, device="cuda")
    norm3 = TritonLlamaRMSNorm(weight3)
    y3 = norm3(x3)
    results['test_case_3'] = y3

    # Test case 4: Input size exceeding 64KB limit (should raise an error)
    try:
        x4 = torch.randn(1, 65536 // 4 + 1, dtype=torch.float32, device="cuda")
        weight4 = torch.ones(65536 // 4 + 1, dtype=torch.float32, device="cuda")
        norm4 = TritonLlamaRMSNorm(weight4)
        y4 = norm4(x4)
    except RuntimeError as e:
        results['test_case_4'] = str(e)

    return results

# Run the test function
result_gold = test_triton_llama_rms_norm()
