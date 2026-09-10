
import triton
import triton.language as tl
import torch

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE : int = 65536

def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y  += row_idx * Y_row_stride
    X  += row_idx * X_row_stride
    r  += row_idx
    mu += row_idx

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask = mask, other = 0).to(tl.float32)

    mean_X  = tl.sum(X_row,   axis = 0) / n_cols
    XX      = X_row - mean_X
    row_var = tl.sum(XX * XX, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store (r, inv_var)
    tl.store (mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask = mask)

@triton.jit
def layernorm_backward(
    dY, dY_row_stride,
    X,   X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx
    mu += row_idx

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)
    b_row  = tl.load(b  + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r) .to(tl.float32)
    mean    = tl.load(mu).to(tl.float32)
    normed  = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis = 0) / n_cols - normed * tl.sum(dY_W * normed, axis = 0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask = mask)

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y  = torch.empty((n_rows, n_cols), dtype = X.dtype, device = "cuda:0")
        r  = torch.empty(n_rows, dtype = torch.float32, device = "cuda:0")
        mu = torch.empty(n_rows, dtype = torch.float32, device = "cuda:0")

        layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W,
            b,
            r,
            mu,
            n_cols, eps,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(X, W, b, r, mu)
        return Y.view(*shape)
    
    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, b, r, mu = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X .stride(0),
            W,
            b,
            r,
            mu,
            n_cols, ctx.eps,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None, None
    
def fast_layernorm(layernorm, X):
    assert(layernorm.elementwise_affine is True)
    W    = layernorm.weight
    bias = layernorm.bias
    eps = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_Layernorm.apply(X, W, bias, eps)
    return out




##################################################################################################################################################


import torch
import torch.nn as nn

def test_fast_layernorm_with_backward():
    # Set the parameters for the layer normalization
    batch_size = 4
    feature_size = 8
    eps = 1e-5

    # Create a random input tensor with gradient tracking enabled
    X = torch.randn(batch_size, feature_size, device='cuda:0', dtype=torch.float32, requires_grad=True)

    # Create a PyTorch LayerNorm module
    layernorm = nn.LayerNorm(feature_size, eps=eps, elementwise_affine=True).cuda()

    # Perform layer normalization using the fast_layernorm function
    Y = fast_layernorm(layernorm, X)

    # Compute a dummy loss (e.g., mean of the output)
    loss = Y.mean()

    # Perform backward propagation
    loss.backward()

    # Check the results for the single branch tested
    results = {"test_case_1": X.grad.clone()}

    return results

result_gold = test_fast_layernorm_with_backward()
# Coverage: [1/4]
