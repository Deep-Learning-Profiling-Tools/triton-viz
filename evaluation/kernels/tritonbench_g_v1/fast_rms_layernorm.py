
import triton
import triton.language as tl
import torch
import torch.nn as nn


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
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)

@triton.heuristics({"GEMMA": lambda args: args["GEMMA"],})
@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask=mask)

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma=False):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda:0")
        r = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        fx[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = X

        _rms_layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X.stride(0),
            W,  W.stride(0),
            r,  r.stride(0),
            dW, dW.stride(0),
            n_cols, ctx.eps,
            GEMMA=ctx.GEMMA,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None

def fast_rms_layernorm(layernorm, X, gemma=False):
    W = layernorm.weight
    eps = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out


# Define a simple layernorm module for testing
class SimpleLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(SimpleLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape).cuda())
        self.eps = eps



##################################################################################################################################################


# Test function for the fast_rms_layernorm with backward pass
def test_fast_rms_layernorm_with_backward():
    # Create a random input tensor with gradient tracking
    X = torch.randn(2, 4, 8, device='cuda', dtype=torch.float32, requires_grad=True)

    # Create a layernorm instance
    layernorm = SimpleLayerNorm(normalized_shape=X.shape[-1])

    results = {}

    # Test without GEMMA
    output = fast_rms_layernorm(layernorm, X, gemma=False)
    loss = output.mean()  # Compute a dummy loss

    # Perform backward propagation
    loss.backward()
    results['test_case_1'] = X.grad.clone()

    # Reset gradients for the next test
    X.grad.zero_()

    # Test with GEMMA
    output_gemma = fast_rms_layernorm(layernorm, X, gemma=True)
    loss_gemma = output_gemma.mean()  # Compute a dummy loss

    # Perform backward propagation
    loss_gemma.backward()
    results['test_case_2'] = X.grad.clone()

    return results

# Run the test
result_gold = test_fast_rms_layernorm_with_backward()
