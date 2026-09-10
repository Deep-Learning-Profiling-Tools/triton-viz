
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
@triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, RESIDUAL, X1, W1, B1, Y1, RESIDUAL_OUT, ROWSCALE, SEEDS, DROPOUT_MASK, Mean, Rstd,
    stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, stride_x1_row, stride_y1_row,
    M, N, eps, dropout_p, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr, HAS_ROWSCALE: tl.constexpr, HAS_X1: tl.constexpr,
    HAS_W1: tl.constexpr, HAS_B1: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = (
                tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            )
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)

def _layer_norm_fwd(
    x, weight, bias, eps, residual=None, x1=None, weight1=None, bias1=None, dropout_p=0.0,
    rowscale=None, out_dtype=None, residual_dtype=None, is_rms_norm=False, return_dropout_mask=False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(y)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    if (
        residual is not None
        or (residual_dtype is not None and residual_dtype != x.dtype)
        or dropout_p > 0.0
        or rowscale is not None
        or x1 is not None
    ):
        residual_out = torch.empty(
            M, N, device=x.device, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(
            2**32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64
        )
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M if x1 is None else 2 * M, N, device=x.device, dtype=torch.bool)
    else:
        dropout_mask = None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x, y, weight, bias, residual, x1, weight1, bias1, y1, residual_out, rowscale, seeds,
            dropout_mask, mean, rstd, x.stride(0), y.stride(0), residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0, x1.stride(0) if x1 is not None else 0,
            y1.stride(0) if y1 is not None else 0, M, N, eps, dropout_p, is_rms_norm, BLOCK_N,
            residual is not None, residual_out is not None, bias is not None, dropout_p > 0.0,
            dropout_mask is not None, rowscale is not None,
        )
    if dropout_mask is not None and x1 is not None:
        dropout_mask, dropout_mask1 = dropout_mask.tensor_split(2, dim=0)
    else:
        dropout_mask1 = None
    return (
        y, y1, mean, rstd, residual_out if residual_out is not None else x, seeds, dropout_mask, dropout_mask1,
    )




##################################################################################################################################################


import torch

def test_layer_norm_fwd():
    # Define the input parameters
    M, N = 64, 128  # Example dimensions
    eps = 1e-5
    dropout_p = 0.1

    # Create random input tensors
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    residual = torch.randn(M, N, device='cuda', dtype=torch.float32)
    x1 = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight1 = torch.randn(N, device='cuda', dtype=torch.float32)
    bias1 = torch.randn(N, device='cuda', dtype=torch.float32)
    rowscale = torch.randn(M, device='cuda', dtype=torch.float32)

    results = {}

    # Test case 1: Basic layer norm
    results['test_case_1'] = _layer_norm_fwd(x, weight, bias, eps)

    # Test case 2: Layer norm with residual
    results['test_case_2'] = _layer_norm_fwd(x, weight, bias, eps, residual=residual)

    # Test case 3: Layer norm with additional input tensors
    results['test_case_3'] = _layer_norm_fwd(x, weight, bias, eps, x1=x1, weight1=weight1, bias1=bias1)

    # Test case 4: Layer norm with dropout
    results['test_case_4'] = _layer_norm_fwd(x, weight, bias, eps, dropout_p=dropout_p, return_dropout_mask=True)

    # Test case 5: Layer norm with row scaling
    results['test_case_5'] = _layer_norm_fwd(x, weight, bias, eps, rowscale=rowscale)

    return results

result_gold = test_layer_norm_fwd()
