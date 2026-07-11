
import torch
import triton
import triton.language as tl
import math

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
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd, 
    stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, 
    N, eps, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr, 
    HAS_RESIDUAL: tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_BIAS: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
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


def _layer_norm_fwd(
    x, weight, bias, eps, residual=None, out_dtype=None, 
    residual_dtype=None, is_rms_norm=False
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
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device="cuda") if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x, y, weight, bias, residual, residual_out, 
            mean, rstd, x.stride(0), y.stride(0), 
            residual.stride(0) if residual is not None else 0, 
            residual_out.stride(0) if residual_out is not None else 0, 
            N, eps, is_rms_norm, BLOCK_N, residual is not None, 
            residual_out is not None, bias is not None
        )
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, DRESIDUAL_IN, 
    Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row, 
    stride_dx_row, stride_dres_row, stride_dres_in_row, M, 
    N, eps, rows_per_program, IS_RMS_NORM: tl.constexpr, 
    BLOCK_N: tl.constexpr, HAS_DRESIDUAL: tl.constexpr, 
    STORE_DRESIDUAL: tl.constexpr, HAS_BIAS: tl.constexpr, 
    RECOMPUTE_OUTPUT: tl.constexpr
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row) if Mean is not None else 0.0  # 修改此行
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def _layer_norm_bwd(
    dy, x, weight, bias, eps, mean, rstd, dresidual=None, 
    has_residual=False, is_rms_norm=False, x_dtype=None, recompute_output=False
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = (
        torch.empty((sm_count, N), dtype=torch.float32, device=bias.device)
        if bias is not None
        else None
    )
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x, weight, bias, y, dy, dx, _dw, _db, dresidual, 
            dresidual_in, mean, rstd, x.stride(0), 
            0 if not recompute_output else y.stride(0), dy.stride(0), 
            dx.stride(0), dresidual.stride(0) if dresidual is not None else 0, 
            dresidual_in.stride(0) if dresidual_in is not None else 0, 
            M, N, eps, rows_per_program, is_rms_norm, BLOCK_N, 
            dresidual is not None, dresidual_in is not None, bias is not None
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)




##################################################################################################################################################


def test_layer_norm_fwd_bwd():
    # 设置测试的基本参数
    M, N = 64, 1024  # 64x1024的矩阵
    x = torch.randn(M, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.float32, device='cuda')
    eps = 1e-6

    results = {}

    # 测试不使用 RMS norm，且没有残差，且不计算输出
    y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps, residual=None, is_rms_norm=False)
    results['test_case_1'] = (y, mean, rstd, residual_out)

    dy = torch.randn_like(y)
    dx, dw, db, dresidual_in = _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd)
    results['test_case_2'] = (dx, dw, db, dresidual_in)

    # 测试使用 RMS norm，且没有残差，且不计算输出
    y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps, residual=None, is_rms_norm=True)
    results['test_case_3'] = (y, mean, rstd, residual_out)

    dy = torch.randn_like(y)
    dx, dw, db, dresidual_in = _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, is_rms_norm=True)
    results['test_case_4'] = (dx, dw, db, dresidual_in)

    # 测试带有残差的情况，且不计算输出
    residual = torch.randn_like(x)
    y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps, residual=residual, is_rms_norm=False)
    results['test_case_5'] = (y, mean, rstd, residual_out)

    dy = torch.randn_like(y)
    dx, dw, db, dresidual_in = _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, dresidual=residual, is_rms_norm=False)
    results['test_case_6'] = (dx, dw, db, dresidual_in)

    # 测试计算输出（recompute_output=True）
    y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps, residual=None, is_rms_norm=False)
    dy = torch.randn_like(y)
    dx, dw, db, dresidual_in, recomputed_y = _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, recompute_output=True)
    results['test_case_7'] = (dx, dw, db, dresidual_in, recomputed_y)

    # 测试带有残差的情况，计算输出
    residual = torch.randn_like(x)
    y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps, residual=residual, is_rms_norm=False)
    dy = torch.randn_like(y)
    dx, dw, db, dresidual_in, recomputed_y = _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, dresidual=residual, recompute_output=True)
    results['test_case_8'] = (dx, dw, db, dresidual_in, recomputed_y)

    return results

result_gold = test_layer_norm_fwd_bwd()

print(result_gold)