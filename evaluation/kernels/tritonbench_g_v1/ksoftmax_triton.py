
import torch
import triton
import triton.language as tl

def get_depth(K):
    return triton.next_power_of_2(K)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['Y'].dtype == torch.float16})
@triton.jit
def _softmax(
    Y, X, M,
    stride_ym, stride_yn,
    stride_xm, stride_xn,
    stride_m,
    K,
    LOG: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    """
    Fused softmax kernel over a 3d tensor.
    The softmax is applied over the last dimension, equivalent to torch.softmax(tensor, dim=-1)
    """
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, DEPTH)
    x_ptrs = X + m * stride_xm + n * stride_xn + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
    if CAUSAL:
        off = float("-inf")
        off = off.to(x.dtype)
        x = tl.where(k > n, off, x)
    if MASK_TYPE is not None:
        if MASK_TYPE == 'qk':
            mask_ptrs = M + n * stride_m + k
        elif MASK_TYPE == 'bk':
            mask_ptrs = M + m * stride_m + k
        add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
        x += add_mask
    z = x - tl.max(x, axis=0)
    if IS_FP16:
        z = z.to(tl.float32)
    num = tl.exp(z)
    denom = tl.sum(num, axis=0)
    if LOG:
        y = z - tl.log(denom)
    else:
        y = num / denom
    y_ptrs = Y + m * stride_ym + n * stride_yn + k
    tl.store(y_ptrs, y, mask=k < K)

def softmax(Y, X, M=None, log=False, mask_type=None, causal=False):
    assert X.ndim == 3, "Input tensor X must be 3D"
    assert Y.shape == X.shape, "Output tensor Y must have the same shape as X"
    M = M if M is not None else torch.empty(0, device=X.device)
    
    K = X.shape[-1]
    stride_ym, stride_yn = Y.stride()[:-1]
    stride_xm, stride_xn = X.stride()[:-1]
    stride_m = M.stride(-1) if M.numel() > 0 else 0

    grid = (X.shape[0], X.shape[1])
    _softmax[grid](
        Y, X, M,
        stride_ym, stride_yn,
        stride_xm, stride_xn,
        stride_m,
        K,
        LOG=log,
        MASK_TYPE=mask_type,
        CAUSAL=causal
    )

def get_depth(K):
    return triton.next_power_of_2(K)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['GradIn'].dtype == torch.float16})
@triton.jit
def _softmax_backward(
    GradIn, GradOut, Out,
    stride_bm, stride_bn,
    stride_gm, stride_gn,
    stride_om, stride_on,
    K,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    """
    Compute the softmax gradients.
    """
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, DEPTH)
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))
    if CAUSAL:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)
    if LOG:
        s = tl.sum(g, 0)
        if IS_FP16:
            o = o.to(tl.float32)
        grad_in = g - tl.exp(o) * s
    else:
        s = tl.sum(g * o, 0)
        grad_in = o * (g - s)
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)


def softmax_backward(GradIn, GradOut, Out, log=False, causal=False):
    assert GradOut.shape == Out.shape, "GradOut and Out must have the same shape"
    assert GradIn.shape == Out.shape, "GradIn and Out must have the same shape"
    
    K = Out.shape[-1]
    stride_bm, stride_bn = GradIn.stride()[:-1]
    stride_gm, stride_gn = GradOut.stride()[:-1]
    stride_om, stride_on = Out.stride()[:-1]

    grid = (Out.shape[0], Out.shape[1])
    _softmax_backward[grid](
        GradIn, GradOut, Out,
        stride_bm, stride_bn,
        stride_gm, stride_gn,
        stride_om, stride_on,
        K,
        LOG=log,
        CAUSAL=causal
    )





##################################################################################################################################################


import torch

def test_softmax():
    # Initialize test tensors
    B, M, N = 2, 3, 8  # Batch size, Rows, Columns
    X = torch.randn((B, M, N), dtype=torch.float32, device="cuda", requires_grad=True)
    Y = torch.empty_like(X)
    M_mask = torch.randn((B, N), dtype=torch.float32, device="cuda")

    # Triton Softmax forward pass
    softmax(Y, X, M_mask, log=False, mask_type='qk', causal=True)
    test_case_1 = Y.clone()

    softmax(Y, X, M_mask, log=True, mask_type='qk', causal=True)
    test_case_2 = Y.clone()

    softmax(Y, X, M_mask, log=False, mask_type='bk', causal=False)
    test_case_3 = Y.clone()

    softmax(Y, X, M_mask, log=True, mask_type='bk', causal=False)
    test_case_4 = Y.clone()

    # Triton Softmax backward pass
    GradOut = torch.randn_like(Y, device="cuda")
    GradIn = torch.empty_like(X)
    softmax_backward(GradIn, GradOut, Y, log=False, causal=True)
    test_case_5 = GradIn.clone()

    softmax_backward(GradIn, GradOut, Y, log=True, causal=True)
    test_case_6 = GradIn.clone()

    softmax_backward(GradIn, GradOut, Y, log=False, causal=False)
    test_case_7 = GradIn.clone()

    softmax_backward(GradIn, GradOut, Y, log=True, causal=False)
    test_case_8 = GradIn.clone()

    return {
        "test_case_1": test_case_1,
        "test_case_2": test_case_2,
        "test_case_3": test_case_3,
        "test_case_4": test_case_4,
        "test_case_5": test_case_5,
        "test_case_6": test_case_6,
        "test_case_7": test_case_7,
        "test_case_8": test_case_8,
    }

result_gold = test_softmax()
