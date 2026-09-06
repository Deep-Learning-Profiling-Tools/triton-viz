
import torch
import triton
import triton.language as tl

@triton.jit
def ff_llama(
    a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    a_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        a = a * rms_w
        b = tl.load(w1_ptrs)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True)
            b = b.to(tl.float32)
            b = b.to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True)
            c = c.to(tl.float32)
            c = c.to(tl.float16)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w

    a_mean = tl.sum(a_sum, axis=1) / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert w1.dtype == w3.dtype == rms_w.dtype
    assert w1.dtype in [torch.int8, torch.float16]
    assert w1.shape == w3.shape

    w1_t = w1.t()
    w3_t = w3.t()

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim

    N = w1_t.shape[1]
    assert K == w1_t.shape[0]
    assert w1_t.shape == w3_t.shape
    x_reshape = x.reshape(M, K)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)
    ff_llama[grid](
        x_reshape, w1_t, w3_t, out, rms_w,
        M, N, K,
        *x_reshape.stride(),
        *w1_t.stride(),
        *w3_t.stride(),
        *out.stride(),
        *rms_w.stride(),
        USE_FP8=w1_t.dtype != torch.float16,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
        num_stages=2, num_warps=4
    )
    out = out.view(batch, seq_len, -1)
    return out



##################################################################################################################################################


# Test case for float16 weights
def test_ff_llama():
    results = {}
    batch, seq_len, dim = 2, 4, 64
    x = torch.randn((batch, seq_len, dim), dtype=torch.float16, device='cuda')
    w1 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    w3 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((dim,), dtype=torch.float16, device='cuda')

    out = kernel_ff(x, w1, w3, rms_w)
    results["test_case_1"] = out

    # Test case for different batch size
    batch, seq_len, dim = 3, 4, 64
    x = torch.randn((batch, seq_len, dim), dtype=torch.float16, device='cuda')
    w1 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    w3 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((dim,), dtype=torch.float16, device='cuda')

    out = kernel_ff(x, w1, w3, rms_w)
    results["test_case_3"] = out

    # Test case for different sequence length
    batch, seq_len, dim = 2, 5, 64
    x = torch.randn((batch, seq_len, dim), dtype=torch.float16, device='cuda')
    w1 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    w3 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((dim,), dtype=torch.float16, device='cuda')

    out = kernel_ff(x, w1, w3, rms_w)
    results["test_case_4"] = out
    return results

# Run tests
result_gold_float16 = test_ff_llama()
