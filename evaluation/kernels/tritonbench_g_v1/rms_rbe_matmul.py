import torch

import triton
import triton.language as tl


torch.manual_seed(1234)


@triton.jit
def rbe_triton(x_ptr, out_ptr,
               M, K,
               stride_x_batch, stride_x_m, stride_x_n,
               stride_out_batch, stride_out_m, stride_out_n,
               start_token_position,
               THETA: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2) * 2  # take only even numbers
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + 1, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
            pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + 1, out_imag, mask=out_imag_mask)


@triton.jit
def rms_matmul_rbe(
        x_ptr, w_ptr, rms_w_ptr, out_ptr,
        M, N, K,
        stride_x_batch, stride_x_m, stride_x_k,
        stride_w_k, stride_w_n,
        stride_rms_w,
        stride_out_batch, stride_out_m, stride_out_n,
        start_token_position,
        USE_FP8: tl.constexpr,
        RBE_EPILOGUE: tl.constexpr,
        THETA: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Prologue: RMS
    Epilogue: nothing or Rotary embeddings
    c = ROBE((rms(a) * rms_w) @ b)
    """
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs)  # TODO add an assert that rms_w is a multiple of BLOCK SIZE K
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs)  # TODO add an assert that w is a multiple of BLOCK SIZE K
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)


def rms_matmul_rbe_wrapper(x: torch.Tensor, weight: torch.Tensor, rms_w: torch.Tensor, use_rbe: bool, start_pos: int,
                           n_heads: int, head_dim: int):
    # 确保 weight 和 rms_w 的数据类型一致
    assert weight.dtype in [torch.float16, torch.int8], "Only torch.float16 or torch.int8 are supported for weight"
    
    # 确保 rms_w 和 weight 的 dtype 一致
    if rms_w.dtype != weight.dtype:
        # print(f"rms_w dtype: {rms_w.dtype}, weight dtype: {weight.dtype}")
        rms_w = rms_w.to(weight.dtype)  # 如果类型不一致，将 rms_w 转换为与 weight 一致的类型

    # 创建输出张量
    batch, M, K = x.shape
    weight_t = weight.t()
    K_W, N = weight_t.shape
    assert K == K_W

    out = torch.empty((batch, M, N), dtype=weight_t.dtype, device=weight_t.device)
    out_ptr = triton.reinterpret(out, tl.float8e5 if out.dtype == torch.int8 else tl.float16)

    grid = lambda META: (
        batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]))

    rms_matmul_rbe[grid](
        x_ptr=x,
        w_ptr=weight_t, rms_w_ptr=rms_w, out_ptr=out_ptr,
        M=M, N=N, K=K,
        stride_x_batch=x.stride(0), stride_x_m=x.stride(1), stride_x_k=x.stride(2),
        stride_w_k=weight_t.stride(0), stride_w_n=weight_t.stride(1),
        stride_rms_w=rms_w.stride(0),
        stride_out_batch=out.stride(0), stride_out_m=out.stride(1), stride_out_n=out.stride(2),
        start_token_position=start_pos,
        USE_FP8=weight_t.dtype == torch.int8,
        RBE_EPILOGUE=use_rbe,
        THETA=10000.,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=4, num_warps=4
    )
    out = out.view(batch, M, n_heads, head_dim)
    return out


##################################################################################################################################################


def test_rms_matmul_rbe():
    batch, M, K = 2, 4, 1024
    N = 64
    n_heads = 8
    assert N % n_heads == 0
    head_dim = N // n_heads

    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    weight = torch.randn((N, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')

    test_results = {}

    # Test case 1: use_rbe = False, weight dtype = float16
    use_rbe = False
    start_pos = 0
    out = rms_matmul_rbe_wrapper(x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim)
    test_results['test_case_1'] = out

    # Test case 2: use_rbe = True, weight dtype = float16
    use_rbe = True
    out = rms_matmul_rbe_wrapper(x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim)
    test_results['test_case_2'] = out

    # Test case 3: use_rbe = False, weight dtype = int8
    weight_int8 = weight.to(torch.int8)
    use_rbe = False
    out = rms_matmul_rbe_wrapper(x, weight_int8, rms_w, use_rbe, start_pos, n_heads, head_dim)
    test_results['test_case_3'] = out

    # Test case 4: use_rbe = True, weight dtype = int8
    use_rbe = True
    out = rms_matmul_rbe_wrapper(x, weight_int8, rms_w, use_rbe, start_pos, n_heads, head_dim)
    test_results['test_case_4'] = out

    return test_results

result_gold = test_rms_matmul_rbe()
