
import math
import torch
import triton
import triton.language as tl

_BLOCK_N = 64
_BLOCK_M = 64

@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _score_kernel(
    Q, K, M, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_oz, stride_oh, stride_on,
    Z, H, H_KV, #
    N_CTX,  #
    ROUND_CTX,
    NKV_CTX,
    sliding_window_offset,
    sliding_window_size,
    SLIDING_WINDOW: tl.constexpr,
    COMPLEMENT_SLIDING_WINDOW: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H//H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
    m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M)
    o = tl.zeros([BLOCK_M], dtype=tl.float32)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    if IS_EVEN_N:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")


    lo = 0
    hi = ROUND_CTX
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634   # 1/log(2)

    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        if IS_EVEN_M:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

        m = tl.load(m_ptrs)

        # calc qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale

        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[None, :] \
                 + start_m - start_n * BLOCK_N + sliding_window_offset

            if COMPLEMENT_SLIDING_WINDOW:
                mask = (dist >= sliding_window_size)
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)

        qk = qk - m[:, None]
        p = tl.math.exp2(qk) # (BLOCK_M, BLOCK_N)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_N:
            p = tl.where(
                ((tl.arange(0, BLOCK_M) + start_m) < N_CTX)[:, None],
                p, 0
            )

        o += tl.sum(p, axis=0)


        Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
        m_ptrs = m_ptrs + BLOCK_M

    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    o_range = tl.arange(0, BLOCK_N) + start_n * BLOCK_N # orange
    o_ptrs = Out + o_offset + o_range
    tl.store(o_ptrs, o.to(Out.type.element_ty), mask = o_range < NKV_CTX)

def get_score(q, k, m, sliding_window, complement_sliding_window):
    N_CTX = q.size(-2)
    NKV_CTX = k.size(-2)
    ROUND_CTX = m.size(-1)
    ret = torch.zeros(
        (q.size(0), q.size(1), k.size(2)),
        dtype=k.dtype, device=k.device
    )
    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None

    grid = lambda META: (
        triton.cdiv(k.shape[2], META["BLOCK_N"]),
        q.shape[0] * q.shape[1]
    )
    sm_scale = 1 / math.sqrt(q.size(-1))

    global _BLOCK_N
    global _BLOCK_M

    try:
        _score_kernel[grid](
            q, k, m, sm_scale, ret,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            ret.stride(0), ret.stride(1), ret.stride(2),
            q.size(0), q.size(1), k.size(1),
            N_CTX, ROUND_CTX, NKV_CTX,
            sliding_window_offset,
            sliding_window_size,
            SLIDING_WINDOW=(sliding_window is not None),
            COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_DMODEL=q.size(-1)
        )
    except triton.OutOfResources as E:
        from warnings import warn
        _BLOCK_N = _BLOCK_N // 2
        _BLOCK_M = _BLOCK_M // 2
        warn(f"Triton Attention Output Resources. {E}\nUse smaller block size {_BLOCK_N}.")
        _score_kernel[grid](
            q, k, m, sm_scale, ret,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            ret.stride(0), ret.stride(1), ret.stride(2),
            q.size(0), q.size(1), k.size(1),
            N_CTX, ROUND_CTX, NKV_CTX,
            sliding_window_offset,
            sliding_window_size,
            SLIDING_WINDOW=(sliding_window is not None),
            COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_DMODEL=q.size(-1)
        )

    return ret



##################################################################################################################################################


import torch

# Define the test function for get_score
def test_get_score():
    # Define input dimensions
    batch_size = 2
    num_heads = 4
    seq_len = 128
    d_model = 64

    # Create random input tensors
    q = torch.randn((batch_size, num_heads, seq_len, d_model), device='cuda', dtype=torch.float16)
    k = torch.randn((batch_size, num_heads, seq_len, d_model), device='cuda', dtype=torch.float16)
    m = torch.zeros((batch_size, num_heads, seq_len), device='cuda', dtype=torch.float32)

    # Define sliding window parameters
    sliding_window = (0, 64)
    complement_sliding_window = False

    # Call the get_score function
    ret1 = get_score(q, k, m, sliding_window, complement_sliding_window)

    # Test with complement_sliding_window = True
    complement_sliding_window = True
    ret2 = get_score(q, k, m, sliding_window, complement_sliding_window)

    # Test without sliding window
    sliding_window = None
    complement_sliding_window = False
    ret3 = get_score(q, k, m, sliding_window, complement_sliding_window)

    # Test with different sliding window size
    sliding_window = (0, 32)
    ret4 = get_score(q, k, m, sliding_window, complement_sliding_window)

    results = {
        "test_case_1": ret1,
        "test_case_2": ret2,
        "test_case_3": ret3,
        "test_case_4": ret4
    }
    return results

# Run the tests
result_gold = test_get_score()
