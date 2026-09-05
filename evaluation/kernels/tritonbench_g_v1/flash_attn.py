
import torch
import triton
import triton.language as tl

def flash_attn_triton(q, k, v, causal=True, sm_scale=1):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        L,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=4)

    return o


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    O,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_k_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom))
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16))




##################################################################################################################################################


# Test cases for the flash_attn_triton function
def test_flash_attn_triton():
    batch_size = 2
    num_heads = 2
    seq_len = 128
    dim = 64

    # Create random input tensors
    q = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')
    k = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')
    v = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')

    # Test with causal=True
    output_causal = flash_attn_triton(q, k, v, causal=True, sm_scale=1.0)

    # Test with causal=False
    output_non_causal = flash_attn_triton(q, k, v, causal=False, sm_scale=1.0)

    results = {
        "test_case_1": output_causal,
        "test_case_2": output_non_causal
    }

    return results

# Run the test
result_gold = test_flash_attn_triton()
