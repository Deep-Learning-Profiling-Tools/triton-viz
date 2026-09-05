
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s,
    o,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k,
    v,
    g,
    h,
    h0,
    ht,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    GATEK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if GATEK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[:, None]
            # [BK, BT]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            # [BV,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[None, :]
            # [BT, BV]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def fwd_pre(g, B, H, T, S, BT):
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_gated_abc_fwd_kernel_cum[grid](
        g_org, g,
        g.stride(1), g.stride(2), g.stride(3),
        T=T, S=S, BT=BT
    )
    return g


def fwd_inner(q, k, v, g, B, H, T, K, V, BT, BK, BV, gatek=False, h0=None, ht=None):
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h = q.new_empty(B, H, NT * K, V)
    grid = (NV, NK, B * H)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h, h0, ht,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        GATEK=gatek,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=ht is not None,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return h




##################################################################################################################################################


def test_fwd_pre_inner():
    # Define the input parameters
    B, H, T, S, K, V = 2, 4, 128, 64, 32, 32  # Batch size, heads, sequence length, etc.
    BT, BK, BV = 32, 16, 16  # Block sizes
    g = torch.randn(B, H, T, S, dtype=torch.float16, device='cuda')
    q = torch.randn(B, H, T, V, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, K, T, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float16, device='cuda')
    h0 = torch.randn(B, H, K, V, dtype=torch.float16, device='cuda')
    ht = torch.empty_like(h0)

    # Test the fwd_pre function
    g_cum = fwd_pre(g, B, H, T, S, BT)

    # Test the fwd_inner function with different branches
    results = {}
    # Case 1: Without initial and final state, gatek=False
    results['test_case_1'] = fwd_inner(q, k, v, g_cum, B, H, T, K, V, BT, BK, BV, gatek=False)

    # Case 2: With initial state, without final state, gatek=True
    results['test_case_2'] = fwd_inner(q, k, v, g_cum, B, H, T, K, V, BT, BK, BV, gatek=True, h0=h0)

    # Case 3: With initial and final state, gatek=False
    results['test_case_3'] = fwd_inner(q, k, v, g_cum, B, H, T, K, V, BT, BK, BV, gatek=False, h0=h0, ht=ht)

    # Case 4: Without initial state, with final state, gatek=True
    results['test_case_4'] = fwd_inner(q, k, v, g_cum, B, H, T, K, V, BT, BK, BV, gatek=True, ht=ht)

    return results

result_gold = test_fwd_pre_inner()
