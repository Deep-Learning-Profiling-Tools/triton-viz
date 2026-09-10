
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dg,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    # Kernel implementation
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    last_idx = min(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + i_bh * T + last_idx)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_dg_last += (tl.sum(b_h * b_dh))
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < min(BT, T-i_t*BT) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_k*n_bh + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

def chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale):
    # Kernel call
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, B, H, T, dtype=torch.float32, device=g.device).fill_(-1e9)
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh, dq, dk, dg,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT
    )
    return dq, dk, dg




##################################################################################################################################################


import torch

# Define the test function for the backward kernel
def test_chunk_bwd_dqkg_fn():
    B, H, T, K, V = 2, 4, 128, 64, 64  # Example dimensions
    scale = 0.1  # Example scale factor

    # Create random input tensors
    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    g = torch.randn(B, H, T, dtype=torch.float32, device='cuda')
    h = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    do = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    dh = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')

    # Initialize a dictionary to store results
    results = {}

    # Test case 1
    dq, dk, dg = chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)
    results['test_case_1'] = (dq, dk, dg)

    # Test case 2 with different scale
    scale = 0.2
    dq, dk, dg = chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)
    results['test_case_2'] = (dq, dk, dg)

    # Test case 3 with different dimensions
    B, H, T, K, V = 3, 2, 256, 32, 32
    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    g = torch.randn(B, H, T, dtype=torch.float32, device='cuda')
    h = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    do = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    dh = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    dq, dk, dg = chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)
    results['test_case_3'] = (dq, dk, dg)

    # Test case 4 with different input values
    q = torch.ones(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.ones(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.ones(B, H, T, V, dtype=torch.float32, device='cuda')
    g = torch.ones(B, H, T, dtype=torch.float32, device='cuda')
    h = torch.ones(B, H, K, V, dtype=torch.float32, device='cuda')
    do = torch.ones(B, H, T, V, dtype=torch.float32, device='cuda')
    dh = torch.ones(B, H, K, V, dtype=torch.float32, device='cuda')
    dq, dk, dg = chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)
    results['test_case_4'] = (dq, dk, dg)

    return results

# Run the test function
result_gold = test_chunk_bwd_dqkg_fn()
