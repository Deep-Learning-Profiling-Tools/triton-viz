
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
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
    BV: tl.constexpr
):
    # Kernel implementation
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_fwd_o_fn(h, q, k, v, g, BT, scale):
    # Kernel call
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.empty_like(v)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H)
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return o



##################################################################################################################################################


import torch

# Define the test function for the forward kernel
def test_chunk_fwd_o_fn():
    B, H, T, K, V = 2, 4, 128, 64, 64  # Example dimensions
    BT = 32  # Block size for T
    scale = 0.1  # Example scale factor

    # Create random input tensors
    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    h = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    g = torch.randn(B, H, T, dtype=torch.float32, device='cuda')

    # Dictionary to store results
    results = {}

    # Test case 1
    o1 = chunk_fwd_o_fn(h, q, k, v, g, BT, scale)
    results['test_case_1'] = o1

    # Test case 2: Different BT
    BT = 64
    o2 = chunk_fwd_o_fn(h, q, k, v, g, BT, scale)
    results['test_case_2'] = o2

    # Test case 3: Different scale
    scale = 0.2
    o3 = chunk_fwd_o_fn(h, q, k, v, g, BT, scale)
    results['test_case_3'] = o3

    # Test case 4: Different dimensions
    B, H, T, K, V = 1, 2, 64, 32, 32
    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    h = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    g = torch.randn(B, H, T, dtype=torch.float32, device='cuda')
    o4 = chunk_fwd_o_fn(h, q, k, v, g, BT, scale)
    results['test_case_4'] = o4

    return results

# Execute the test function
result_gold = test_chunk_fwd_o_fn()
