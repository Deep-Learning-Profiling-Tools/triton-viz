
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
    key=["BT", "BK", "BV"], 
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k,
    v,
    d, 
    v_new,
    h,
    initial_state,
    final_state,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_h_cumsum = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))   
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_d, b_h.to(b_k.dtype), allow_tf32=False)
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_h_cumsum += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h += b_h_cumsum      
        
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def chunk_fwd_h_fn(k, w, u, BT, initial_state, final_state):
    B, H, T, K, V = *k.shape, u.shape[-1]

    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    BV = 16 if BK > 128 else 32        
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32 
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    h = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    v_new = torch.empty_like(u)
    chunk_delta_rule_fwd_kernel_h[grid](
        k, u, w, v_new, h, initial_state, final_state,
        k.stride(1), k.stride(2), k.stride(3),
        u.stride(1), u.stride(2), u.stride(3),
        h.stride(1), h.stride(2),
        H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        )
    return h, v_new




##################################################################################################################################################


import torch

# Test function for chunk_fwd_h_fn
def test_chunk_fwd_h_fn():
    B, H, T, K, V = 2, 4, 128, 64, 64  # Example dimensions
    BT = 32  # Block size for T dimension

    k = torch.randn(B, H, K, T, dtype=torch.float32, device='cuda')
    w = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    u = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')

    results = {}

    # Test without initial and final states
    h, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state=None, final_state=None)
    results['test_case_1'] = (h.shape, v_new.shape)

    # Test with initial and final states
    initial_state = torch.zeros(B, H, K, V, dtype=torch.float32, device='cuda')
    final_state = torch.zeros(B, H, K, V, dtype=torch.float32, device='cuda')
    h, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state=initial_state, final_state=final_state)
    results['test_case_2'] = (h.shape, v_new.shape)

    return results

# Run tests
result_gold = test_chunk_fwd_h_fn()
