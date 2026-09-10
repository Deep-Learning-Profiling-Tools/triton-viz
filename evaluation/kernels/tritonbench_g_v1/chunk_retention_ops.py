
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    # Triton kernel code for forward pass of chunk retention with initial and final state handling
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if i_t == NT - 1 and (T % BT) != 0:
            d_b = tl.math.exp2((T % BT) * b_b)
            d_i = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    scale, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Triton kernel code for forward pass of chunk retention with output scaling
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s *= d_s
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_bwd_kernel_dh(
    q, do, dh, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    scale, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr
):
    # Triton kernel code for backward pass of chunk retention, computing gradients for hidden state
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=False)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q, k, v, h, do, dh, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    scale, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr
):
    # Triton kernel code for backward pass of chunk retention, computing gradients for q, k, v
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    n_bh = tl.num_programs(2)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    d_q = (d_q * scale).to(d_q.dtype)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    b_ds = (b_ds * d_s).to(b_q.dtype)
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h_fn(k, v, BT, initial_state, output_final_state):
    B, H, T, K, V = *k.shape, v.shape[-1]
    final_state = None
    if output_final_state:
        final_state = k.new_empty(B, H, K, V, dtype=torch.float32)
    BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    h = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    chunk_retention_fwd_kernel_h[grid](
        k, v, h, initial_state, final_state,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2),
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state
    )
    return h, final_state


def chunk_fwd_o_fn(h, q, k, v, BT, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.empty_like(v)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H)
    chunk_retention_fwd_kernel_o[grid](
        q, k, v, h, o,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2),
        scale,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return o


def chunk_bwd_dh_fn(do, q, k, v, BT, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    dh = k.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    chunk_retention_bwd_kernel_dh[grid](
        q, do, dh,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        dh.stride(1), dh.stride(2),
        scale,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT
    )
    return dh


def chunk_bwd_dqkv_fn(do, q, k, v, h, dh, scale):
    B, H, T, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = v.new_empty(NK, *v.shape)
    chunk_retention_bwd_kernel_dqkv[grid](
        q, k, v, h, do, dh, dq, dk, dv,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2),
        scale,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT
    )
    dv = dv.sum(0)
    return dq, dk, dv


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, initial_state, output_final_state, scale, checkpoint_level):
        BT = 64
        h, final_state = chunk_fwd_h_fn(k, v, BT, initial_state, output_final_state)
        o = chunk_fwd_o_fn(h, q, k, v, BT, scale)
        if checkpoint_level == 1:
            h = None
        ctx.save_for_backward(q, k, v, h, initial_state)
        ctx.BT, ctx.scale = BT, scale
        return o.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do, d_ht=None):
        BT, scale = ctx.BT, ctx.scale
        q, k, v, h, initial_state = ctx.saved_tensors
        if h is None:
            h, _ = chunk_fwd_h_fn(k, v, BT, initial_state, False)
        dh = chunk_bwd_dh_fn(do, q, k, v, BT, scale)
        dq, dk, dv = chunk_bwd_dqkv_fn(do, q, k, v, h, dh, scale)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    scale: float = None,
    checkpoint_level: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert checkpoint_level in [0, 1], "checkpoint_level must be 0, 1"
    assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dimensions (b, h, l, d)"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    if scale is None:
        scale = q.size(-1) ** -0.5
    o, final_state = ChunkRetentionFunction.apply(
        q, k, v, initial_state, output_final_state, scale, checkpoint_level)
    return o, final_state




##################################################################################################################################################


import torch

def test_chunk_retention_with_backward():
    # Define dimensions
    B, H, T, K, V = 2, 4, 128, 64, 64

    # Create random input tensors
    q = torch.randn(B, H, T, K, dtype=torch.float32, requires_grad=True, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, requires_grad=True, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, requires_grad=True, device='cuda')

    # Test case 1: Without initial state and without final state output
    o, _ = chunk_retention(q, k, v, output_final_state=False, checkpoint_level=0)
    loss = o.sum()  # Define a simple loss function
    loss.backward()  # Perform backward pass

    # Reset gradients for the next test
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # Test case 2: With initial state and final state output
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32, requires_grad=True, device='cuda')
    o, final_state = chunk_retention(q, k, v, initial_state=initial_state, output_final_state=True, checkpoint_level=1)
    loss = o.sum() + final_state.sum()
    loss.backward()

    # Reset gradients for the next test
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # Test case 3: Different checkpoint levels
    for checkpoint_level in [0, 1]:
        o, _ = chunk_retention(q, k, v, output_final_state=False, checkpoint_level=checkpoint_level)
        loss = o.sum()
        loss.backward()
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()

    # Test case 4: Verify all kernels are executed correctly
    h, _ = chunk_fwd_h_fn(k, v, BT=64, initial_state=None, output_final_state=False)
    o = chunk_fwd_o_fn(h, q, k, v, BT=64, scale=0.1)
    dh = chunk_bwd_dh_fn(o, q, k, v, BT=64, scale=0.1)
    dq, dk, dv = chunk_bwd_dqkv_fn(o, q, k, v, h, dh, scale=0.1)

    # Collect results in a dictionary
    results = {
        "test_case_1": (o.shape, loss.item()),
        "test_case_2": (o.shape, final_state.shape, loss.item()),
        "test_case_3": [(o.shape, loss.item()) for _ in range(2)],
        "test_case_4": (h.shape, o.shape, dh.shape, dq.shape, dk.shape, dv.shape)
    }
    return results

# Execute the test function
result_gold = test_chunk_retention_with_backward()
