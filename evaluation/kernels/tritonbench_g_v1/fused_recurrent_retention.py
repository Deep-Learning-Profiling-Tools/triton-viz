
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state, 
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d, 
    B, H, T, scale, 
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, 
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    # Kernel logic
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        h = b_b * h + _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state, 
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d, 
    B, H, T, scale, 
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, 
    USE_INITIAL_STATE: tl.constexpr
):
    # Kernel logic
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h = b_b * h + _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += DK
        p_do += DV
        p_v += DV
        p_dq += DK

    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + (T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)

        d_h *= b_b
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do -= DV
        p_q -= DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV

class FusedRecurrentRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, output_final_state=False):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]

        scale = d_head_qk ** -0.5
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)

        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None

        grid = (NV, NK, batch_size * n_heads)
        fused_recurrent_retention_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = d_head_qk ** -0.5

        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NK, batch_size * n_heads)

        fused_recurrent_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv, None, None

def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentRetentionFunction.apply(q, k, v, initial_state, output_final_state)
    return o, final_state




##################################################################################################################################################


import torch

# Extended test function with backward propagation
def test_fused_recurrent_retention_with_backward():
    test_results = {}

    # Test parameters
    batch_size = 2
    n_heads = 4
    seq_len = 8
    d_head_qk = 16
    d_head_v = 16

    # Create random input tensors
    q = torch.randn(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32, requires_grad=True, device='cuda')
    k = torch.randn(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32, requires_grad=True, device='cuda')
    v = torch.randn(batch_size, n_heads, seq_len, d_head_v, dtype=torch.float32, requires_grad=True, device='cuda')

    # Test 1: Without initial state and without final state
    initial_state = None
    output_final_state = False
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=output_final_state)
    loss = o.sum()  # Define a simple loss function
    loss.backward()  # Perform backward pass
    test_results['test_case_1'] = {
        "output_shape": o.shape,
        "final_state": final_state,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item()
    }

    # Reset gradients for the next test
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # Test 2: With initial state and without final state
    initial_state = torch.randn(batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, device='cuda', requires_grad=True)
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=False)
    loss = o.sum()
    loss.backward()
    test_results['test_case_2'] = {
        "output_shape": o.shape,
        "final_state": final_state,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item(),
    }

    # Reset gradients for the next test
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # Test 3: With initial state and with final state
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=True)
    loss = o.sum() + final_state.sum()
    loss.backward()
    test_results['test_case_3'] = {
        "output_shape": o.shape,
        "final_state_shape": final_state.shape,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item()
    }

    # Test 4: Without initial state and with final state
    initial_state = None
    output_final_state = True
    o, final_state = fused_recurrent_retention(q, k, v, initial_state=initial_state, output_final_state=output_final_state)
    loss = o.sum() + final_state.sum()
    loss.backward()
    test_results['test_case_4'] = {
        "output_shape": o.shape,
        "final_state_shape": final_state.shape,
        "loss": loss.item(),
        "gradients_q": q.grad.norm().item(),
        "gradients_k": k.grad.norm().item(),
        "gradients_v": v.grad.norm().item()
    }

    return test_results

# Run the test function with backward propagation
result_gold = test_fused_recurrent_retention_with_backward()