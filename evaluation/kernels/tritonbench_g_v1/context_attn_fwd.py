import torch

import triton
import triton.language as tl

TESLA = "Tesla" in torch.cuda.get_device_name(0)

@triton.jit
def _fwd_kernel_int8kv(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    b_prompt_cache_len,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # k = tl.load(
        #     k_ptrs + (start_n + offs_n[None, :]) * stride_ks,
        #     mask=(start_n + offs_n[None, :]) < block_end_loc,
        #     other=0,
        # )
        off_k = (
            cur_batch * stride_kb
            + (start_n + offs_n[None, :]) * stride_ks
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)

        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        # v = tl.load(
        #     v_ptrs + (start_n + offs_n[:, None]) * stride_vs,
        #     mask=(start_n + offs_n[:, None]) < block_end_loc,
        #     other=0.0,
        # )
        off_v = (
            cur_batch * stride_vb
            + (start_n + offs_n[:, None]) * stride_vs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len):
    BLOCK_M = 128 if not TESLA else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    # 计算scale系数, 并乘以 1/log(2) = 1.4426950408889634,
    # 算子内部使用 tl.math.exp2 来使计算与标准attention等价。
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)
    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel_int8kv[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_group_num=kv_group_num,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )




##################################################################################################################################################


import torch

def test_context_attention_fwd_ppl_int8kv():
    Z, H, N_CTX, D_HEAD = 16, 16, 2048, 128
    dtype = torch.float16
    prompt_cache_len = 0
    q = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    kv = torch.empty((Z, 2 * H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    k = kv[:, :H]
    v = kv[:, H:]
    o = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    max_input_len = N_CTX
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")

    results = {}

    # Test case 1
    context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len)
    results['test_case_1'] = o.clone()

    # Test case 2: Different prompt_cache_len
    prompt_cache_len = 10
    q = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    b_prompt_cache_len = torch.full((Z,), prompt_cache_len, dtype=torch.int32, device="cuda")
    context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len)
    results['test_case_2'] = o.clone()

    # Test case 3: Different max_input_len
    max_input_len = N_CTX // 2
    context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len)
    results['test_case_3'] = o.clone()

    # Test case 4: Different batch size
    Z = 8
    q = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    kv = torch.empty((Z, 2 * H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    k = kv[:, :H]
    v = kv[:, H:]
    o = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")
    context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len)
    results['test_case_4'] = o.clone()

    return results

result_gold = test_context_attention_fwd_ppl_int8kv()
