
import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel_token_att2(
    Prob,
    V,
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    B_Att_Start_Loc,
    B_Att_Seqlen,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_ph,
    stride_pbs,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    sliding_window,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Triton kernel for computing token attention
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = tl.maximum(cur_batch_seq_len - sliding_window, 0)
    cur_batch_in_all_start_index = tl.load(B_Att_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_att_seq_len = tl.load(B_Att_Seqlen + cur_batch)

    v_loc_off = (
        cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    )
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_att_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n, mask=(start_n + offs_n) < cur_att_seq_len, other=0.0)
        v_loc = tl.load(
            Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s,
            mask=(start_n + offs_n + cur_batch_start_index) < cur_batch_seq_len,
            other=0.0,
        )
        v_value = tl.load(
            V + v_offs + v_loc[:, None] * stride_vbs,
            mask=(start_n + offs_n[:, None] + cur_batch_start_index) < cur_batch_seq_len,
            other=0.0,
        )
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(Out.dtype.element_ty)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_att_fwd2(
    prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window
):
    # Launch the Triton kernel for token attention
    BLOCK = 128
    batch, head = B_req_idx.shape[0], prob.shape[0]
    grid = (batch, head)
    num_warps = 4
    dim = v.shape[-1]

    kv_group_num = prob.shape[0] // v.shape[1]

    _fwd_kernel_token_att2[grid](
        prob,
        v,
        out,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        B_Att_Start_Loc,
        B_Att_Seqlen,
        Req_to_tokens.stride(0),
        Req_to_tokens.stride(1),
        prob.stride(0),
        prob.stride(1),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        kv_group_num=kv_group_num,
        sliding_window=sliding_window,
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return




##################################################################################################################################################


# Define the test function
def test_token_att_fwd2():
    # Define the dimensions
    batch_size = 2
    num_heads = 4
    seq_len = 128
    d_model = 64
    sliding_window = 64

    # Create random tensors for inputs
    prob = torch.rand((num_heads, seq_len), dtype=torch.float32, device='cuda')
    v = torch.rand((num_heads, seq_len, d_model), dtype=torch.float32, device='cuda')
    Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int32, device='cuda')
    B_req_idx = torch.randint(0, batch_size, (batch_size,), dtype=torch.int32, device='cuda')
    B_Start_Loc = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
    B_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32, device='cuda')
    B_Att_Start_Loc = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
    B_Att_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32, device='cuda')

    results = {}

    # Test case 1
    out1 = torch.zeros((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')
    token_att_fwd2(
        prob, v, out1, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window
    )
    results['test_case_1'] = out1.clone()

    # Test case 2 (different sliding_window size)
    sliding_window = 32
    out2 = torch.zeros((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')
    token_att_fwd2(
        prob, v, out2, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window
    )
    results['test_case_2'] = out2.clone()

    # Test case 3 (different sequence length for Req_to_tokens)
    Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len // 2), dtype=torch.int32, device='cuda')
    out3 = torch.zeros((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')
    token_att_fwd2(
        prob, v, out3, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window
    )
    results['test_case_3'] = out3.clone()

    # Test case 4 (different batch size)
    batch_size = 4
    prob = torch.rand((num_heads, seq_len), dtype=torch.float32, device='cuda')
    v = torch.rand((num_heads, seq_len, d_model), dtype=torch.float32, device='cuda')
    Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int32, device='cuda')
    B_req_idx = torch.randint(0, batch_size, (batch_size,), dtype=torch.int32, device='cuda')
    B_Start_Loc = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
    B_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32, device='cuda')
    B_Att_Start_Loc = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
    B_Att_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32, device='cuda')

    out4 = torch.zeros((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')
    token_att_fwd2(
        prob, v, out4, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window
    )
    results['test_case_4'] = out4.clone()

    return results


# Execute the test function
result_gold = test_token_att_fwd2()
