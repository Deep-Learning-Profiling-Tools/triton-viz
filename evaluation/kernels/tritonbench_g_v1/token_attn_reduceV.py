
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
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = 0
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    v_loc_off = cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_loc = tl.load(
            Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s,
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0.0,
        )
        v_value = tl.load(
            V + v_offs + v_loc[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0
        )
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(Out.dtype.element_ty)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_att_fwd2(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen):
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
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return



##################################################################################################################################################


import torch

# Define the test function for token_att_fwd2
def test_token_att_fwd2():
    torch.cuda.empty_cache()
    # Define input dimensions
    batch_size = 2
    num_heads = 4
    seq_len = 128
    d_model = 64

    # Create random input tensors
    prob = torch.rand((num_heads, seq_len), dtype=torch.float32, device='cuda')
    v = torch.rand((num_heads, seq_len, d_model), dtype=torch.float32, device='cuda')
    out = torch.zeros((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')
    Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int32, device='cuda')
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device='cuda')
    B_Start_Loc = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
    B_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32, device='cuda')

    # Call the function
    token_att_fwd2(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen)
    torch.cuda.synchronize()
    result = {"test_case_1": out.clone()}

    # Additional test cases to cover more branches
    # Test case 2: Different sequence length
    seq_len_2 = 64
    prob_2 = torch.rand((num_heads, seq_len_2), dtype=torch.float32, device='cuda')
    v_2 = torch.rand((num_heads, seq_len_2, d_model), dtype=torch.float32, device='cuda')
    out_2 = torch.zeros((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')
    Req_to_tokens_2 = torch.randint(0, seq_len_2, (batch_size, seq_len_2), dtype=torch.int32, device='cuda')
    B_Seqlen_2 = torch.full((batch_size,), seq_len_2, dtype=torch.int32, device='cuda')

    token_att_fwd2(prob_2, v_2, out_2, Req_to_tokens_2, B_req_idx, B_Start_Loc, B_Seqlen_2)
    torch.cuda.synchronize()
    result["test_case_2"] = out_2.clone()

    # Test case 3: Different batch size
    batch_size_3 = 3
    prob_3 = torch.rand((num_heads, seq_len), dtype=torch.float32, device='cuda')
    v_3 = torch.rand((num_heads, seq_len, d_model), dtype=torch.float32, device='cuda')
    out_3 = torch.zeros((batch_size_3, num_heads, d_model), dtype=torch.float32, device='cuda')
    Req_to_tokens_3 = torch.randint(0, seq_len, (batch_size_3, seq_len), dtype=torch.int32, device='cuda')
    B_req_idx_3 = torch.arange(batch_size_3, dtype=torch.int32, device='cuda')
    B_Start_Loc_3 = torch.zeros(batch_size_3, dtype=torch.int32, device='cuda')
    B_Seqlen_3 = torch.full((batch_size_3,), seq_len, dtype=torch.int32, device='cuda')

    token_att_fwd2(prob_3, v_3, out_3, Req_to_tokens_3, B_req_idx_3, B_Start_Loc_3, B_Seqlen_3)
    torch.cuda.synchronize()
    result["test_case_3"] = out_3.clone()
    torch.cuda.empty_cache()

    return result

# Run the tests
result_gold = test_token_att_fwd2()
