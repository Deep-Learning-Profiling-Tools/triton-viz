
import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b_loc_b, stride_b_loc_s,
    other_kv_index, # Avoid reading NaN data
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=other_kv_index)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs, 
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
    
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)
    num_warps = 1
    _fwd_kernel[grid](
        logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len,
        logics.stride(0), logics.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        b_loc.stride(0), b_loc.stride(1),
        other_kv_index,
        BLOCK_DMODEL=v.shape[-1],
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3
    )
    return




##################################################################################################################################################


import torch

def test_token_softmax_reducev_fwd():
    # Define the input parameters
    batch_size = 2
    num_heads = 2
    max_input_len = 128
    d_model = 64
    BLOCK = 64

    # Create random tensors for inputs
    logics = torch.randn((num_heads, batch_size * max_input_len), dtype=torch.float32, device='cuda')
    v = torch.randn((num_heads, max_input_len, d_model), dtype=torch.float32, device='cuda')
    o = torch.empty((batch_size, num_heads, d_model), dtype=torch.float32, device='cuda')

    # Create auxiliary tensors
    b_loc = torch.randint(0, max_input_len, (batch_size, max_input_len), dtype=torch.int32, device='cuda')
    b_start_loc = torch.randint(0, max_input_len, (batch_size,), dtype=torch.int32, device='cuda')
    b_seq_len = torch.randint(1, max_input_len + 1, (batch_size,), dtype=torch.int32, device='cuda')
    other_kv_index = -1  # Assuming -1 is used to avoid reading NaN data

    # First branch execution
    token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
    result_1 = o.clone()

    # Modify inputs to test second branch
    b_seq_len = torch.tensor([max_input_len, max_input_len], dtype=torch.int32, device='cuda')
    token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
    result_2 = o.clone()

    # Modify inputs to test third branch
    b_start_loc = torch.tensor([0, 0], dtype=torch.int32, device='cuda')
    token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
    result_3 = o.clone()

    # Modify inputs to test fourth branch
    other_kv_index = 0
    token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
    result_4 = o.clone()

    results = {
        "test_case_1": result_1,
        "test_case_2": result_2,
        "test_case_3": result_3,
        "test_case_4": result_4
    }

    return results

result_gold = test_token_softmax_reducev_fwd()
