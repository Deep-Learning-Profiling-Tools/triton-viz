
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O, block_seq):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)
    
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, mid_out, mid_out_logexpsum, O,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )
    return




##################################################################################################################################################


import torch

# Define the test function
def test_flash_decode_stage2():
    # Define the parameters
    batch_size = 2
    head_num = 4
    seq_block_num = 3
    head_dim = 32  # This should be one of {16, 32, 64, 128}
    block_seq = 8

    results = {}

    # Create input tensors for test case 1
    B_Seqlen_1 = torch.tensor([24, 16], dtype=torch.int32, device='cuda')
    mid_out_1 = torch.randn(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32, device='cuda')
    mid_out_logexpsum_1 = torch.randn(batch_size, head_num, seq_block_num, dtype=torch.float32, device='cuda')
    O_1 = torch.empty(batch_size, head_num, head_dim, dtype=torch.float32, device='cuda')
    # Call the wrapper function
    flash_decode_stage2(mid_out_1, mid_out_logexpsum_1, B_Seqlen_1, O_1, block_seq)
    results['test_case_1'] = O_1.clone().cpu()

    # Create input tensors for test case 2
    B_Seqlen_2 = torch.tensor([0, 0], dtype=torch.int32, device='cuda')  # Edge case: zero sequence lengths
    mid_out_2 = torch.randn(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32, device='cuda')
    mid_out_logexpsum_2 = torch.randn(batch_size, head_num, seq_block_num, dtype=torch.float32, device='cuda')
    O_2 = torch.empty(batch_size, head_num, head_dim, dtype=torch.float32, device='cuda')
    # Call the wrapper function
    flash_decode_stage2(mid_out_2, mid_out_logexpsum_2, B_Seqlen_2, O_2, block_seq)
    results['test_case_2'] = O_2.clone().cpu()

    # Create input tensors for test case 3
    B_Seqlen_3 = torch.tensor([8, 8], dtype=torch.int32, device='cuda')  # Edge case: minimum non-zero sequence lengths
    mid_out_3 = torch.randn(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32, device='cuda')
    mid_out_logexpsum_3 = torch.randn(batch_size, head_num, seq_block_num, dtype=torch.float32, device='cuda')
    O_3 = torch.empty(batch_size, head_num, head_dim, dtype=torch.float32, device='cuda')
    # Call the wrapper function
    flash_decode_stage2(mid_out_3, mid_out_logexpsum_3, B_Seqlen_3, O_3, block_seq)
    results['test_case_3'] = O_3.clone().cpu()

    # Create input tensors for test case 4
    B_Seqlen_4 = torch.tensor([32, 64], dtype=torch.int32, device='cuda')  # Larger sequence lengths
    mid_out_4 = torch.randn(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32, device='cuda')
    mid_out_logexpsum_4 = torch.randn(batch_size, head_num, seq_block_num, dtype=torch.float32, device='cuda')
    O_4 = torch.empty(batch_size, head_num, head_dim, dtype=torch.float32, device='cuda')
    # Call the wrapper function
    flash_decode_stage2(mid_out_4, mid_out_logexpsum_4, B_Seqlen_4, O_4, block_seq)
    results['test_case_4'] = O_4.clone().cpu()

    return results

# Execute the test function
result_gold = test_flash_decode_stage2()
