
import torch
import triton
import triton.language as tl

# Triton kernel for forward token softmax
@triton.jit
def _fwd_kernel_token_softmax(
    Logics, B_Start_Loc, B_Seqlen,
    Prob_Out,
    stride_logic_h, stride_logic_bs,
    stride_prob_h, stride_prob_bs,
    BLOCK_SIZE: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    row = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs,
                  mask=col_offsets < cur_batch_seq_len, other=-float('inf')).to(tl.float32)

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(Prob_Out + cur_head * stride_prob_h + (cur_batch_in_all_start_index + col_offsets)
             * stride_prob_bs, softmax_output, mask=col_offsets < cur_batch_seq_len)
    return

# Function to launch the Triton kernel
@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    batch, head_num = B_Start_Loc.shape[0], Logics.shape[0]

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _fwd_kernel_token_softmax[(batch, head_num)](
        Logics, B_Start_Loc, B_Seqlen,
        Prob_Out,
        Logics.stride(0), Logics.stride(1),
        Prob_Out.stride(0), Prob_Out.stride(1),
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return




##################################################################################################################################################


import torch

# Define the test function
def test_token_softmax_fwd():
    results = {}

    # Test case 1: Small input size
    batch_size = 2
    head_num = 2
    max_input_len = 8

    # Create random input tensors
    Logics = torch.randn((head_num, batch_size * max_input_len), dtype=torch.float32, device='cuda')
    B_Start_Loc = torch.tensor([0, max_input_len], dtype=torch.int32, device='cuda')
    B_Seqlen = torch.tensor([max_input_len, max_input_len], dtype=torch.int32, device='cuda')
    Prob_Out = torch.empty_like(Logics)

    # Call the Triton softmax function
    token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len)

    # Store the output
    results['test_case_1'] = Prob_Out.clone()

    # Test case 2: Larger input size
    batch_size = 1
    head_num = 1
    max_input_len = 16

    # Create random input tensors
    Logics = torch.randn((head_num, batch_size * max_input_len), dtype=torch.float32, device='cuda')
    B_Start_Loc = torch.tensor([0], dtype=torch.int32, device='cuda')
    B_Seqlen = torch.tensor([max_input_len], dtype=torch.int32, device='cuda')
    Prob_Out = torch.empty_like(Logics)

    # Call the Triton softmax function
    token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len)

    # Store the output
    results['test_case_2'] = Prob_Out.clone()

    return results

# Run the test function
result_gold = test_token_softmax_fwd()
