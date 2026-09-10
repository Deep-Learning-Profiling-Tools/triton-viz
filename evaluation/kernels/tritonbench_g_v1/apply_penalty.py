
import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits, presence_penalty, freqency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len, 
    stride_logit_b, stride_logit_s,
    BLOCK_P: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)

    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0)
    
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset<cur_batch_end_index)

    return

@torch.no_grad()
def apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
    assert Logits.is_contiguous()
    BLOCK = triton.next_power_of_2(p_max_len_in_batch)
    if BLOCK <= 512:
        BLOCK = 512
    elif BLOCK <= 1024:
        BLOCK = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0], )](
        Logits, presence_penalty, freqency_penalty, repetition_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK
    )
    return




##################################################################################################################################################


import torch

# Define the test function
def test_apply_penalty():
    # Define the dimensions
    results = {}
    batch_size = 2
    seq_len = 10
    vocab_size = 50

    # Create random logits tensor
    Logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device='cuda').contiguous()

    # Define penalties
    presence_penalty = torch.tensor([0.1, 0.2], dtype=torch.float32, device='cuda')
    freqency_penalty = torch.tensor([0.1, 0.2], dtype=torch.float32, device='cuda')
    repetition_penalty = torch.tensor([1.2, 1.3], dtype=torch.float32, device='cuda')

    # Define token ids and counts
    p_token_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int32, device='cuda')
    p_token_counts = torch.randint(1, 5, (seq_len,), dtype=torch.int32, device='cuda')

    # Define cumulative sequence lengths
    p_cumsum_seq_len = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')

    # Maximum length in batch
    p_max_len_in_batch = seq_len

    # Call the apply_penalty function for the first branch
    apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch)
    results['test_case_1'] = Logits.clone()

    # Modify p_max_len_in_batch to test another branch
    p_max_len_in_batch = 600  # Testing BLOCK set to 512
    apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch)
    results['test_case_2'] = Logits.clone()

    # Modify p_max_len_in_batch to test another branch
    p_max_len_in_batch = 800  # Testing BLOCK set to 1024
    apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch)
    results['test_case_3'] = Logits.clone()

    return results

# Run the test and capture results
result_gold = test_apply_penalty()
