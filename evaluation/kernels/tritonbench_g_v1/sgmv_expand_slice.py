
import torch
import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    b_seq_start_loc,
    seq_lens,
    lora_indices,
    xm_stride,
    xk_stride,  # 1
    l0_stride,  # hidden_size*max_rank
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    slice_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M > M:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride +
             offset_k[None, :] * xk_stride, )
    b_ptr = (lora_ptr + l0_stride * lora_index +
             offset_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :] < K - k * BLOCK_K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None] < K - k * BLOCK_K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * xk_stride
        b_ptr += BLOCK_K * lora_n_stride
    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_offset
    c_ptr = (out_ptr + offset_cm[:, None] * cm_stride +
             offset_cn[None, :] * cn_stride)
    M = tl.load(seq_lens + cur_batch)
    c_mask = (offset_cm[:, None] < (cur_seq_start + M)) & (offset_cn[None, :] <
                                                           (slice_offset + N))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@torch.inference_mode()
def _sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
) -> None:

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(0) == token_nums
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert slice_size == lora_b_weights.size(-2)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)

    assert lora_b_weights.is_contiguous()

    N, K = lora_b_weights.shape[-2:]

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = (
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        batches,
    )
    _sgmv_expand_slice_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        slice_offset,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
    )
    return




##################################################################################################################################################


import torch

# Define the test function
def test_sgmv_expand_slice():
    # Test parameters
    batches = 2
    max_seq_length = 64
    token_nums = 128
    slice_size = 32
    rank = 32

    # Create input tensors
    inputs = torch.randn(token_nums, slice_size, dtype=torch.float16, device='cuda').contiguous()
    lora_b_weights = torch.randn(1, rank, slice_size, dtype=torch.float16, device='cuda').contiguous()
    output_tensor = torch.zeros(token_nums, slice_size, dtype=torch.float16, device='cuda').contiguous()
    b_seq_start_loc = torch.tensor([0, 64], dtype=torch.int32, device='cuda')
    seq_len_tensor = torch.tensor([64, 64], dtype=torch.int32, device='cuda')
    lora_indices_tensor = torch.tensor([0, 0], dtype=torch.int32, device='cuda')

    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: add_inputs is False
    _sgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor.clone(),
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        0,  # slice_offset
        slice_size,
        False  # add_inputs
    )
    results["test_case_1"] = output_tensor.clone()

    # Test case 2: add_inputs is True
    _sgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor.clone(),
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        0,  # slice_offset
        slice_size,
        True  # add_inputs
    )
    results["test_case_2"] = output_tensor.clone()

    # Test case 3: Different slice_offset
    _sgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor.clone(),
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        16,  # slice_offset
        slice_size,
        False  # add_inputs
    )
    results["test_case_3"] = output_tensor.clone()

    # Test case 4: Different slice size
    slice_size = 16
    rank = 16
    inputs = torch.randn(token_nums, slice_size, dtype=torch.float16, device='cuda').contiguous()
    lora_b_weights = torch.randn(1, rank, slice_size, dtype=torch.float16, device='cuda').contiguous()
    output_tensor = torch.zeros(token_nums, slice_size, dtype=torch.float16, device='cuda').contiguous()
    
    _sgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor.clone(),
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        0,  # slice_offset
        slice_size,
        False  # add_inputs
    )
    results["test_case_4"] = output_tensor.clone()

    return results

# Run the test
result_gold = test_sgmv_expand_slice()