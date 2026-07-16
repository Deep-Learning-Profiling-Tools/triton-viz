
import torch
import triton
import triton.language as tl
# from .utils import get_lora_op_configs

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    scaling,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_sk = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return

    offset_n = tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K) + pid_sk * BLOCK_K
    a_ptr = input_ptr + cur_batch * xm_stride
    b_ptr = lora_ptr + l0_stride * lora_index
    accumulator = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for k in range(0, K, BLOCK_K * SPLIT_K):
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, BLOCK_K)
        tiled_a = tl.load(
            a_ptr + current_k_c,
            mask=current_k < K,
            other=0.0,
        )
        b_ptr_mask = (offset_n[:, None] < N) & (current_k[None, :] < K)

        tiled_b = tl.load(
            b_ptr + offset_n[:, None] * lora_k_stride +
            current_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )

        accumulator += tl.sum(tiled_a * tiled_b, 1)
    accumulator *= scaling
    offset_cn = tl.arange(0, BLOCK_N)
    c_ptr = out_ptr + cur_batch * cm_stride + offset_cn * cn_stride
    c_mask = offset_cn < N
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@torch.inference_mode()
def _bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    assert inputs.dtype == lora_a_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert lora_a_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_a_weights.size(-1)
    assert inputs.is_contiguous()

    if lora_a_weights.ndim == 4:
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(dim=1)
    else:
        assert lora_a_weights.ndim == 3
    assert lora_a_weights.is_contiguous()
    assert output_tensor.is_contiguous()

    batches = lora_indices_tensor.size(0)
    N, K = lora_a_weights.shape[-2:]
    BLOCK_N = triton.next_power_of_2(N)
    # config = get_lora_op_configs("bgmv_shrink", batches, K)

    grid = lambda META: (
        META["SPLIT_K"],
        batches,
    )
    _bgmv_shrink_kernel[grid](
        inputs,
        lora_a_weights,
        output_tensor,
        N,
        K,
        lora_indices_tensor,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_a_weights.stride(0),
        lora_a_weights.stride(1),
        lora_a_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=256,
        SPLIT_K=64,

        # **config,
    )
    return




##################################################################################################################################################


import torch

# Test function for _bgmv_shrink
def test_bgmv_shrink():
    # Define input parameters
    batch_size = 2
    N = 16
    K = 32
    scaling = 1.0

    # Create input tensors
    inputs = torch.randn((batch_size, K), dtype=torch.float16, device='cuda').contiguous()
    lora_a_weights = torch.randn((batch_size, 1, N, K), dtype=torch.float16, device='cuda').contiguous()
    output_tensor = torch.zeros((batch_size, N), dtype=torch.float16, device='cuda').contiguous()
    lora_indices_tensor = torch.tensor([0, 1], dtype=torch.int32, device='cuda')

    # Call the _bgmv_shrink function
    _bgmv_shrink(
        inputs=inputs,
        lora_a_weights=lora_a_weights,
        output_tensor=output_tensor,
        lora_indices_tensor=lora_indices_tensor,
        scaling=scaling
    )

    # Store the result in a dictionary
    results = {
        "test_case_1": output_tensor.clone()
    }

    # Additional test cases to cover more branches
    lora_indices_tensor = torch.tensor([-1, 1], dtype=torch.int32, device='cuda')
    _bgmv_shrink(
        inputs=inputs,
        lora_a_weights=lora_a_weights,
        output_tensor=output_tensor,
        lora_indices_tensor=lora_indices_tensor,
        scaling=scaling
    )
    results["test_case_2"] = output_tensor.clone()

    lora_indices_tensor = torch.tensor([0, -1], dtype=torch.int32, device='cuda')
    _bgmv_shrink(
        inputs=inputs,
        lora_a_weights=lora_a_weights,
        output_tensor=output_tensor,
        lora_indices_tensor=lora_indices_tensor,
        scaling=scaling
    )
    results["test_case_3"] = output_tensor.clone()

    lora_indices_tensor = torch.tensor([-1, -1], dtype=torch.int32, device='cuda')
    _bgmv_shrink(
        inputs=inputs,
        lora_a_weights=lora_a_weights,
        output_tensor=output_tensor,
        lora_indices_tensor=lora_indices_tensor,
        scaling=scaling
    )
    results["test_case_4"] = output_tensor.clone()

    return results

# Run the test
result_gold = test_bgmv_shrink()
