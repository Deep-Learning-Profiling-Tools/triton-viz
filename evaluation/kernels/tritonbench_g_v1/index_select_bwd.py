
import torch
import triton
import triton.language as tl


@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr,  # *Pointer* to grad_source tensor.
    index_ptr,  # *Pointer* to index tensor.
    grad_output_ptr,  # *Pointer* to grad_output tensor.
    num_rows,
    num_indices,
    num_cols,
    stride0,  # Stride information of input and source tensor.
    stride1,
    BLOCK_SIZE_INDEX: tl.constexpr,  # Number of indices each program should process.
    BLOCK_SIZE_COL: tl.constexpr,  # Number of cols each program should process.
):
    pid0 = tl.program_id(axis=0)  # We use 3D launch grid
    pid1 = tl.program_id(axis=1)

    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # load grad_output
    grad_output_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    grad_output_offsets = (
        grad_output_ptr
        + grad_output_indices[:, None] * stride0
        + cols[None, :] * stride1
    )
    grad_output_mask = (grad_output_indices[:, None] < num_indices) & (
        cols[None, :] < num_cols
    )
    grad_output = tl.load(grad_output_offsets, mask=grad_output_mask).to(tl.float32)

    # select indices from grad_source
    grad_source_indices = tl.load(
        index_ptr + grad_output_indices, mask=(grad_output_indices < num_indices)
    )
    grad_source_offsets = (
        grad_source_ptr
        + grad_source_indices[:, None] * stride0
        + cols[None, :] * stride1
    )

    # compute scaled index add and save
    tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)


def index_select_cat_bwd(
    grad_source: torch.Tensor,
    index: torch.Tensor,
    grad_output: torch.Tensor,
):
    if not (grad_source.is_cuda and grad_output.is_cuda):
        raise ValueError("The grad_source and grad_output tensor must be of type CUDA!")

    if not (grad_source.ndim == 2 and grad_output.ndim == 2):
        raise ValueError(
            f"The grad_source and grad_output must be three-dimensional "
            f"(got {grad_source.ndim} and {grad_output.ndim})!"
        )
    if not grad_source.shape[1] == grad_output.shape[1]:
        raise ValueError(
            f"The number of elements along dimension 1 of grad_source and grad_output must be the same "
            f"(got {grad_source.shape[1]} and {grad_output.shape[1]})"
        )

    num_rows, num_cols = grad_source.shape
    num_indices, num_cols = grad_output.shape
    if not num_rows >= num_indices:
        raise ValueError(
            f"The number of elements along dimension 0 of grad_source must be larger than that of grad_output "
            f"(got {num_rows} and {num_indices})!"
        )
    if not index.shape[0] == num_indices:
        raise ValueError(
            f"The number of indices and the number of elements along dimension 0 of grad_output must match "
            f"(got {index.shape[0]} and {num_indices})!"
        )

    stride0, stride1 = grad_source.stride(0), grad_source.stride(1)
    if not (grad_output.stride(0) == stride0 and grad_output.stride(1) == stride1):
        raise ValueError(
            f"The strides of the grad_source and grad_output tensors must match "
            f"(got {stride0} vs. {grad_output.stride(0)}, {stride1} vs. {grad_output.stride(1)})!"
        )

    def grid(meta):
        return (
            triton.cdiv(num_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    index_select_cat_bwd_kernel[grid](
        grad_source,
        index,
        grad_output,
        num_rows,
        num_indices,
        num_cols,
        grad_source.stride(0),
        grad_source.stride(1),
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_COL=512,
    )

    return




##################################################################################################################################################


import torch

# Test for index_select_cat_bwd
def test_index_select_cat_bwd():
    results = {}

    # Test case 1: Basic test
    grad_source = torch.zeros(10, 512, device='cuda')
    index = torch.tensor([0, 2, 4, 6, 8], device='cuda')
    grad_output = torch.randn(len(index), grad_source.size(1), device='cuda')
    index_select_cat_bwd(grad_source, index, grad_output)
    results['test_case_1'] = grad_source.clone()

    # Test case 2: Different indices
    grad_source = torch.zeros(10, 512, device='cuda')
    index = torch.tensor([1, 3, 5, 7, 9], device='cuda')
    grad_output = torch.randn(len(index), grad_source.size(1), device='cuda')
    index_select_cat_bwd(grad_source, index, grad_output)
    results['test_case_2'] = grad_source.clone()

    # Test case 3: All indices the same
    grad_source = torch.zeros(10, 512, device='cuda')
    index = torch.tensor([0, 0, 0, 0, 0], device='cuda')
    grad_output = torch.randn(len(index), grad_source.size(1), device='cuda')
    index_select_cat_bwd(grad_source, index, grad_output)
    results['test_case_3'] = grad_source.clone()

    # Test case 4: Maximum index
    grad_source = torch.zeros(10, 512, device='cuda')
    index = torch.tensor([9, 9, 9, 9, 9], device='cuda')
    grad_output = torch.randn(len(index), grad_source.size(1), device='cuda')
    index_select_cat_bwd(grad_source, index, grad_output)
    results['test_case_4'] = grad_source.clone()

    return results

# Run the tests
result_gold = test_index_select_cat_bwd()
