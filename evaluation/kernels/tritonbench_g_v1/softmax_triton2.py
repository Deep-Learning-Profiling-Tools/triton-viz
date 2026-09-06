
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y




##################################################################################################################################################


import torch

# Test cases for the softmax function
def test_softmax():
    result_dict = {}

    # Test case 1: Small matrix
    x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device='cuda')
    y1 = softmax(x1)
    result_dict["test_case_1"] = y1

    # Test case 2: Larger matrix
    x2 = torch.randn(128, 256, dtype=torch.float32, device='cuda')
    y2 = softmax(x2)
    result_dict["test_case_2"] = y2

    # Test case 3: Single row
    x3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device='cuda')
    y3 = softmax(x3)
    result_dict["test_case_3"] = y3

    # Test case 4: Single column
    x4 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device='cuda')
    y4 = softmax(x4)
    result_dict["test_case_4"] = y4

    # Test case 5: Large matrix with power of two columns
    x5 = torch.randn(64, 512, dtype=torch.float32, device='cuda')
    y5 = softmax(x5)
    result_dict["test_case_5"] = y5

    return result_dict

# Run the test cases
result_gold = test_softmax()
