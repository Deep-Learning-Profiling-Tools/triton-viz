
import math
import torch
import triton
import triton.language as tl

# Triton kernel for rowwise dequantization
@triton.jit
def _dequantize_rowwise(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)
    max_val = tl.load(state_x + pid)
    output = max_val * x * inv_127
    tl.store(output_ptr + offsets, output, mask=row_mask)

# Function to call the Triton kernel
def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _dequantize_rowwise[grid](x, state_x, output, 1.0 / 127, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output




##################################################################################################################################################


# Test function for dequantize_rowwise
def test_dequantize_rowwise():
    results = {}

    # Test case 1: Simple case
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8, device='cuda')
    state_x = torch.tensor([4.0, 8.0], dtype=torch.float32, device='cuda')
    output = dequantize_rowwise(x, state_x)
    results['test_case_1'] = output

    # Test case 2: Larger input
    x = torch.randint(-128, 127, (10, 16), dtype=torch.int8, device='cuda')
    state_x = torch.rand(10, dtype=torch.float32, device='cuda') * 10
    output = dequantize_rowwise(x, state_x)
    results['test_case_2'] = output

    # Test case 3: Edge case with zeros
    x = torch.zeros((5, 8), dtype=torch.int8, device='cuda')
    state_x = torch.ones(5, dtype=torch.float32, device='cuda')
    output = dequantize_rowwise(x, state_x)
    results['test_case_3'] = output

    # Test case 4: Different block size
    x = torch.randint(-128, 127, (3, 32), dtype=torch.int8, device='cuda')
    state_x = torch.rand(3, dtype=torch.float32, device='cuda') * 10
    output = dequantize_rowwise(x, state_x)
    results['test_case_4'] = output

    return results

# Run the test function
result_gold = test_dequantize_rowwise()
