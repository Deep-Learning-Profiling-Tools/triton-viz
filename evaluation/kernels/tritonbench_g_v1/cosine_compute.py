import torch
import triton
import triton.language as tl
import math

# Kernel function: Computes the cosine of each element in the input tensor.
@triton.jit
def cos_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block and thread
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Determine which elements are valid within the block
    mask = offset < n_elements
    # Load elements from tensor `a` into `a_value`
    a_value = tl.load(a + offset, mask=mask)
    # Compute the cosine of each element in `a_value`
    b_value = tl.cos(a_value.to(tl.float32))
    # Store the result back to tensor `b`
    tl.store(b + offset, b_value, mask=mask)  

# Function to invoke the Triton kernel and perform the computation
def cos(A):
    # Prepare output tensor `B` with the same shape and type as `A`
    B = torch.empty_like(A)
    # Determine the total number of elements in the input tensor `A`
    n_elements = A.numel()
    # Calculate the optimal block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Determine the grid size needed to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    cos_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B




##################################################################################################################################################


def test_cos_function():
    # Create test cases with various input sizes
    test_cases = {
        'test_case_1': torch.rand(1024, device='cuda') * 2 * math.pi,
        'test_case_2': torch.rand(2048, device='cuda') * 2 * math.pi,
        'test_case_3': torch.rand(4096, device='cuda') * 2 * math.pi,
        'test_case_4': torch.rand(8192, device='cuda') * 2 * math.pi
    }
    
    results = {}
    
    for case_name, input_tensor in test_cases.items():
        # Compute cosine using Triton
        B_triton = cos(input_tensor)
        results[case_name] = B_triton
    
    return results

# Run the test
result_gold = test_cos_function()
