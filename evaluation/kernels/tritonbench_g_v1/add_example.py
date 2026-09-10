
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add_wrapper(x, y):
    out = torch.zeros_like(x)
    
    BLOCK_SIZE = 4
    n_elements = x.numel()

    # Calculate the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)

    return out




##################################################################################################################################################


# Test the kernel with appropriate inputs
def test_add_kernel():
    results = {}
    
    # Test case 1
    x1 = torch.randn(16, device='cuda')
    y1 = torch.randn(16, device='cuda')
    out1 = add_wrapper(x1, y1)
    results['test_case_1'] = out1

    # Test case 2: Different size
    x2 = torch.randn(8, device='cuda')
    y2 = torch.randn(8, device='cuda')
    out2 = add_wrapper(x2, y2)
    results['test_case_2'] = out2

    # Test case 3: Larger size
    x3 = torch.randn(32, device='cuda')
    y3 = torch.randn(32, device='cuda')
    out3 = add_wrapper(x3, y3)
    results['test_case_3'] = out3

    # Test case 4: Edge case with zero elements
    x4 = torch.randn(0, device='cuda')
    y4 = torch.randn(0, device='cuda')
    out4 = add_wrapper(x4, y4)
    results['test_case_4'] = out4

    return results

# Run the test
result_gold = test_add_kernel()
