
import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length)

def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)

    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE)




##################################################################################################################################################


import torch

def test_launch_var_len_copy_kernel_triton():
    # Define test input data
    num_arrays = 3
    BLOCK_SIZE = 256

    # Old array start indices
    old_a_start = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')

    # Lengths of each array
    old_a_len = torch.tensor([50, 150, 200], dtype=torch.int32, device='cuda')

    # Flattened old array data
    old_a_location = torch.arange(500, dtype=torch.float32, device='cuda')

    # New array start indices
    new_a_start = torch.tensor([0, 60, 260], dtype=torch.int32, device='cuda')

    # Target flattened array for copying
    new_a_location = torch.zeros(500, dtype=torch.float32, device='cuda')

    # Launch the Triton kernel
    launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location)

    # Store results in a dictionary
    results = {}
    for i in range(num_arrays):
        old_start = old_a_start[i].item()
        new_start = new_a_start[i].item()
        length = old_a_len[i].item()
        # Store the result of each test case
        results[f"test_case_{i+1}"] = torch.equal(
            old_a_location[old_start:old_start + length],
            new_a_location[new_start:new_start + length]
        )
    
    return results

result_gold = test_launch_var_len_copy_kernel_triton()
