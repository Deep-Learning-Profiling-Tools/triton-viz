import torch
import triton
import triton.language as tl
import triton_viz
import argparse
from triton_viz.interpreter import record_builder
import numpy as np
from triton_viz.data import Load


@triton_viz.trace
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output, grid

    # Directly use x and y here even though they are defined later in the file


def perform_vec_add(device, size):
    torch.manual_seed(0)
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output, grid = add(x, y)  # Assuming add() is your custom function
    return x, y, output


def test_add():
    device = "cpu"
    size = 5000
    BLOCK_SIZE = 1024
    input_vector1, input_vector2, result = perform_vec_add(device, size)
    t_size = input_vector1.element_size()
    expected_offsets = [i * t_size for i in np.arange(0, BLOCK_SIZE)]
    expected_offsets_len = len(expected_offsets)
    expected = input_vector1 + input_vector2
    expected_masks = np.ones(expected_offsets_len, dtype=bool)
    expected_invalid_masks = np.logical_not(expected_masks)
    for op in record_builder.launches[0].records:
        if isinstance(op, Load):
            result_offsets = op.offsets.tolist()
            result_offsets_len = len(result_offsets)
            result_masks = op.access_masks
            result_invalid_masks = op.invalid_access_masks
            break
    assert torch.allclose(result, expected)
    assert result.shape == expected.shape
    assert result_offsets == expected_offsets
    assert result_offsets_len == expected_offsets_len
    assert (result_masks == expected_masks).all()
    assert (result_invalid_masks == expected_invalid_masks).all()


def test_out_of_bounds_add():
    device = "cpu"
    size = 960
    BLOCK_SIZE = 1024
    input_vector1, input_vector2, result = perform_vec_add(device, size)
    t_size = input_vector1.element_size()
    expected_offsets = [(i * t_size) if i < size else 0 for i in range(BLOCK_SIZE)]
    expected_offsets_len = len(expected_offsets)
    expected = input_vector1 + input_vector2
    expected_masks = [i < size for i in range(BLOCK_SIZE)]
    expected_invalid_masks = np.logical_not(expected_masks)
    for op in record_builder.launches[0].records:
        if isinstance(op, Load):
            result_offsets = op.offsets.tolist()
            result_offsets_len = len(result_offsets)
            result_masks = op.access_masks
            result_invalid_masks = op.invalid_access_masks
            break
    assert torch.allclose(result, expected)
    assert result.shape == expected.shape
    assert result_offsets == expected_offsets
    assert result_offsets_len == expected_offsets_len
    assert (result_masks == expected_masks).all()
    assert (result_invalid_masks == expected_invalid_masks).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = args.device

    size = 5000
    input_vector1, input_vector2, output_triton = perform_vec_add(device, size)
    triton_viz.launch()
