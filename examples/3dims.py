import torch
import triton
import triton.language as tl


import triton_viz
from triton_viz.clients import Tracer

BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 8
BLOCK_SIZE_Z = 4


@triton_viz.trace(clients=Tracer())
@triton.jit
def add_3d_slices_kernel(
    input_ptr1,
    input_ptr2,
    output_ptr,
    stride_x,
    stride_y,
    stride_z,
    slice_x,
    slice_y,
    slice_z,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    # Compute the 3D position in the output tensor
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    # Compute the starting position for this block
    x_start = pid_x * BLOCK_SIZE_X
    y_start = pid_y * BLOCK_SIZE_Y
    z_start = pid_z * BLOCK_SIZE_Z

    # Compute offsets within the block
    x_offsets = x_start + tl.arange(0, BLOCK_SIZE_X)
    y_offsets = y_start + tl.arange(0, BLOCK_SIZE_Y)
    z_offsets = z_start + tl.arange(0, BLOCK_SIZE_Z)

    # Create a mask to handle boundary conditions
    mask = (
        (x_offsets < slice_x)
        & (y_offsets < slice_y)[:, None]
        & (z_offsets < slice_z)[:, None, None]
    )

    # Compute the input and output offsets
    offsets = (
        z_offsets[:, None, None] * stride_z
        + y_offsets[:, None] * stride_y
        + x_offsets * stride_x
    )

    # Load input slices
    slice1 = tl.load(input_ptr1 + offsets, mask=mask)
    slice2 = tl.load(input_ptr2 + offsets, mask=mask)

    # Perform addition
    result = slice1 + slice2

    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)


def add_3d_slices(input1, input2, output):
    # Get tensor shapes
    slice_z, slice_y, slice_x = input1.shape

    # Compute strides
    stride_z, stride_y, stride_x = input1.stride()

    # Determine grid size
    grid = (
        triton.cdiv(slice_x, BLOCK_SIZE_X),
        triton.cdiv(slice_y, BLOCK_SIZE_Y),
        triton.cdiv(slice_z, BLOCK_SIZE_Z),
    )

    # Launch kernel
    add_3d_slices_kernel[grid](
        input1,
        input2,
        output,
        stride_x,
        stride_y,
        stride_z,
        slice_x,
        slice_y,
        slice_z,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_Z=BLOCK_SIZE_Z,
    )


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Create example input tensor
    input1 = torch.randn(16, 16, 32, device="cpu")
    input2 = torch.randn(16, 16, 32, device="cpu")
    output = torch.empty_like(input1)

    # Call the kernel
    add_3d_slices(input1, input2, output)
    triton_viz.launch()

    # Verify the result
    expected_output = input1 + input2
    assert torch.allclose(
        output, expected_output
    ), "Kernel output does not match expected result"
