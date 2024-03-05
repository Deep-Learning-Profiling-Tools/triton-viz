import torch
import numpy as np
import triton
import triton.language as tl
import triton_viz
import argparse
from triton_viz.interpreter import record_builder
from triton_viz.data import Dot


@triton_viz.trace
@triton.jit
def dot_kernel(x_ptr, y_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0) * BLOCK_SIZE
    c = tl.program_id(1) * BLOCK_SIZE
    b = tl.program_id(2)
    bid = b * 4 * BLOCK_SIZE * BLOCK_SIZE
    x_val = tl.load(
        x_ptr
        + bid
        + (r + tl.arange(0, BLOCK_SIZE)[:, None]) * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    y_val = tl.load(
        y_ptr
        + bid
        + tl.arange(0, BLOCK_SIZE)[:, None] * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + c
    )
    z = tl.dot(x_val, y_val)
    x_val = tl.load(
        x_ptr
        + bid
        + (r + tl.arange(0, BLOCK_SIZE)[:, None]) * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + BLOCK_SIZE
    )
    y_val = tl.load(
        y_ptr
        + bid
        + (BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]) * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + c
    )
    z = z + tl.dot(x_val, y_val)
    tl.store(
        z_ptr
        + (b * (2 * BLOCK_SIZE) * (2 * BLOCK_SIZE - 10))
        + (r + tl.arange(0, BLOCK_SIZE)[:, None]) * (2 * BLOCK_SIZE - 10)
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + c,
        z,
        mask=tl.arange(0, BLOCK_SIZE)[None, :] + c < 2 * BLOCK_SIZE - 10,
    )


def perform_dot(device, BLOCK_SIZE):
    x = torch.randn((2 * BLOCK_SIZE, 2 * BLOCK_SIZE), device=device)
    y = torch.randn((2 * BLOCK_SIZE, 2 * BLOCK_SIZE), device=device)
    z = torch.zeros((2 * BLOCK_SIZE, 2 * BLOCK_SIZE - 10), device=device)
    dot_kernel[(2, 2)](x, y, z, BLOCK_SIZE)
    return x, y, z


def test_dot():
    BLOCK_SIZE = 32
    device = "cpu"
    input_matrix1, input_matrix2, result = perform_dot(device, BLOCK_SIZE)
    initial_expected = torch.from_numpy(np.dot(input_matrix1, input_matrix2))
    expected_output = initial_expected[:, : 2 * BLOCK_SIZE - 10]
    for op in record_builder.launches[0].records:
        if isinstance(op, Dot):
            result_input_shape = op.input_shape
            result_output_shape = op.output_shape
            result_other_shape = op.other_shape
    assert torch.allclose(result, expected_output, atol=1e-5, rtol=1e-3)
    assert result_input_shape == ((BLOCK_SIZE, BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
    assert result_output_shape == (BLOCK_SIZE, BLOCK_SIZE)
    assert result_other_shape == (BLOCK_SIZE, BLOCK_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = args.device

    BLOCK_SIZE = 32
    input_matrix1, input_matrix2, result = perform_dot(device, BLOCK_SIZE)
    triton_viz.launch()
