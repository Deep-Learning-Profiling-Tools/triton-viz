import torch
import numpy as np
import triton
import triton.language as tl
import triton_viz
import argparse
from triton_viz.core.trace import launches
from triton_viz.core import ReduceSum
from triton_viz.clients import Tracer


@triton_viz.trace(clients=Tracer(grid_idx=(0,)))
@triton.jit
def sum_kernel(
    x_ptr,
    y_ptr,
    STRIDE: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    x_val = tl.load(
        x_ptr
        + tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE
        + tl.arange(0, CHANNEL_SIZE)[None, :]
    )
    x_sum = tl.sum(x_val, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), x_sum)


def perform_sum(device, BLOCK_SIZE, CHANNEL_SIZE):
    x = torch.ones((BLOCK_SIZE, CHANNEL_SIZE), device=device, dtype=torch.long)
    y = torch.zeros((BLOCK_SIZE), device=device, dtype=torch.long)
    sum_kernel[(1,)](x, y, CHANNEL_SIZE, CHANNEL_SIZE, BLOCK_SIZE)
    return x, y


def test_sum():
    BLOCK_SIZE = 128
    CHANNEL_SIZE = 8
    device = "cpu"
    input_matrix, result = perform_sum(device, BLOCK_SIZE, CHANNEL_SIZE)
    expected_output = torch.from_numpy(
        np.sum(input_matrix.numpy(), axis=1, keepdims=False)
    )
    expected_op_name = "reduce_sum"
    result_variables = {}
    variables = {"axis": 1, "keepdims": False}
    for record in launches[0].records:
        if isinstance(record, ReduceSum):
            result_input_shape = record.input_shape
            result_output_shape = record.output_shape
            result_op_name = record.name
            result_variables["index"] = record.index
            result_variables["keep_dims"] = record.keep_dims
    assert torch.allclose(result, expected_output)
    assert result_input_shape == input_matrix.shape
    assert result_output_shape == expected_output.shape
    assert sorted(result_variables.values()) == sorted(variables.values())
    assert result_op_name == expected_op_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = args.device

    BLOCK_SIZE = 128
    CHANNEL_SIZE = 8
    input_matrix, result = perform_sum(device, BLOCK_SIZE, CHANNEL_SIZE)
    triton_viz.launch()
