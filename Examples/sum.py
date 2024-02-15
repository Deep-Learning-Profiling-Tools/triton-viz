import torch

import triton
import triton.language as tl
import triton_viz


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


triton_viz.sample((0,))
BLOCK_SIZE = 128
CHANNEL_SIZE = 8
x = torch.ones((BLOCK_SIZE, CHANNEL_SIZE), device="cuda", dtype=torch.long)
y = torch.zeros((BLOCK_SIZE), device="cuda", dtype=torch.long)

sum_kernel[(1,)](x, y, CHANNEL_SIZE, CHANNEL_SIZE, BLOCK_SIZE)

triton_viz.dump("./sum.json")
