import torch

import triton
import triton.language as tl
import triton_viz
import argparse


@triton_viz.trace
@triton.jit
def dot_kernel(x_ptr, y_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0) * BLOCK_SIZE
    c = tl.program_id(1) * BLOCK_SIZE
    x_val = tl.load(
        x_ptr
        + (r + tl.arange(0, BLOCK_SIZE)[:, None]) * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    y_val = tl.load(
        y_ptr
        + tl.arange(0, BLOCK_SIZE)[:, None] * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + c
    )
    z = tl.dot(x_val, y_val)
    x_val = tl.load(
        x_ptr
        + (r + tl.arange(0, BLOCK_SIZE)[:, None]) * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + BLOCK_SIZE
    )
    y_val = tl.load(
        y_ptr
        + (BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]) * 2 * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + c
    )
    z = z + tl.dot(x_val, y_val)
    tl.store(
        z_ptr
        + (r + tl.arange(0, BLOCK_SIZE)[:, None]) * (2 * BLOCK_SIZE - 10)
        + tl.arange(0, BLOCK_SIZE)[None, :]
        + c,
        z,
        mask=tl.arange(0, BLOCK_SIZE)[None, :] + c < 2 * BLOCK_SIZE - 10,
    )


argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cpu")
argparser.add_argument("--grid", type=int, default=0)
args = argparser.parse_args()
device = args.device

triton_viz.sample((args.grid // 2, args.grid % 2))
BLOCK_SIZE = 32
x = torch.randn((2 * BLOCK_SIZE, 2 * BLOCK_SIZE), device=device)
y = torch.randn((2 * BLOCK_SIZE, 2 * BLOCK_SIZE), device=device)
z = torch.zeros((2 * BLOCK_SIZE, 2 * BLOCK_SIZE - 10), device=device)

dot_kernel[(2, 2)](x, y, z, BLOCK_SIZE)

triton_viz.dump("./dot.json")
triton_viz.draw(f"dot{args.grid}.png")
