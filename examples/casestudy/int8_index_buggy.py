import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer

N = 16
BLOCK_SIZE = 16
WRAP_START = 8  # treat idx >= 8 as the "overflow" window


@triton_viz.trace(clients=Tracer())
@triton.jit
def buggy_dequant_kernel(
    x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr, WRAP_START: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Simulate int8 overflow by wrapping the last half of the tile back to the head
    wrapped = tl.where(offsets < WRAP_START, offsets, offsets - WRAP_START)
    vals = tl.load(x_ptr + wrapped, mask=mask, other=0)
    tl.store(y_ptr + offsets, vals.to(tl.float32), mask=mask)


def run_demo():
    # Compact tensor so the visualizer only draws a single 16-element tile
    x = torch.arange(100, 116, dtype=torch.int8)
    y_bug = torch.empty(N, dtype=torch.float32)

    buggy_dequant_kernel[(1,)](x, y_bug, N, BLOCK_SIZE=BLOCK_SIZE, WRAP_START=WRAP_START)

    print("buggy output:", y_bug.cpu().tolist())

    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()

