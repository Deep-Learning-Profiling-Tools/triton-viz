import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer

N = 16
BLOCK_SIZE = 16


@triton_viz.trace(clients=Tracer())
@triton.jit
def correct_dequant_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(y_ptr + offsets, vals.to(tl.float32), mask=mask)


def run_demo():
    x = torch.arange(100, 116, dtype=torch.int8)
    y_ok = torch.empty(N, dtype=torch.float32)

    correct_dequant_kernel[(1,)](x, y_ok, N, BLOCK_SIZE=BLOCK_SIZE)

    print("correct output:", y_ok.cpu().tolist())

    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()
