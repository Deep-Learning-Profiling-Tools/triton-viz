import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz import config as cfg


cfg.sanitizer_backend = "symexec"


@triton.jit
def kernel_B(ptr, offset):
    # a simple function that adds 1
    val = tl.load(ptr + offset)
    return val + 1


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def kernel_A(ptr, n):
    pid = tl.program_id(0)
    # if pid >= n, there will be an out-of-bounds error
    val = kernel_B(ptr, pid)
    tl.store(ptr + pid, val)


if __name__ == "__main__":
    x = torch.arange(4, dtype=torch.float32)
    print("Input:", x)

    # We'll launch a grid bigger than x.numel() to force a out-of-bounds error
    grid = (x.numel() + 4,)
    kernel_A[grid](x, x.numel())

    print("Output:", x)
