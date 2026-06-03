import torch
import triton
import triton.language as tl


@triton.jit
def oob_kernel(ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.load(ptr + offsets)  # OOB: BLOCK is larger than the tensor size.


if __name__ == "__main__":
    oob_kernel[(1,)](torch.zeros(4), BLOCK=16)
