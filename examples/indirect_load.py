import torch
import triton
import triton.language as tl

import triton_viz

N = 64
BLOCK = 64


@triton_viz.trace("tracer")
@triton.jit
def indirect_load_kernel(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    idx = tl.load(rand_ptr + offs)
    vals = tl.load(x_ptr + idx)
    tl.store(out_ptr + offs, vals)


def run():
    device = "cpu"
    torch.manual_seed(0)
    x = torch.arange(N, dtype=torch.float32, device=device)
    rand = torch.randperm(N, dtype=torch.int32, device=device)
    out = torch.empty_like(x)
    indirect_load_kernel[(1,)](x, rand, out, BLOCK=BLOCK)
    triton_viz.launch(share=False)


if __name__ == "__main__":
    run()
