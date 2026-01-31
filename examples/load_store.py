import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.config import config as cfg
from triton_viz.core.trace import launches


@triton_viz.trace("profiler")
@triton.jit
def simple_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


if __name__ == "__main__":
    cfg.reset()
    device = "cpu"
    size = 12
    BLOCK_SIZE = 8
    torch.manual_seed(0)
    x = torch.arange(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    simple_kernel[grid](x, output, size, BLOCK_SIZE)
