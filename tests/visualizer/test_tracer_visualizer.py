import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core import config as cfg


@triton_viz.trace(clients=Tracer())
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.zeros(x.shape, dtype=x.dtype)
    output = output + x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_add():
    cfg.reset()
    device = "cpu"
    size = 5000
    BLOCK_SIZE = 1024
    torch.manual_seed(0)
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE)
    assert (x + y == output).all()
    triton_viz.launch()
