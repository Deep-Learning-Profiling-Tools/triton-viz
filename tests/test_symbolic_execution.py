import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Sanitizer


def test_tl_addptr():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def program_id_kernel(x):
        addr = x + 1
        tl.load(addr)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    program_id_kernel[(16,)](a)

def test_tl_program_id():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel(x):
        pid = tl.program_id(0)
        addr = x + pid
        tl.load(addr)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    add_kernel[(16,)](a)

def test_tl_make_range():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def make_range_kernel(x, BLOCK_SIZE: tl.constexpr):
        offset = x + tl.arange(0, BLOCK_SIZE)
        tl.load(offset)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    make_range_kernel[(1,)](a, BLOCK_SIZE=16)

def test_vec_add():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = tl.zeros(x.shape, dtype=x.dtype) + x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    access_size = 16
    size = 16
    BLOCK_SIZE = 8
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    output = torch.empty_like(a, device='cuda')

    grid = lambda meta: (triton.cdiv(access_size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, output, access_size, BLOCK_SIZE=BLOCK_SIZE)
