import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Sanitizer


def test_tl_program_id():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel(x):
        pid = tl.program_id(0)
        addr = x + pid
        tl.load(addr)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    add_kernel[(2,)](a)

def test_tl_make_range():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def make_range_kernel(x, BLOCK_SIZE: tl.constexpr):
        tl.load(x)
        offset = x + tl.arange(0, BLOCK_SIZE)
        tl.load(offset)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    make_range_kernel[(1,)](a, BLOCK_SIZE=16)

def test_tl_add():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def program_id_kernel(x):
        addr = x + 1
        tl.load(addr)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    program_id_kernel[(2,)](a)

def test_tl_sub():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def sub_kernel(x):
        addr = x - 1
        tl.load(addr)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    sub_kernel[(2,)](a)

def test_tl_mul():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def mul_kernel(x, BLOCK_SIZE: tl.constexpr):
        addr = x + (tl.arange(0, BLOCK_SIZE) * 2)
        tl.load(addr)
    a = torch.randn(32, dtype=torch.float32, device='cuda')
    mul_kernel[(1,)](a, BLOCK_SIZE=16)

def test_tl_div():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def div_kernel(x, BLOCK_SIZE: tl.constexpr):
        tl.load(x)
        tl.load(x + (tl.arange(0, BLOCK_SIZE) // 2))
        tl.load(x + tl.arange(0, BLOCK_SIZE))
    a = torch.randn(32, dtype=torch.float32, device='cuda')
    div_kernel[(1,)](a, BLOCK_SIZE=16)

# def test_tl_mod():
#     @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
#     @triton.jit
#     def mod_kernel(x, BLOCK_SIZE: tl.constexpr):
#         tl.load(x)
#         tl.load(x + (tl.arange(0, BLOCK_SIZE) % 10))
#         tl.load(x + tl.arange(0, BLOCK_SIZE))
#     a = torch.randn(32, dtype=torch.float32, device='cuda')
#     mod_kernel[(1,)](a, BLOCK_SIZE=16)

def test_vec_add():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        output = tl.zeros(x.shape, dtype=x.dtype) + x + y
        tl.store(output_ptr + offsets, output)

    access_size = 24
    size = 17
    BLOCK_SIZE = 8
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    output = torch.empty_like(a, device='cuda')

    grid = lambda meta: (triton.cdiv(access_size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, output, BLOCK_SIZE=BLOCK_SIZE)

def test_vec_add_mask():
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

    access_size = 24
    size = 17
    BLOCK_SIZE = 8
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    output = torch.empty_like(a, device='cuda')

    grid = lambda meta: (triton.cdiv(access_size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, output, size, BLOCK_SIZE=BLOCK_SIZE)

def test_new_axis_column():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def new_axis_kernel(out_ptr, BLOCK_ROW_SIZE: tl.constexpr):
        pid = out_ptr + tl.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        tl.load(pid)

    BLOCK_ROW_SIZE = 8
    out = torch.empty((BLOCK_ROW_SIZE, 1), dtype=torch.int32, device='cuda')
    grid = lambda meta: (1,)
    new_axis_kernel[grid](out, BLOCK_ROW_SIZE=BLOCK_ROW_SIZE)


def test_new_axis_row():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def new_axis_kernel(out_ptr, BLOCK_ROW_SIZE: tl.constexpr):
        pid = out_ptr + tl.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[None, :]
        tl.load(pid)

    BLOCK_ROW_SIZE = 8
    out = torch.empty((BLOCK_ROW_SIZE, 1), dtype=torch.int32, device='cuda')
    grid = lambda meta: (1,)
    new_axis_kernel[grid](out, BLOCK_ROW_SIZE=BLOCK_ROW_SIZE)

def test_tl_maximum():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def maximum_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # read data from x_ptr and y_ptr
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        # calculate maximum
        out = tl.maximum(x, y)
        # write back
        tl.store(out_ptr + offsets, out, mask=mask)

    size = 20
    BLOCK_SIZE = 8
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    out = torch.empty_like(a, device='cuda')

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    maximum_kernel[grid](a, b, out, size, BLOCK_SIZE=BLOCK_SIZE)