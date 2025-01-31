import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Sanitizer


def test_program_id():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def program_id_kernel(X):
        pid = tl.program_id(0)
        print('pid:', pid)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    program_id_kernel[(16,)](a)

def test_tl_add():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel(X, Y, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        y = tl.load(Y + pid)
        z = x + y
        tl.store(Z + pid, z)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    b = torch.randn(16, dtype=torch.float32, device='cuda')
    c = torch.empty_like(a, device='cuda')
    add_kernel[(16,)](a, b, c)

def test_tl_subtract():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def subtract_kernel(X, Y, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        y = tl.load(Y + pid)
        z = x - y
        tl.store(Z + pid, z)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    b = torch.randn(16, dtype=torch.float32, device='cuda')
    c = torch.empty_like(a, device='cuda')
    subtract_kernel[(16,)](a, b, c)

def test_tl_multiply():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def multiply_kernel(X, Y, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        y = tl.load(Y + pid)
        z = x * y
        tl.store(Z + pid, z)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    b = torch.randn(16, dtype=torch.float32, device='cuda')
    c = torch.empty_like(a, device='cuda')
    multiply_kernel[(16,)](a, b, c)

def test_tl_divide():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def divide_kernel(X, Y, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        y = tl.load(Y + pid)
        z = x / y
        tl.store(Z + pid, z)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    b = torch.randn(16, dtype=torch.float32, device='cuda')
    c = torch.empty_like(a, device='cuda')
    divide_kernel[(16,)](a, b, c)
