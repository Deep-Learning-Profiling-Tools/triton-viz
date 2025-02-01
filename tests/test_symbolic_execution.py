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
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    program_id_kernel[(16,)](a)

def test_tl_add():
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel(x):
        pid = tl.program_id(0)
        addr = x + pid
        tl.load(addr)
    a = torch.randn(16, dtype=torch.float32, device='cuda')
    add_kernel[(16,)](a)
