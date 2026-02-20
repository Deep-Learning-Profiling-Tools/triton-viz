import torch

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients.sanitizer.sanitizer import SymbolicSanitizer


sanitizer = SymbolicSanitizer(abort_on_error=False)


@triton_viz.trace(client=sanitizer)
@triton.jit
def max_return_indices_kernel(inp_ptr, out_val_ptr, out_idx_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    inp = tl.load(inp_ptr + offs)
    max_val, max_idx = tl.max(inp, axis=0, return_indices=True)
    tl.store(out_val_ptr, max_val)
    tl.store(out_idx_ptr, max_idx)


@triton_viz.trace(client=sanitizer)
@triton.jit
def min_return_indices_kernel(inp_ptr, out_val_ptr, out_idx_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    inp = tl.load(inp_ptr + offs)
    min_val, min_idx = tl.min(inp, axis=0, return_indices=True)
    tl.store(out_val_ptr, min_val)
    tl.store(out_idx_ptr, min_idx)


def test_tl_max_return_indices():
    """tl.max with return_indices=True should not crash the sanitizer (issue #167)."""
    sanitizer.records.clear()

    N = 32
    inp = torch.arange(N, dtype=torch.float32)
    out_val = torch.empty(1, dtype=torch.float32)
    out_idx = torch.empty(1, dtype=torch.int32)

    # Should complete without TypeError: 'int' object is not iterable
    max_return_indices_kernel[(1,)](inp, out_val, out_idx, N=N)


def test_tl_min_return_indices():
    """tl.min with return_indices=True should not crash the sanitizer (issue #167)."""
    sanitizer.records.clear()

    N = 32
    inp = torch.arange(N, dtype=torch.float32)
    out_val = torch.empty(1, dtype=torch.float32)
    out_idx = torch.empty(1, dtype=torch.int32)

    # Should complete without TypeError: 'int' object is not iterable
    min_return_indices_kernel[(1,)](inp, out_val, out_idx, N=N)
