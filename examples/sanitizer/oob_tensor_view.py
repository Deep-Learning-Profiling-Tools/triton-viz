"""Reproduces triton-lang/triton#10336.

When a masked ``tl.load`` uses a logical length larger than the view's
element count, the JIT performs a "wild pointer read" past the underlying
storage. This example wraps the kernel with the triton-viz sanitizer so the
OOB access is caught symbolically instead of silently returning OOB memory.
"""

import torch

import triton
import triton.language as tl

import triton_viz


@triton_viz.trace("sanitizer")
@triton.jit
def oob_load_kernel(
    vals_ptr,
    bins_ptr,
    out_ptr,
    L: tl.constexpr,
    SV: tl.constexpr,
    SB: tl.constexpr,
):
    offs = tl.arange(0, 8)
    v = tl.load(vals_ptr + offs * SV, mask=offs < L, other=0)
    b = tl.load(bins_ptr + offs * SB, mask=offs < L, other=-1)  # OOB HERE!
    tl.store(out_ptr + offs, v * 100 + b, mask=offs < L)


def test_oob_tensor_view():
    vals_base = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    bins_base = torch.tensor([10, 99, 11, 99, 12], dtype=torch.int32)

    vals = vals_base
    bins = bins_base[0::2]  # visible values: [10, 11, 12], stride = 2
    out = torch.empty((4,), dtype=torch.int32)

    # L=4 is valid for vals but invalid for bins.
    # For offs=3 the load resolves to bins_ptr + 3*2 = bins_base[6], which is
    # past the 5-element underlying storage.
    oob_load_kernel[(1,)](vals, bins, out, L=4, SV=vals.stride(0), SB=bins.stride(0))
    print(out)


test_oob_tensor_view()
