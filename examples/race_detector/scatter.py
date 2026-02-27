"""WAW race: Different blocks scatter to overlapping destination indices."""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector


@triton_viz.trace(RaceDetector())
@triton.jit
def scatter_kernel(src_ptr, idx_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(idx_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + indices, values, mask=mask)


if __name__ == "__main__":
    from triton_viz.core.trace import launches

    n, bs = 32, 8
    src = torch.randn(n, dtype=torch.float32)
    # Create conflicts: all indices point to [0, 3]
    idx = torch.randint(0, 4, (n,), dtype=torch.int32)
    dst = torch.zeros(4, dtype=torch.float32)
    scatter_kernel[(triton.cdiv(n, bs),)](src, idx, dst, n, bs)

    races = launches[-1].records
    print(f"Detected {len(races)} race(s)")
    for r in races:
        print(f"  {r.race_type.name} at address offset {r.address_offset}")
