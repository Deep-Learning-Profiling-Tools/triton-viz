"""RAW race: Block N reads x[N+1 region] while Block N+1 writes x[N+1 region]."""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector


@triton_viz.trace(RaceDetector())
@triton.jit
def inplace_neighbor_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    own = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    own_mask = own < n_elements
    own_data = tl.load(x_ptr + own, mask=own_mask, other=0.0)

    neighbor = (pid + 1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    neighbor_mask = neighbor < n_elements
    neighbor_data = tl.load(x_ptr + neighbor, mask=neighbor_mask, other=0.0)

    tl.store(x_ptr + own, own_data + neighbor_data * 0.5, mask=own_mask)


if __name__ == "__main__":
    from triton_viz.core.trace import launches

    n, bs = 32, 8
    x = torch.randn(n, dtype=torch.float32)
    inplace_neighbor_kernel[(triton.cdiv(n, bs),)](x, n, bs)

    races = launches[-1].records
    print(f"Detected {len(races)} race(s)")
    for r in races:
        print(f"  {r.race_type.name} at address offset {r.address_offset}")
