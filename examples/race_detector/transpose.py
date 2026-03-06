"""WAR+WAW race: In-place matrix transpose where blocks read/write overlapping addresses."""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector


@triton_viz.trace(RaceDetector())
@triton.jit
def transpose_kernel(matrix_ptr, N, BLOCK: tl.constexpr):
    # Each block handles one row: reads row pid, writes column pid
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    read_mask = cols < N
    # Read row pid: matrix[pid, :]
    read_off = pid * N + cols
    vals = tl.load(matrix_ptr + read_off, mask=read_mask, other=0.0)
    # Write column pid: matrix[:, pid]  (i.e. matrix[col, pid] for each col)
    write_off = cols * N + pid
    tl.store(matrix_ptr + write_off, vals, mask=read_mask)


if __name__ == "__main__":
    from triton_viz.core.trace import launches

    N, block = 8, 8
    mat = torch.randn(N, N, dtype=torch.float32)
    transpose_kernel[(N,)](mat, N, block)

    races = launches[-1].records
    print(f"Detected {len(races)} race(s)")
    for r in races:
        print(f"  {r.race_type.name} at address offset {r.address_offset}")
