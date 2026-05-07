"""WAW race: All blocks write to the same scalar address without atomics."""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector


_detector = RaceDetector()


@triton_viz.trace(_detector)
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(values)
    tl.store(output_ptr, block_sum)  # BUG: all blocks -> same addr


if __name__ == "__main__":
    n, bs = 64, 8
    inp = torch.randn(n, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    reduction_kernel[(triton.cdiv(n, bs),)](inp, out, n, bs)

    if _detector.last_status == "unsupported":
        print(f"Race analysis unsupported: {_detector.unsupported_reason}")
    else:
        races = _detector.last_reports
        print(f"Detected {len(races)} race(s)")
        for r in races:
            print(
                f"  {r.race_type.name} witness_addr=0x{r.witness_addr:x} "
                f"grid_a={r.witness_grid_a} grid_b={r.witness_grid_b}"
            )
