"""Example: Release/acquire flag handoff between producer and consumer blocks.

Demonstrates how triton-viz's race detector handles producer-consumer
synchronization using atomic release/acquire semantics on a flag variable.

The producer block writes data and then sets a flag with release semantics.
The consumer block reads the flag with acquire semantics and then reads data.
The race detector's HB solver proves that the data access is ordered through
the release/acquire flag handoff.

Usage:
    python examples/race_detector/producer_consumer_flag.py
"""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.core.trace import launches


@triton_viz.trace(RaceDetector())
@triton.jit
def producer_consumer_kernel(
    data_ptr,
    flag_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPIN: tl.constexpr,
):
    """Block 0 produces data, blocks 1+ consume it after flag handoff."""
    pid = tl.program_id(0)

    if pid == 0:
        # Producer: write data, then signal via release
        offsets = tl.arange(0, BLOCK_SIZE)
        tl.store(data_ptr + offsets, offsets.to(tl.float32))

        # Release: set flag to 1
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
    else:
        # Consumer: wait for flag, then read data
        spins = 0
        while tl.atomic_cas(flag_ptr, 1, 1, sem="acquire", scope="gpu") != 1:
            spins += 1
            if spins >= MAX_SPIN:
                return

        # Read data (should be ordered after producer's write)
        offsets = tl.arange(0, BLOCK_SIZE)
        _data = tl.load(data_ptr + offsets)


def main():
    n_blocks = 2
    block_size = 16
    max_spin = 1000

    data = torch.zeros(block_size, dtype=torch.float32)
    flag = torch.zeros(1, dtype=torch.int32)

    producer_consumer_kernel[(n_blocks,)](data, flag, block_size, max_spin)

    races = launches[-1].records
    print(f"Blocks: {n_blocks}")
    print(f"Races detected: {len(races)}")
    for r in races:
        print(f"  {r.race_type.name} at offset {r.address_offset}")


if __name__ == "__main__":
    main()
