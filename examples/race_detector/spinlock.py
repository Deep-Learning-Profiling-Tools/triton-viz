"""Example: CAS-based spinlock protecting a shared counter.

Demonstrates how triton-viz's race detector handles CAS spinlock patterns
with release/acquire semantics. The race detector uses the HB solver to
prove that data accesses are ordered through the lock, suppressing false
race reports.

Usage:
    python examples/race_detector/spinlock.py
"""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.core.trace import launches


@triton_viz.trace(RaceDetector())
@triton.jit
def spinlock_increment_kernel(
    counter_ptr,
    lock_ptr,
    n_increments,
    MAX_SPIN: tl.constexpr,
):
    """Each block acquires a spinlock, increments a counter, and releases."""
    # Acquire: spin until CAS succeeds (lock 0 -> 1)
    spins = 0
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") != 0:
        spins += 1
        if spins >= MAX_SPIN:
            return  # give up after MAX_SPIN attempts

    # Critical section: load-modify-store the counter
    val = tl.load(counter_ptr)
    tl.store(counter_ptr, val + 1)

    # Release: unlock (write 0)
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")


def main():
    n_blocks = 4
    max_spin = 1000

    counter = torch.zeros(1, dtype=torch.int32)
    lock = torch.zeros(1, dtype=torch.int32)

    spinlock_increment_kernel[(n_blocks,)](counter, lock, n_blocks, max_spin)

    races = launches[-1].records
    print(f"Blocks: {n_blocks}")
    print(f"Races detected: {len(races)}")
    for r in races:
        print(f"  {r.race_type.name} at offset {r.address_offset}")


if __name__ == "__main__":
    main()
