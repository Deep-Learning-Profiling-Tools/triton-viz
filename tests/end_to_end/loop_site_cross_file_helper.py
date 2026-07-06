"""Helper jit function for the cross-file LoopSite regression test.

The for-loop below must stay at function-relative line 2 (the line right
after ``def``), matching the outer loop in ``loop_site_cross_file_kernel.py``.
"""

import triton
import triton.language as tl


@triton.jit
def inner_store_helper(out_ptr, i):
    for j in range(0, 2):
        tl.store(out_ptr + 2 * i + j, j)
