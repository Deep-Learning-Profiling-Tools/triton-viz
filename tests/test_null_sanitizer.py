import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz import config as cfg


cfg.sanitizer_backend = "off"


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def null_sanitizer_kernel(idx_ptr):
    a = tl.load(idx_ptr)
    tl.store(idx_ptr, a + 5)

def test_null_sanitizer():
    idx = torch.arange(128, dtype=torch.int32)
    null_sanitizer_kernel[(1,)](idx)
