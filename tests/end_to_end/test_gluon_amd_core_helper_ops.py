import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client


class _NoOpClient(Client):
    NAME = "noop"

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        pass

    def grid_idx_callback(self, grid_idx):
        pass

    def register_op_callback(self, op_type, *args, **kwargs):
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return True

    def post_warmup_callback(self, jit_fn, ret):
        pass


@gluon.jit
def _amd_core_helpers_kernel(values_ptr, abs_out, old_out, lock, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    gl.assume(BLOCK > 0)
    offsets = gl.multiple_of(offsets, [1])
    values = gl.load(values_ptr + offsets)
    gl.store(abs_out + offsets, gl.abs(values))
    old = gl.atomic_cas(lock, 0, 7)
    gl.store(old_out, old)


def test_gluon_amd_core_helpers_match_cpu_results():
    values = torch.tensor([-3, 4, -5, 6], dtype=torch.int32)
    abs_out = torch.empty_like(values)
    old_out = torch.empty((), dtype=torch.int32)
    lock = torch.zeros((), dtype=torch.int32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_amd_core_helpers_kernel)

    kernel[(1,)](values, abs_out, old_out, lock, values.numel(), num_warps=1)

    torch.testing.assert_close(abs_out, torch.abs(values), atol=0, rtol=0)
    assert old_out.item() == 0
    assert lock.item() == 7
