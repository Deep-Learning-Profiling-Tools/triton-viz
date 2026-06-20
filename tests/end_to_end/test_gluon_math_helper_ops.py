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
def _math_helpers_kernel(out, clamp_out, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    values = gl.exp2(offsets.to(gl.float32))
    logs = gl.log2(values)
    mask = offsets < (BLOCK // 2)
    selected = gl.where(mask, logs, 0.0)
    gl.store(out, gl.sum(selected, axis=0))
    centered = offsets.to(gl.float32) - (BLOCK // 2)
    gl.store(clamp_out + offsets, gl.clamp(centered, -2.0, 2.0))


def test_gluon_math_helpers_match_cpu_results():
    out = torch.empty((1,), dtype=torch.float32)
    clamp_out = torch.empty((8,), dtype=torch.float32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_math_helpers_kernel)

    kernel[(1,)](out, clamp_out, 8, num_warps=1)

    assert out.item() == sum(range(4))
    expected_clamp = torch.clamp(torch.arange(8, dtype=torch.float32) - 4, -2.0, 2.0)
    torch.testing.assert_close(clamp_out, expected_clamp, atol=0, rtol=0)
