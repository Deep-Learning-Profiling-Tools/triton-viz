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
def _amd_warp_pipeline_stage_kernel(x_ptr, y_ptr, out_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    with gl.amd.warp_pipeline_stage("load", priority=3):
        x = gl.load(x_ptr + offsets)
        y = gl.load(y_ptr + offsets)
    with gl.amd.warp_pipeline_stage("compute", priority=0):
        values = x * 2.0 + y
    gl.store(out_ptr + offsets, values)


def test_gluon_amd_warp_pipeline_stage_matches_cpu_results():
    x = torch.tensor([1.0, -2.0, 3.5, 4.0], dtype=torch.float32)
    y = torch.tensor([0.5, 8.0, -1.5, 2.0], dtype=torch.float32)
    out = torch.empty_like(x)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _amd_warp_pipeline_stage_kernel
    )

    kernel[(1,)](x, y, out, x.numel(), num_warps=1)

    torch.testing.assert_close(out, x * 2.0 + y, atol=0, rtol=0)
