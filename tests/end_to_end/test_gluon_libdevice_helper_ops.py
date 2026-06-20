import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton.language.extra.libdevice as libdevice

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
def _libdevice_helpers_kernel(
    x_ptr,
    y_ptr,
    exp_out,
    fast_exp_out,
    div_out,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    x = gl.load(x_ptr + offsets)
    y = gl.load(y_ptr + offsets)
    gl.store(exp_out + offsets, libdevice.exp(x))
    gl.store(fast_exp_out + offsets, libdevice.fast_expf(x))
    gl.store(div_out + offsets, libdevice.fast_dividef(x, y))


def test_gluon_libdevice_helpers_match_cpu_results():
    x = torch.tensor([-2.0, -0.5, 0.25, 1.5], dtype=torch.float32)
    y = torch.tensor([2.0, -4.0, 0.5, 3.0], dtype=torch.float32)
    exp_out = torch.empty_like(x)
    fast_exp_out = torch.empty_like(x)
    div_out = torch.empty_like(x)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _libdevice_helpers_kernel
    )

    kernel[(1,)](x, y, exp_out, fast_exp_out, div_out, x.numel(), num_warps=1)

    torch.testing.assert_close(exp_out, torch.exp(x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_exp_out, torch.exp(x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(div_out, x / y, atol=0, rtol=0)
