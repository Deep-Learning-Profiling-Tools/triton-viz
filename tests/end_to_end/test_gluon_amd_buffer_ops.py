import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client

amd_gfx1250 = pytest.importorskip(
    "triton.experimental.gluon.language.amd.gfx1250",
    reason="AMD gfx1250 buffer helpers are unavailable",
)


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
def _amd_buffer_ops_kernel(inp, out_module, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout).to(gl.int32)
    mask = offsets < (BLOCK - 1)
    module_values = gl.amd.gfx1250.buffer_load(inp, offsets, mask=mask, other=-1.0)
    gl.amd.gfx1250.buffer_store(module_values + 1.0, out_module, offsets, mask=mask)


def test_gluon_amd_buffer_ops_match_cpu_results():
    values = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32)
    out_module = torch.full_like(values, -9.0)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_amd_buffer_ops_kernel)

    kernel[(1,)](values, out_module, values.numel(), num_warps=1)

    expected_module = torch.tensor([3.0, 5.0, 9.0, -9.0], dtype=torch.float32)
    torch.testing.assert_close(out_module, expected_module, atol=0, rtol=0)
