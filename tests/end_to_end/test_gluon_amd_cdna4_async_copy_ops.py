import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client

cdna4_async_copy = pytest.importorskip(
    "triton.experimental.gluon.language.amd.cdna4.async_copy",
    reason="AMD CDNA4 async-copy helpers are unavailable",
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
def _amd_cdna4_async_copy_kernel(inp, out_global, out_buffer, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    offsets = gl.arange(0, BLOCK, layout).to(gl.int32)
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], shared_layout)
    cdna4_async_copy.global_load_to_shared(smem, inp + offsets)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    global_values = cdna4_async_copy.load_shared_relaxed(smem, layout)
    gl.store(out_global + offsets, global_values)

    mask = offsets < (BLOCK - 1)
    cdna4_async_copy.buffer_load_to_shared(smem, inp, offsets, mask, other=-3.0)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    buffer_values = cdna4_async_copy.load_shared_relaxed(smem, layout)
    gl.store(out_buffer + offsets, buffer_values)


def test_gluon_amd_cdna4_async_copy_matches_cpu_results():
    values = torch.tensor([1.0, -2.0, 3.5, 4.25], dtype=torch.float32)
    out_global = torch.full_like(values, -1.0)
    out_buffer = torch.full_like(values, -1.0)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _amd_cdna4_async_copy_kernel
    )

    kernel[(1,)](values, out_global, out_buffer, values.numel(), num_warps=1)

    torch.testing.assert_close(out_global, values, atol=0, rtol=0)
    expected_buffer = torch.tensor([1.0, -2.0, 3.5, -3.0], dtype=torch.float32)
    torch.testing.assert_close(out_buffer, expected_buffer, atol=0, rtol=0)
