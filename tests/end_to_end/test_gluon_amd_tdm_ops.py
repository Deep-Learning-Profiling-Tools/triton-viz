import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

try:
    from triton.experimental.gluon.language.amd.gfx1250 import tdm as amd_tdm
except ImportError as exc:
    amd_tdm = pytest.importorskip(
        "triton.experimental.gluon.language.amd.gfx1250.tdm",
        reason=f"AMD gfx1250 TDM helpers are unavailable: {exc}",
    )

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
def _amd_tdm_copy_kernel(
    inp,
    copy_out,
    gather_out,
    scatter_out,
    row_indices_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    index_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    desc = amd_tdm.make_tensor_descriptor(
        inp,
        [M, N],
        [N, 1],
        [BLOCK_M, BLOCK_N],
        shared_layout,
    )
    copy_desc = amd_tdm.make_tensor_descriptor(
        copy_out,
        [M, N],
        [N, 1],
        [BLOCK_M, BLOCK_N],
        shared_layout,
    )
    gather_desc = amd_tdm.make_tensor_descriptor(
        gather_out,
        [BLOCK_M, BLOCK_N],
        [BLOCK_N, 1],
        [BLOCK_M, BLOCK_N],
        shared_layout,
    )
    scatter_desc = amd_tdm.make_tensor_descriptor(
        scatter_out,
        [M, N],
        [N, 1],
        [BLOCK_M, BLOCK_N],
        shared_layout,
    )
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], shared_layout)
    amd_tdm.async_load(desc, [0, 0], smem)
    amd_tdm.async_wait(0)
    amd_tdm.async_store(copy_desc, [0, 0], smem)
    amd_tdm.prefetch(desc, [0, 0])

    row_offsets = gl.arange(0, BLOCK_M, index_layout)
    row_indices = gl.load(row_indices_ptr + row_offsets)
    gather_smem = gl.allocate_shared_memory(
        gl.float32, [BLOCK_M, BLOCK_N], shared_layout
    )
    amd_tdm.async_gather(desc, row_indices, 0, gather_smem)
    amd_tdm.async_wait(0)
    amd_tdm.async_store(gather_desc, [0, 0], gather_smem)
    amd_tdm.async_scatter(scatter_desc, row_indices, 0, gather_smem)
    amd_tdm.async_wait(0)


@gluon.jit
def _amd_tdm_update_descriptor_kernel(
    inp,
    out_offset,
    out_bounded,
    M: gl.constexpr,
    N: gl.constexpr,
):
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    desc = amd_tdm.make_tensor_descriptor(
        inp,
        [M, N],
        [N, 1],
        [2, N],
        layout,
    )
    offset_desc = amd_tdm.update_tensor_descriptor(desc, add_offsets=[2, 0])
    offset_smem = gl.allocate_shared_memory(gl.float32, [2, N], layout)
    amd_tdm.async_load(offset_desc, [0, 0], offset_smem)
    amd_tdm.async_store(
        amd_tdm.make_tensor_descriptor(out_offset, [2, N], [N, 1], [2, N], layout),
        [0, 0],
        offset_smem,
    )
    bounded_desc = amd_tdm.update_tensor_descriptor(
        desc,
        add_offsets=[M - 1, 0],
        set_bounds=[1, N],
    )
    bounded_smem = gl.allocate_shared_memory(gl.float32, [2, N], layout)
    amd_tdm.async_load(bounded_desc, [0, 0], bounded_smem)
    amd_tdm.async_store(
        amd_tdm.make_tensor_descriptor(out_bounded, [2, N], [N, 1], [2, N], layout),
        [0, 0],
        bounded_smem,
    )


def test_gluon_amd_tdm_copy_gather_scatter_match_cpu_results():
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    copy_out = torch.full_like(values, -1.0)
    gather_out = torch.full((4, 4), -1.0, dtype=torch.float32)
    scatter_out = torch.full_like(values, -1.0)
    row_indices = torch.tensor([4, 1, 3, 0], dtype=torch.int32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_amd_tdm_copy_kernel)

    kernel[(1,)](
        values,
        copy_out,
        gather_out,
        scatter_out,
        row_indices,
        values.shape[0],
        values.shape[1],
        row_indices.numel(),
        values.shape[1],
        num_warps=1,
    )

    expected_copy = torch.full_like(values, -1.0)
    expected_copy[: row_indices.numel(), :] = values[: row_indices.numel(), :]
    torch.testing.assert_close(copy_out, expected_copy, atol=0, rtol=0)
    torch.testing.assert_close(gather_out, values[row_indices], atol=0, rtol=0)
    expected_scatter = torch.full_like(values, -1.0)
    expected_scatter[row_indices] = values[row_indices]
    torch.testing.assert_close(scatter_out, expected_scatter, atol=0, rtol=0)


def test_gluon_amd_tdm_descriptor_updates_match_cpu_results():
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    out_offset = torch.full((2, 4), -1.0, dtype=torch.float32)
    out_bounded = torch.full((2, 4), -1.0, dtype=torch.float32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _amd_tdm_update_descriptor_kernel
    )

    kernel[(1,)](
        values,
        out_offset,
        out_bounded,
        values.shape[0],
        values.shape[1],
        num_warps=1,
    )

    torch.testing.assert_close(out_offset, values[2:4], atol=0, rtol=0)
    expected_bounded = torch.zeros_like(out_bounded)
    expected_bounded[0] = values[-1]
    torch.testing.assert_close(out_bounded, expected_bounded, atol=0, rtol=0)
