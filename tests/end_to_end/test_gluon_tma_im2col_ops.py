import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import (
    TensorDescriptor,
    TensorDescriptorIm2Col,
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
def _tma_im2col_kernel(
    in_desc,
    out_desc,
    coord_n,
    coord_h,
    coord_w,
    coord_c,
    offset_h: gl.constexpr,
    offset_w: gl.constexpr,
):
    smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    barrier = gl.allocate_shared_memory(gl.int64, [1], in_desc.layout)
    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, in_desc.block_type.nbytes)
    tma.async_load_im2col(
        in_desc,
        [coord_n, coord_h, coord_w, coord_c],
        [offset_h, offset_w],
        barrier,
        smem,
    )
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)
    tma.async_store(out_desc, [0, 0], smem)
    tma.store_wait(0)


def _run_im2col_case(
    inp,
    pixel_box_lower_corner,
    pixel_box_upper_corner,
    coord,
    offsets,
):
    out = torch.zeros((16, 32), dtype=torch.float32)
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=32,
        rank=2,
    )
    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        [16, 32],
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=pixel_box_lower_corner,
        pixel_box_upper_corner=pixel_box_upper_corner,
    )
    out_desc = TensorDescriptor.from_tensor(out, [16, 32], layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_tma_im2col_kernel)

    kernel[(1,)](in_desc, out_desc, *coord, *offsets, num_warps=1)

    return out


def test_gluon_tma_im2col_runs_simple_tile_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [0, 0], [0, 0], [0, 0, 0, 0], [0, 0])

    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


def test_gluon_tma_im2col_zero_fills_padded_pixels_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, -1, -1, 0], [0, 0])

    expected_first_channel = torch.tensor(
        [0, 0, 0, 0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out[:, 0], expected_first_channel, atol=0, rtol=0)


def test_gluon_tma_im2col_honors_runtime_offsets_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, -1, -1, 0], [1, 1])

    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)
