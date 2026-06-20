import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import tma as blackwell_tma
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

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
def _blackwell_tma_gather_kernel(
    out_ptr,
    out_stride_x,
    out_stride_y,
    tensor_desc,
    x_offsets_ptr,
    y_offset,
    BLOCK_X: gl.constexpr,
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]
    offsets_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, offsets_layout))
    smem_dest = gl.allocate_shared_memory(
        tensor_desc.dtype,
        [BLOCK_X, BLOCK_Y],
        tensor_desc.layout,
    )
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, BLOCK_X * tensor_desc.block_type.nbytes)
    blackwell_tma.async_gather(
        tensor_desc,
        x_offsets,
        y_offset,
        barrier=barrier,
        result=smem_dest,
    )
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)

    output_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    out = smem_dest.load(output_layout)
    rows = gl.arange(0, BLOCK_X, gl.SliceLayout(1, output_layout))[:, None]
    cols = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, output_layout))[None, :]
    gl.store(out_ptr + rows * out_stride_x + cols * out_stride_y, out)


@gluon.jit
def _blackwell_tma_scatter_kernel(
    tensor_desc,
    x_offsets_ptr,
    y_offset,
    src_ptr,
    src_stride_x,
    src_stride_y,
    BLOCK_X: gl.constexpr,
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]
    source_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    rows = gl.arange(0, BLOCK_X, gl.SliceLayout(1, source_layout))[:, None]
    cols = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, source_layout))[None, :]
    src = gl.load(src_ptr + rows * src_stride_x + cols * src_stride_y)
    offsets_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, offsets_layout))
    smem_src = gl.allocate_shared_memory(
        tensor_desc.dtype,
        [BLOCK_X, BLOCK_Y],
        tensor_desc.layout,
    )
    smem_src.store(src)
    hopper.fence_async_shared()
    blackwell_tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    blackwell_tma.store_wait(0)


def test_gluon_blackwell_tma_runs_gather_on_cpu():
    block_x = 8
    block_y = 8
    y_offset = -2
    inp = torch.arange(6 * 12, dtype=torch.float32).reshape(6, 12)
    x_offsets = torch.tensor([-1, 0, 4, 2, 6, 5, 1, 3], dtype=torch.int32)
    out = torch.full((block_x, block_y), -1.0)
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl.float32)
    desc = TensorDescriptor.from_tensor(inp, [1, block_y], layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _blackwell_tma_gather_kernel
    )

    kernel[(1,)](
        out,
        *out.stride(),
        desc,
        x_offsets,
        y_offset,
        block_x,
        num_warps=4,
    )

    expected = torch.zeros_like(out)
    for out_row, src_row in enumerate(x_offsets.tolist()):
        for out_col in range(block_y):
            src_col = y_offset + out_col
            if 0 <= src_row < inp.shape[0] and 0 <= src_col < inp.shape[1]:
                expected[out_row, out_col] = inp[src_row, src_col]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_gluon_blackwell_tma_runs_scatter_on_cpu():
    block_x = 8
    block_y = 8
    y_offset = 6
    out = torch.full((6, 12), -1.0)
    x_offsets = torch.tensor([0, 5, 6, 3, 2, 8, 1, 4], dtype=torch.int32)
    src = torch.arange(block_x * block_y, dtype=torch.float32).reshape(
        block_x,
        block_y,
    )
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl.float32)
    desc = TensorDescriptor.from_tensor(out, [1, block_y], layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _blackwell_tma_scatter_kernel
    )

    kernel[(1,)](
        desc,
        x_offsets,
        y_offset,
        src,
        *src.stride(),
        block_x,
        num_warps=4,
    )

    expected = torch.full_like(out, -1.0)
    for src_row, dst_row in enumerate(x_offsets.tolist()):
        for src_col in range(block_y):
            dst_col = y_offset + src_col
            if 0 <= dst_row < out.shape[0] and 0 <= dst_col < out.shape[1]:
                expected[dst_row, dst_col] = src[src_row, src_col]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
