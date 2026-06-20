import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell

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
def _tensor_memory_roundtrip_kernel(
    in_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    num_warps: gl.constexpr,
):
    global_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, num_warps],
        [1, 0],
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_layout))
    offsets = offs_m[:, None] * N + offs_n[None, :]
    value = gl.load(in_ptr + offsets)
    tmem_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )
    tmem = blackwell.allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )
    reg_layout: gl.constexpr = tmem.get_reg_layout(num_warps=num_warps)
    value = gl.convert_layout(value, reg_layout)
    tmem.store(value)
    out = gl.convert_layout(tmem.load(reg_layout), global_layout)
    gl.store(out_ptr + offsets, out)


def test_gluon_blackwell_tensor_memory_roundtrips_tile_on_cpu():
    inp = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    out = torch.empty_like(inp)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _tensor_memory_roundtrip_kernel
    )

    kernel[(1,)](inp, out, *inp.shape, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
