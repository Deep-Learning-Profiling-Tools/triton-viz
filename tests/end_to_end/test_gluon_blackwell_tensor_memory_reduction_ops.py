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
def _tensor_memory_load_reduction_kernel(
    in_ptr,
    values_out,
    max_out,
    min_abs_out,
    M: gl.constexpr,
    N: gl.constexpr,
    num_warps: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    data = gl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :])

    tmem_layout: gl.constexpr = blackwell.TensorMemoryLayout([64, 64], col_stride=1)
    tmem = blackwell.allocate_tensor_memory(data.dtype, [M, N], tmem_layout)
    tmem.store(data)
    sliced = tmem.slice(8, 16)
    reg_layout: gl.constexpr = sliced.get_reg_layout(num_warps=num_warps)
    values, maxima = sliced.load_max(reg_layout)
    _, minima = sliced.load_min(reg_layout, abs=True)

    out_n = gl.arange(0, 16, gl.SliceLayout(0, layout))
    gl.store(values_out + offs_m[:, None] * 16 + out_n[None, :], values)
    row_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    row_offsets = gl.arange(0, M, row_layout)
    gl.store(max_out + row_offsets, gl.convert_layout(maxima, row_layout))
    gl.store(min_abs_out + row_offsets, gl.convert_layout(minima, row_layout))


def test_gluon_blackwell_tensor_memory_load_reductions_on_cpu():
    inp = (torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) - 2048) / 64
    values_out = torch.empty((64, 16), dtype=torch.float32)
    max_out = torch.empty((64,), dtype=torch.float32)
    min_abs_out = torch.empty((64,), dtype=torch.float32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _tensor_memory_load_reduction_kernel
    )

    kernel[(1,)](
        inp,
        values_out,
        max_out,
        min_abs_out,
        *inp.shape,
        num_warps=4,
    )

    sliced = inp[:, 8:24]
    torch.testing.assert_close(values_out, sliced, atol=0, rtol=0)
    torch.testing.assert_close(max_out, sliced.max(dim=1).values, atol=0, rtol=0)
    torch.testing.assert_close(min_abs_out, sliced.abs().min(dim=1).values)
