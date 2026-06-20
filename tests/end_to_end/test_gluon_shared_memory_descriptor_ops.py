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
def _shared_memory_slice_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    data = gl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :])
    smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [M, N],
        data.dtype,
    )
    smem = gl.allocate_shared_memory(data.dtype, [M, N], smem_layout)
    smem.store(data)
    left = smem.slice(0, N // 2, dim=1)
    right = smem.slice(N // 2, N // 2, dim=1)
    out = left.load(layout) + right.load(layout)
    half_n = gl.arange(0, N // 2, gl.SliceLayout(0, layout))
    gl.store(out_ptr + offs_m[:, None] * (N // 2) + half_n[None, :], out)


def test_gluon_shared_memory_slice_sums_halves_on_cpu():
    inp = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    out = torch.empty((32, 8), dtype=torch.float32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _shared_memory_slice_kernel
    )

    kernel[(1,)](inp, out, *inp.shape, num_warps=4)

    torch.testing.assert_close(out, inp[:, :8] + inp[:, 8:], atol=0, rtol=0)
