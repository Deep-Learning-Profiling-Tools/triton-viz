import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell, hopper
from triton.experimental.gluon.language.nvidia.hopper import mbarrier

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
def _tcgen05_copy_kernel(
    in_ptr,
    in_stride0,
    in_stride1,
    out_ptr,
    out_stride0,
    out_stride1,
    M: gl.constexpr,
    N: gl.constexpr,
    smem_layout: gl.constexpr,
    tmem_layout: gl.constexpr,
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
    value = gl.load(
        in_ptr + offs_m[:, None] * in_stride0 + offs_n[None, :] * in_stride1
    )
    smem = gl.allocate_shared_memory(value.dtype, [M, N], smem_layout)
    tmem = blackwell.allocate_tensor_memory(value.dtype, [M, N], tmem_layout)
    smem.store(value)
    hopper.fence_async_shared()
    blackwell.tcgen05_copy(smem, tmem)

    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    blackwell.tcgen05_commit(barrier)
    mbarrier.wait(barrier, 0)

    reg_layout: gl.constexpr = tmem.get_reg_layout(num_warps=num_warps)
    output = gl.convert_layout(tmem.load(reg_layout), layout)
    gl.store(
        out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1,
        output,
    )


def test_gluon_blackwell_tcgen05_copy_roundtrips_tile_on_cpu():
    base = torch.arange(96 * 96, dtype=torch.float32).reshape(96, 96)
    inp = base[:64, :64]
    out_base = torch.full_like(base, -1)
    out = out_base[16:80, 8:72]
    smem_layout = gl.NVMMASharedLayout.get_default_for([64, 64], gl.float32)
    tmem_layout = blackwell.TensorMemoryLayout((64, 64), col_stride=1)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_tcgen05_copy_kernel)

    kernel[(1,)](
        inp,
        *inp.stride(),
        out,
        *out.stride(),
        *inp.shape,
        smem_layout,
        tmem_layout,
        num_warps=4,
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert torch.all(out_base[:16] == -1)
    assert torch.all(out_base[80:] == -1)
