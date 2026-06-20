import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell, hopper
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
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
def _two_cta_tcgen05_kernel(
    a_desc,
    b_desc,
    out_desc,
    num_warps: gl.constexpr,
):
    gl.static_assert(gl.num_ctas() == 2)
    cluster_m: gl.constexpr = a_desc.block_shape[0]
    tile_n: gl.constexpr = b_desc.block_shape[1]
    cta_m: gl.constexpr = cluster_m // 2
    cga_layout: gl.constexpr = out_desc.layout.cga_layout

    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    tma_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_barrier, count=1)
    mbarrier.init(mma_barrier, count=1)

    mbarrier.expect(tma_barrier, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_load(a_desc, [0, 0], tma_barrier, smem_a)
    tma.async_load(b_desc, [0, 0], tma_barrier, smem_b)
    mbarrier.wait(tma_barrier, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_barrier)

    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        block=(cta_m, tile_n),
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=True,
    )
    acc = blackwell.allocate_tensor_memory(gl.float32, [cluster_m, tile_n], acc_layout)
    blackwell.tcgen05_mma(smem_a, smem_b, acc, use_acc=False, mbarriers=[mma_barrier])
    mbarrier.wait(mma_barrier, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_barrier)

    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_shape,
        out_desc.layout,
    )
    acc_reg_layout: gl.constexpr = acc.get_reg_layout(num_warps=num_warps)
    out_smem.store(acc.load(acc_reg_layout).to(out_desc.dtype))
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


def test_gluon_blackwell_two_cta_tcgen05_mma_on_cpu():
    torch.manual_seed(12)
    a = torch.randn(128, 16, dtype=torch.float16) / 4
    b = torch.randn(16, 64, dtype=torch.float16) / 4
    out = torch.empty((128, 64), dtype=torch.float16)
    a_layout = gl.NVMMASharedLayout.get_default_for(
        list(a.shape),
        gl.float16,
        cga_layout=[(1, 0)],
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        list(b.shape),
        gl.float16,
        cga_layout=[(0, 1)],
    )
    out_layout = gl.NVMMASharedLayout.get_default_for(
        list(out.shape),
        gl.float16,
        cga_layout=[(1, 0)],
    )
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    out_desc = TensorDescriptor.from_tensor(out, list(out.shape), out_layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_two_cta_tcgen05_kernel)

    kernel[(1,)](a_desc, b_desc, out_desc, num_warps=4, num_ctas=2)

    expected = (a.float() @ b.float()).half()
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
