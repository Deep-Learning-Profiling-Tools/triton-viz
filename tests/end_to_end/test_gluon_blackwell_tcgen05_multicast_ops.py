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
def _tma_tcgen05_multicast_kernel(
    a_desc,
    b_desc,
    out_desc,
    NUM_K_TILES: gl.constexpr,
    acc_tmem_layout: gl.constexpr,
    num_warps: gl.constexpr,
):
    gl.static_assert(gl.num_ctas() == 4)
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_k: gl.constexpr = a_desc.block_shape[1]
    block_n: gl.constexpr = b_desc.block_shape[1]

    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    acc_tmem = blackwell.allocate_tensor_memory(
        gl.float32,
        [block_m, block_n],
        acc_tmem_layout,
    )
    tma_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_barrier, count=1)
    mbarrier.init(mma_barrier, count=1)

    phase_tma = 0
    phase_mma = 0
    for k in range(NUM_K_TILES):
        mbarrier.expect(tma_barrier, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, k * block_k], tma_barrier, smem_a, multicast=True)
        tma.async_load(b_desc, [k * block_k, 0], tma_barrier, smem_b, multicast=True)
        mbarrier.wait(tma_barrier, phase=phase_tma, deps=[smem_a, smem_b])
        phase_tma ^= 1
        blackwell.tcgen05_mma(
            smem_a,
            smem_b,
            acc_tmem,
            use_acc=(k != 0),
            multicast=True,
            mbarriers=[mma_barrier],
        )
        mbarrier.wait(mma_barrier, phase=phase_mma, deps=[smem_a, smem_b])
        phase_mma ^= 1

    mbarrier.invalidate(tma_barrier)
    mbarrier.invalidate(mma_barrier)
    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_shape,
        out_desc.layout,
    )
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout(num_warps=num_warps)
    out_smem.store(acc_tmem.load(acc_reg_layout).to(out_desc.dtype))
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


def test_gluon_blackwell_tma_tcgen05_multicast_on_cpu():
    torch.manual_seed(21)
    block_m = 256
    block_n = 64
    block_k = 16
    num_k_tiles = 2
    a = torch.randn((block_m, block_k * num_k_tiles), dtype=torch.float16) / 4
    b = torch.randn((block_k * num_k_tiles, block_n), dtype=torch.float16) / 4
    out = torch.empty((block_m, block_n), dtype=torch.float16)
    cga_layout_a = ((1, 0), (2, 0))
    cga_layout_b = ((0, 1), (0, 0))
    cga_layout_out = ((1, 0), (2, 0))
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_k],
        gl.float16,
        cga_layout=cga_layout_a,
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [block_k, block_n],
        gl.float16,
        cga_layout=cga_layout_b,
    )
    out_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_n],
        gl.float16,
        cga_layout=cga_layout_out,
    )
    acc_layout = blackwell.TensorMemoryLayout(
        block=(64, block_n),
        col_stride=1,
        cga_layout=cga_layout_out,
        two_ctas=True,
    )
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    out_desc = TensorDescriptor.from_tensor(out, [block_m, block_n], out_layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _tma_tcgen05_multicast_kernel
    )

    kernel[(1,)](
        a_desc,
        b_desc,
        out_desc,
        num_k_tiles,
        acc_layout,
        num_warps=4,
        num_ctas=4,
    )

    expected = (a.float() @ b.float()).half()
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
