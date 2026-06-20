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
def _tcgen05_small_mma_kernel(
    a_desc,
    b_desc,
    c_desc,
    out_desc,
    tmem_block: gl.constexpr,
    LHS_IN_TMEM: gl.constexpr,
    USE_COMMIT: gl.constexpr,
    num_warps: gl.constexpr,
):
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype,
        c_desc.block_type.shape,
        c_desc.layout,
    )
    mbarrier.expect(
        barrier,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], barrier, a_smem)
    tma.async_load(b_desc, [0, 0], barrier, b_smem)
    tma.async_load(c_desc, [0, 0], barrier, c_smem)
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)
    mbarrier.init(barrier, count=1)

    M: gl.constexpr = out_desc.block_type.shape[0]
    N: gl.constexpr = out_desc.block_type.shape[1]
    K: gl.constexpr = a_desc.block_type.shape[1]
    acc_tmem_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        tmem_block,
        col_stride=32 // out_desc.dtype.primitive_bitwidth,
    )
    acc_tmem = blackwell.allocate_tensor_memory(
        out_desc.dtype,
        [M, N],
        acc_tmem_layout,
    )
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout(num_warps=num_warps)
    acc = c_smem.load(acc_reg_layout)
    acc_tmem.store(acc)

    if LHS_IN_TMEM:
        lhs_tmem_layout: gl.constexpr = blackwell.TensorMemoryLayout(
            tmem_block,
            col_stride=1,
        )
        lhs_tmem = blackwell.allocate_tensor_memory(
            a_desc.dtype,
            [M, K],
            lhs_tmem_layout,
        )
        lhs_reg_layout: gl.constexpr = lhs_tmem.get_reg_layout(num_warps=num_warps)
        lhs = a_smem.load(lhs_reg_layout)
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    if USE_COMMIT:
        blackwell.tcgen05_mma(a, b_smem, acc_tmem)
        blackwell.tcgen05_commit(barrier)
    else:
        blackwell.tcgen05_mma(
            a,
            b_smem,
            acc_tmem,
            mbarriers=[barrier],
            mbarrier_preds=[True],
        )
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)

    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_type.shape,
        out_desc.layout,
    )
    out_smem.store(acc_tmem.load(acc_reg_layout))
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


def _run_tcgen05_small_mma_case(seed: int, lhs_in_tmem: bool, use_commit: bool):
    torch.manual_seed(seed)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 64, dtype=torch.float16) / 4
    c = torch.randn(64, 64, dtype=torch.float32) / 4
    out = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    c_desc = TensorDescriptor.from_tensor(c, list(c.shape), c_layout)
    out_desc = TensorDescriptor.from_tensor(out, list(out.shape), c_layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _tcgen05_small_mma_kernel
    )

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        out_desc,
        (64, 64),
        lhs_in_tmem,
        use_commit,
        num_warps=4,
    )

    torch.testing.assert_close(out, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_blackwell_tcgen05_runs_small_mma_on_cpu():
    _run_tcgen05_small_mma_case(seed=5, lhs_in_tmem=False, use_commit=True)


def test_gluon_blackwell_tcgen05_runs_small_mma_with_lhs_tensor_memory_on_cpu():
    _run_tcgen05_small_mma_case(seed=6, lhs_in_tmem=True, use_commit=False)
