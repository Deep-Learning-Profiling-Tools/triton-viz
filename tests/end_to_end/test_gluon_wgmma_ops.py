import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
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


@gluon.constexpr_function
def _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    instr_m = 16
    instr_n = min(BLOCK_N, 64)
    while BLOCK_N % instr_n != 0:
        instr_n -= 8
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[instr_m, instr_n, 256 // dtype.primitive_bitwidth],
    )


@gluon.jit
def _small_wgmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    out_desc,
    LHS_IN_REG: gl.constexpr,
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

    acc_layout: gl.constexpr = _wgmma_layout(
        a_desc.dtype,
        out_desc.block_type.shape[0],
        out_desc.block_type.shape[1],
        num_warps,
    )
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=acc_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )
    a = a_smem.load(a_reg_layout) if LHS_IN_REG else a_smem
    c = c_smem.load(acc_layout)
    out = hopper.warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
    out = hopper.warpgroup_mma_wait(num_outstanding=0, deps=(out,))

    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_type.shape,
        out_desc.layout,
    )
    out_smem.store(out)
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


def _run_small_wgmma_case(seed: int, lhs_in_reg: bool):
    torch.manual_seed(seed)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 32, dtype=torch.float16) / 4
    c = torch.randn(64, 32, dtype=torch.float32) / 4
    out = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    c_desc = TensorDescriptor.from_tensor(c, list(c.shape), c_layout)
    out_desc = TensorDescriptor.from_tensor(out, list(out.shape), c_layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_small_wgmma_kernel)

    kernel[(1,)](a_desc, b_desc, c_desc, out_desc, lhs_in_reg, num_warps=4)

    torch.testing.assert_close(out, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_wgmma_runs_small_mma_on_cpu():
    _run_small_wgmma_case(seed=0, lhs_in_reg=False)


def test_gluon_wgmma_runs_small_mma_with_lhs_registers_on_cpu():
    _run_small_wgmma_case(seed=1, lhs_in_reg=True)
