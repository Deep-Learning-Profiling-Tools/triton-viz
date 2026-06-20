import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
)
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
def _tma_copy_1d_kernel(in_desc, out_desc, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(in_desc.dtype, [BLOCK], in_desc.layout)
    barrier = gl.allocate_shared_memory(gl.int64, [1], in_desc.layout)

    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid * BLOCK], barrier, smem)
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)
    tma.async_store(out_desc, [pid * BLOCK], smem)
    tma.store_wait(0)


@gluon.jit
def _tma_elementwise_add_kernel(
    a_desc,
    b_desc,
    out_desc,
    xnumel,
    ynumel,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    a_smem = gl.allocate_shared_memory(a_desc.dtype, [XBLOCK, YBLOCK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [XBLOCK, YBLOCK], b_desc.layout)
    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        [XBLOCK, YBLOCK],
        out_desc.layout,
    )
    barrier = gl.allocate_shared_memory(gl.int64, [1], a_desc.layout)

    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [0, 0], barrier, a_smem)
    tma.async_load(b_desc, [0, 0], barrier, b_smem)
    mbarrier.wait(barrier, phase=0)
    out_smem.store(a_smem.load(layout) + b_smem.load(layout))
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


@gluon.jit
def _tma_atomic_float_kernel(
    add_desc,
    min_desc,
    max_desc,
    src_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    src = gl.load(src_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    smem = gl.allocate_shared_memory(src.dtype, [BLOCK_M, BLOCK_N], add_desc.layout)
    smem.store(src)
    hopper.fence_async_shared()
    tma.async_atomic_add(add_desc, [1, 2], smem)
    tma.async_atomic_min(min_desc, [1, 2], smem)
    tma.async_atomic_max(max_desc, [1, 2], smem)
    tma.store_wait(0)


def test_gluon_tma_runs_1d_copy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    layout = gl.NVMMASharedLayout.get_default_for([64], gl.float32)
    in_desc = TensorDescriptor.from_tensor(inp, [64], layout)
    out_desc = TensorDescriptor.from_tensor(out, [64], layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_tma_copy_1d_kernel)

    kernel[(1,)](in_desc, out_desc, 64, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_tma_runs_staged_elementwise_add_on_cpu():
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b = 10 + a
    out = torch.full_like(a, -1)
    layout = gl.NVMMASharedLayout.get_default_for([4, 8], gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, [4, 8], layout)
    b_desc = TensorDescriptor.from_tensor(b, [4, 8], layout)
    out_desc = TensorDescriptor.from_tensor(out, [4, 8], layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _tma_elementwise_add_kernel
    )

    kernel[(1,)](a_desc, b_desc, out_desc, *a.shape, 4, 8, num_warps=4)

    torch.testing.assert_close(out, a + b, atol=0, rtol=0)


def test_gluon_tma_runs_float_atomics_on_cpu():
    block_m = 2
    block_n = 4
    src = torch.tensor(
        [[3.0, -2.0, 0.5, 8.0], [1.5, 7.0, -4.0, 0.25]],
        dtype=torch.float32,
    )
    add_dst = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    min_dst = add_dst + 10.0
    max_dst = add_dst - 10.0
    layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float32)
    add_desc = TensorDescriptor.from_tensor(add_dst, [block_m, block_n], layout)
    min_desc = TensorDescriptor.from_tensor(min_dst, [block_m, block_n], layout)
    max_desc = TensorDescriptor.from_tensor(max_dst, [block_m, block_n], layout)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_tma_atomic_float_kernel)

    expected_add = add_dst.clone()
    expected_min = min_dst.clone()
    expected_max = max_dst.clone()
    expected_add[1:3, 2:6] += src
    expected_min[1:3, 2:6] = torch.minimum(expected_min[1:3, 2:6], src)
    expected_max[1:3, 2:6] = torch.maximum(expected_max[1:3, 2:6], src)

    kernel[(1,)](add_desc, min_desc, max_desc, src, block_m, block_n, num_warps=4)

    torch.testing.assert_close(add_dst, expected_add, atol=0, rtol=0)
    torch.testing.assert_close(min_dst, expected_min, atol=0, rtol=0)
    torch.testing.assert_close(max_dst, expected_max, atol=0, rtol=0)
