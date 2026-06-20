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
def _tcgen05_scaled_mma_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    VEC: gl.constexpr,
    A_TYPE: gl.constexpr,
    B_TYPE: gl.constexpr,
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
    offs_n_rows = gl.arange(0, N, gl.SliceLayout(1, layout))
    offs_k = gl.arange(0, K, gl.SliceLayout(0, layout))
    offs_k_rows = gl.arange(0, K, gl.SliceLayout(1, layout))
    offs_scale = gl.arange(0, K // VEC, gl.SliceLayout(0, layout))

    a = gl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k_rows[:, None] * N + offs_n[None, :])
    a_scales = gl.load(a_scale_ptr + offs_m[:, None] * (K // VEC) + offs_scale[None, :])
    b_scales = gl.load(
        b_scale_ptr + offs_n_rows[:, None] * (K // VEC) + offs_scale[None, :]
    )

    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [M, K],
        a_ptr.dtype.element_ty,
    )
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [K, N],
        b_ptr.dtype.element_ty,
    )
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [M, K], a_smem_layout)
    b_smem = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [K, N], b_smem_layout)
    a_smem.store(a)
    b_smem.store(b)

    scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    a_scale_tmem = blackwell.allocate_tensor_memory(
        a_scale_ptr.dtype.element_ty,
        [M, K // VEC],
        scale_layout,
    )
    b_scale_tmem = blackwell.allocate_tensor_memory(
        b_scale_ptr.dtype.element_ty,
        [N, K // VEC],
        scale_layout,
    )
    a_scale_tmem.store(a_scales)
    b_scale_tmem.store(b_scales)

    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout([M, N], col_stride=1)
    acc = blackwell.allocate_tensor_memory(gl.float32, [M, N], acc_layout)
    blackwell.tcgen05_mma_scaled(
        a_smem,
        b_smem,
        acc,
        a_scale_tmem,
        b_scale_tmem,
        A_TYPE,
        B_TYPE,
        use_acc=False,
    )
    result = gl.convert_layout(
        acc.load(acc.get_reg_layout(num_warps=num_warps)), layout
    )
    gl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], result)


def test_gluon_blackwell_tcgen05_scaled_mma_uses_scale_groups_on_cpu():
    torch.manual_seed(7)
    a = torch.randn(128, 4, dtype=torch.float32) / 4
    b = torch.randn(4, 16, dtype=torch.float32) / 4
    a_scale = (
        torch.arange(a.shape[0] * 2, dtype=torch.uint8).reshape(a.shape[0], 2) % 3
    ) + 126
    b_scale = (
        torch.arange(b.shape[1] * 2, dtype=torch.uint8).reshape(b.shape[1], 2) % 3
    ) + 126
    out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _tcgen05_scaled_mma_kernel
    )

    kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        out,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        2,
        "e4m3",
        "e4m3",
        num_warps=4,
    )

    a_scale_float = torch.pow(2.0, a_scale.float() - 127.0).repeat_interleave(
        2,
        dim=1,
    )
    b_scale_float = torch.pow(2.0, b_scale.float() - 127.0).repeat_interleave(
        2,
        dim=1,
    )
    expected = (a * a_scale_float) @ (b * b_scale_float.T)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
