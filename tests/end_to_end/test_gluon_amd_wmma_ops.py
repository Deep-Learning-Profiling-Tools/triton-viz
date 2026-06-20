import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client

pytest.importorskip(
    "triton.experimental.gluon.language.amd.gfx1250",
    reason="AMD gfx1250 WMMA helpers are unavailable",
)
pytest.importorskip(
    "triton.experimental.gluon.language.amd.rdna3",
    reason="AMD RDNA3 WMMA helpers are unavailable",
)
pytest.importorskip(
    "triton.experimental.gluon.language.amd.rdna4",
    reason="AMD RDNA4 WMMA helpers are unavailable",
)


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
def _amd_wmma_kernel(
    a_ptr,
    b_ptr,
    out_gfx1250,
    out_rdna3,
    out_rdna4,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    offs_k = gl.arange(0, K, gl.SliceLayout(0, layout))
    offs_k_rows = gl.arange(0, K, gl.SliceLayout(1, layout))
    a = gl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k_rows[:, None] * N + offs_n[None, :])
    acc = gl.zeros((M, N), dtype=gl.float32, layout=layout)
    gfx1250_result = gl.amd.gfx1250.wmma(a, b, acc)
    rdna3_result = gl.amd.rdna3.wmma(a, b, acc)
    rdna4_result = gl.amd.rdna4.wmma(a, b, acc)
    gl.store(out_gfx1250 + offs_m[:, None] * N + offs_n[None, :], gfx1250_result)
    gl.store(out_rdna3 + offs_m[:, None] * N + offs_n[None, :], rdna3_result)
    gl.store(out_rdna4 + offs_m[:, None] * N + offs_n[None, :], rdna4_result)


def test_gluon_amd_wmma_ops_match_cpu_results():
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2) - 2
    out_gfx1250 = torch.empty((2, 2), dtype=torch.float32)
    out_rdna3 = torch.empty((2, 2), dtype=torch.float32)
    out_rdna4 = torch.empty((2, 2), dtype=torch.float32)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_amd_wmma_kernel)

    kernel[(1,)](
        a,
        b,
        out_gfx1250,
        out_rdna3,
        out_rdna4,
        2,
        2,
        4,
        num_warps=1,
    )

    expected = a @ b
    torch.testing.assert_close(out_gfx1250, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_rdna3, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_rdna4, expected, atol=0, rtol=0)
