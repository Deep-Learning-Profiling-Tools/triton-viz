import numpy as np
import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client

pytest.importorskip(
    "triton.experimental.gluon.language.amd.cdna3",
    reason="AMD CDNA3 helpers are unavailable",
)
pytest.importorskip(
    "triton.experimental.gluon.language.amd.cdna4",
    reason="AMD CDNA4 helpers are unavailable",
)
pytest.importorskip(
    "triton.experimental.gluon.language.amd.gfx1250",
    reason="AMD gfx1250 helpers are unavailable",
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
def _amd_scaled_upcast_kernel(
    src_ptr,
    scale_ptr,
    out_cdna3,
    out_cdna4,
    out_gfx1250,
    ROWS: gl.constexpr,
    PACKED_COLS: gl.constexpr,
    COLS: gl.constexpr,
):
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    packed_rows = gl.arange(0, ROWS, gl.SliceLayout(1, packed_layout))
    packed_cols = gl.arange(0, PACKED_COLS, gl.SliceLayout(0, packed_layout))
    rows = gl.arange(0, ROWS, gl.SliceLayout(1, scale_layout))
    cols = gl.arange(0, COLS, gl.SliceLayout(0, scale_layout))
    src = gl.load(src_ptr + packed_rows[:, None] * PACKED_COLS + packed_cols[None, :])
    scale = gl.load(scale_ptr + rows[:, None] * COLS + cols[None, :])

    cdna3 = gl.amd.cdna3.scaled_upcast(src, scale, gl.float16, axis=1)
    cdna4 = gl.amd.cdna4.scaled_upcast(src, scale, gl.float16, axis=1)
    gfx1250 = gl.amd.gfx1250.scaled_upcast(src, scale, gl.float16, axis=1)
    offsets = rows[:, None] * COLS + cols[None, :]
    gl.store(out_cdna3 + offsets, cdna3)
    gl.store(out_cdna4 + offsets, cdna4)
    gl.store(out_gfx1250 + offsets, gfx1250)


def _unpack_e2m1_cpu(packed: torch.Tensor) -> torch.Tensor:
    packed_np = packed.numpy()
    low = packed_np & np.uint8(0x0F)
    high = packed_np >> np.uint8(4)
    unpacked = np.empty((packed_np.shape[0], packed_np.shape[1] * 2), dtype=np.uint8)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    positive_lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    values = positive_lut[unpacked & np.uint8(0x07)]
    signs = (unpacked & np.uint8(0x08)) != 0
    return torch.from_numpy(np.where(signs, -values, values))


def test_gluon_amd_scaled_upcast_ops_match_cpu_results():
    src = torch.tensor([[0x21, 0x83], [0x74, 0xA5]], dtype=torch.uint8)
    scale = torch.tensor(
        [[127, 128, 126, 129], [127, 127, 128, 126]], dtype=torch.uint8
    )
    out_cdna3 = torch.empty((2, 4), dtype=torch.float16)
    out_cdna4 = torch.empty((2, 4), dtype=torch.float16)
    out_gfx1250 = torch.empty((2, 4), dtype=torch.float16)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(
        _amd_scaled_upcast_kernel
    )

    kernel[(1,)](
        src,
        scale,
        out_cdna3,
        out_cdna4,
        out_gfx1250,
        src.shape[0],
        src.shape[1],
        scale.shape[1],
        num_warps=1,
    )

    scale_values = torch.pow(2.0, scale.to(torch.float32) - 127.0)
    expected = (_unpack_e2m1_cpu(src) * scale_values).to(torch.float16)
    torch.testing.assert_close(out_cdna3, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_cdna4, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_gfx1250, expected, atol=0, rtol=0)
