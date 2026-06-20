import numpy as np
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
def _fp4_to_fp_kernel(
    src_ptr,
    out_ptr,
    ROWS: gl.constexpr,
    PACKED_COLS: gl.constexpr,
    COLS: gl.constexpr,
):
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    packed_rows = gl.arange(0, ROWS, gl.SliceLayout(1, packed_layout))
    packed_cols = gl.arange(0, PACKED_COLS, gl.SliceLayout(0, packed_layout))
    rows = gl.arange(0, ROWS, gl.SliceLayout(1, out_layout))
    cols = gl.arange(0, COLS, gl.SliceLayout(0, out_layout))
    src = gl.load(src_ptr + packed_rows[:, None] * PACKED_COLS + packed_cols[None, :])
    result = gl.fp4_to_fp(src, gl.float16, axis=1)
    gl.store(out_ptr + rows[:, None] * COLS + cols[None, :], result)


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
    return torch.from_numpy(np.where(signs, -values, values)).to(torch.float16)


def test_gluon_fp4_to_fp_matches_cpu_results():
    src = torch.tensor([[0x21, 0x83], [0x74, 0xA5]], dtype=torch.uint8)
    out = torch.empty((2, 4), dtype=torch.float16)
    kernel = triton_viz.trace(_NoOpClient(), frontend="gluon")(_fp4_to_fp_kernel)

    kernel[(1,)](
        src,
        out,
        src.shape[0],
        src.shape[1],
        src.shape[1] * 2,
        num_warps=1,
    )

    torch.testing.assert_close(out, _unpack_e2m1_cpu(src), atol=0, rtol=0)
