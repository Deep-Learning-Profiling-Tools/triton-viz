import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz


@gluon.jit
def _range_memcpy_kernel(in_ptr, out_ptr, xnumel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * BLOCK
    end = min(start + BLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


@gluon.jit
def _masked_1d_memcpy_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    BLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    pid = gl.program_id(0)
    offsets = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offsets < xnumel
    value = gl.load(in_ptr + offsets, mask=mask, other=0.0)
    gl.store(out_ptr + offsets, value, mask=mask)


@gluon.jit
def _masked_2d_memcpy_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_in,
    ystride_in,
    xstride_out,
    ystride_out,
    layout: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    offsets_x = start_x + gl.arange(
        0,
        XBLOCK,
        layout=gl.SliceLayout(dim=1, parent=layout),
    )
    offsets_y = start_y + gl.arange(
        0,
        YBLOCK,
        layout=gl.SliceLayout(dim=0, parent=layout),
    )
    in_offsets = xstride_in * offsets_x[:, None] + ystride_in * offsets_y[None, :]
    out_offsets = xstride_out * offsets_x[:, None] + ystride_out * offsets_y[None, :]
    mask = (offsets_x[:, None] < xnumel) & (offsets_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    gl.store(out_ptr + out_offsets, value, mask=mask)


@gluon.jit
def _converted_layout_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_out,
    ystride_out,
    layout_in: gl.constexpr,
    layout_out: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid = gl.program_id(0)
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout_in))
    yoffs = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout_in))
    mask = (xoffs[:, None] < xnumel) & (yoffs[None, :] < ynumel)
    a_offsets = xstride_a * xoffs[:, None] + ystride_a * yoffs[None, :]
    b_offsets = xstride_b * xoffs[:, None] + ystride_b * yoffs[None, :]
    out_offsets = xstride_out * xoffs[:, None] + ystride_out * yoffs[None, :]
    values = gl.load(a_ptr + a_offsets, mask=mask, other=0.0) + gl.load(
        b_ptr + b_offsets,
        mask=mask,
        other=0.0,
    )
    gl.store(out_ptr + out_offsets, gl.convert_layout(values, layout_out), mask=mask)


def test_gluon_core_ops_run_scalar_range_memcpy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_range_memcpy_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert kernel.client_manager.launch.grid == (1, 1, 1)


def test_gluon_core_ops_run_masked_1d_memcpy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    layout = gl.BlockedLayout([1], [32], [1], [0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(_masked_1d_memcpy_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, layout, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_core_ops_run_masked_2d_memcpy_on_cpu():
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    out = torch.full_like(inp, -1)
    layout = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(_masked_2d_memcpy_kernel)

    kernel[(1, 1)](
        inp,
        out,
        *inp.shape,
        *inp.stride(),
        *out.stride(),
        layout,
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_core_ops_run_converted_layout_elementwise_add_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    out = torch.full_like(a, -1)
    layout_in = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    layout_out = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(_converted_layout_add_kernel)

    kernel[(1,)](
        a,
        b,
        out,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        layout_in,
        layout_out,
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, a + b, atol=0, rtol=0)
