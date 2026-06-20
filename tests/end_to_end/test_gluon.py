import importlib.util
from pathlib import Path

import pytest
import torch
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.clients.sanitizer.sanitizer import SymbolicSanitizer
from triton_viz.core.data import Load

try:
    from triton.experimental.gluon.language.amd.cdna4 import (
        async_copy as amd_cdna4_cp,
    )
except ImportError:
    amd_cdna4_cp = None

try:
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
except ImportError:
    cp = None

_HAS_AMPERE_ASYNC_COPY = (
    cp is not None and getattr(cp, "async_copy_global_to_local", None) is not None
)
_HAS_AMD_CDNA4_ASYNC_COPY = (
    amd_cdna4_cp is not None
    and getattr(amd_cdna4_cp, "global_load_to_shared", None) is not None
)


@gluon.jit
def _copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


@gluon.jit
def _oob_scalar_offset_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr + 1)
    gl.store(out_ptr, value)


@gluon.jit
def _masked_safe_kernel(
    in_ptr,
    out_ptr,
    n: gl.constexpr,
    BLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    offs = gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    value = gl.load(in_ptr + offs, mask=mask, other=0.0)
    gl.store(out_ptr + offs, value, mask=mask)


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


@gluon.jit
def _async_copy_1d_kernel(in_ptr, out_ptr, xnumel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offsets < xnumel
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], layout=smem_layout)

    cp.async_copy_global_to_local(smem, in_ptr + offsets, mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    gl.store(out_ptr + offsets, smem.load(layout), mask=mask)


@gluon.jit
def _amd_async_copy_other_kernel(in_ptr, out_ptr, xnumel, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    mask = offsets < xnumel
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], layout=smem_layout)

    amd_cdna4_cp.global_load_to_shared(
        smem,
        in_ptr + offsets,
        mask=mask,
        other=-7.0,
    )
    amd_cdna4_cp.wait_group(0)
    values = smem.load(layout)
    gl.store(out_ptr + offsets, values)


@gluon.jit
def _async_copy_elementwise_add_kernel(
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
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    smem_layout: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yoffs = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
    a_smem = gl.allocate_shared_memory(gl.float32, [XBLOCK, YBLOCK], smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float32, [XBLOCK, YBLOCK], smem_layout)

    cp.async_copy_global_to_local(
        a_smem,
        a_ptr + xstride_a * xoffs[:, None] + ystride_a * yoffs[None, :],
        mask=mask,
    )
    cp.async_copy_global_to_local(
        b_smem,
        b_ptr + xstride_b * xoffs[:, None] + ystride_b * yoffs[None, :],
        mask=mask,
    )
    cp.commit_group()
    cp.wait_group(0)
    values = a_smem.load(layout) + b_smem.load(layout)
    gl.store(
        out_ptr + xstride_out * xoffs[:, None] + ystride_out * yoffs[None, :],
        values,
        mask=mask,
    )


def test_gluon_trace_runs_copy_scalar_kernel():
    kernel = triton_viz.trace("tracer", frontend="gluon")(_copy_scalar_kernel)

    inp = torch.tensor([42.0])
    out = torch.empty_like(inp)
    kernel[(1,)](inp, out, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert kernel.client_manager.launch.grid == (1, 1, 1)
    assert len(kernel.client_manager.launch.records) >= 1


def test_gluon_sanitizer_allows_in_bounds_kernel():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(
        client=sanitizer,
        frontend="gluon",
    )(_copy_scalar_kernel)

    inp = torch.tensor([42.0])
    out = torch.empty_like(inp)
    ret = kernel[(1,)](inp, out, num_warps=1)

    assert ret is None
    assert sanitizer.records == []


def test_gluon_sanitizer_reports_real_oob_load_kernel(monkeypatch):
    monkeypatch.setattr(knobs.compilation, "always_compile", True)
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(
        client=sanitizer,
        frontend="gluon",
    )(_oob_scalar_offset_kernel)
    kernel.fn.device_caches.clear()

    inp = torch.tensor([42.0])
    out = torch.empty_like(inp)
    ret = kernel[(1,)](inp, out, num_warps=1)

    assert ret is None
    assert sanitizer.records
    assert sanitizer.records[0].op_type is Load
    assert (
        sanitizer.records[0].violation_address
        == sanitizer.records[0].tensor.data_ptr() + inp.element_size()
    )


def test_gluon_sanitizer_allows_masked_in_bounds_kernel():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(
        client=sanitizer,
        frontend="gluon",
    )(_masked_safe_kernel)

    inp = torch.arange(4, dtype=torch.float32)
    out = torch.empty((8,), dtype=torch.float32)
    layout = gl.BlockedLayout([1], [32], [1], [0])
    ret = kernel[(1,)](inp, out, 4, 8, layout, num_warps=1)

    assert ret is None
    assert sanitizer.records == []


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


@pytest.mark.skipif(
    not _HAS_AMPERE_ASYNC_COPY,
    reason="Gluon Ampere async copy global-to-local builtin is unavailable",
)
def test_gluon_async_copy_runs_masked_1d_copy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_async_copy_1d_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


@pytest.mark.skipif(
    not _HAS_AMD_CDNA4_ASYNC_COPY,
    reason="Gluon AMD CDNA4 async copy builtins are unavailable",
)
def test_gluon_amd_async_copy_preserves_masked_other_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full((64,), -1, dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_async_copy_other_kernel)

    kernel[(1,)](inp, out, inp.numel(), out.numel(), num_warps=4)

    expected = torch.cat([inp, torch.full((24,), -7, dtype=torch.float32)])
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not _HAS_AMPERE_ASYNC_COPY,
    reason="Gluon Ampere async copy global-to-local builtin is unavailable",
)
def test_gluon_async_copy_runs_staged_elementwise_add_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    out = torch.full_like(a, -1)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _async_copy_elementwise_add_kernel
    )

    kernel[(1,)](
        a,
        b,
        out,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        8,
        8,
        smem_layout,
        num_warps=4,
    )

    torch.testing.assert_close(out, a + b, atol=0, rtol=0)


def test_gluon_tma_oob_example_reports(capsys):
    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sanitizer"
        / "gluon_tma_oob.py"
    )
    spec = importlib.util.spec_from_file_location("gluon_tma_oob_example", example_path)
    assert spec is not None
    assert spec.loader is not None
    gluon_tma_oob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gluon_tma_oob)

    with pytest.raises(SystemExit):
        gluon_tma_oob.run()

    output = capsys.readouterr().out
    assert "ILLEGAL MEMORY ACCESS" in output
    assert "gluon_tma_oob.py" in output
    assert "descriptor_access" in output
