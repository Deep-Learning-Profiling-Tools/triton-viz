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
