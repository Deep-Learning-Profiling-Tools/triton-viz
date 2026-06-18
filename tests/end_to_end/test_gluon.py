import pytest
import torch
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton_viz
from triton_viz.clients.sanitizer.sanitizer import SymbolicSanitizer
from triton_viz.core.data import Load


def _has_cuda_device() -> bool:
    try:
        return torch.cuda.is_available()
    except RuntimeError:
        return False


def _has_tma_device() -> bool:
    if not _has_cuda_device():
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except RuntimeError:
        return False


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


@pytest.mark.skipif(not _has_cuda_device(), reason="CUDA required for Gluon execution")
def test_gluon_trace_runs_copy_scalar_kernel():
    kernel = triton_viz.trace("tracer", frontend="gluon")(_copy_scalar_kernel)

    inp = torch.tensor([42.0], device="cuda")
    out = torch.empty_like(inp)
    kernel[(1,)](inp, out, num_warps=1)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert kernel.client_manager.launch.grid == (1, 1, 1)
    assert len(kernel.client_manager.launch.records) >= 1


@pytest.mark.skipif(not _has_cuda_device(), reason="CUDA required for Gluon execution")
def test_gluon_sanitizer_allows_in_bounds_kernel():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(
        client=sanitizer,
        frontend="gluon",
    )(_copy_scalar_kernel)

    inp = torch.tensor([42.0], device="cuda")
    out = torch.empty_like(inp)
    ret = kernel[(1,)](inp, out, num_warps=1)
    torch.cuda.synchronize()

    assert ret is None
    assert sanitizer.records == []
    torch.testing.assert_close(out, inp, atol=0, rtol=0)


@pytest.mark.skipif(not _has_cuda_device(), reason="CUDA required for Gluon execution")
def test_gluon_sanitizer_reports_real_oob_load_kernel(monkeypatch):
    monkeypatch.setattr(knobs.compilation, "always_compile", True)
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(
        client=sanitizer,
        frontend="gluon",
    )(_oob_scalar_offset_kernel)
    kernel.fn.device_caches.clear()

    inp = torch.tensor([42.0], device="cuda")
    out = torch.empty_like(inp)
    ret = kernel[(1,)](inp, out, num_warps=1)
    torch.cuda.synchronize()

    assert ret is None
    assert sanitizer.records
    assert sanitizer.records[0].op_type is Load
    assert (
        sanitizer.records[0].violation_address
        == sanitizer.records[0].tensor.data_ptr() + inp.element_size()
    )


@pytest.mark.skipif(not _has_cuda_device(), reason="CUDA required for Gluon execution")
def test_gluon_sanitizer_allows_masked_in_bounds_kernel():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(
        client=sanitizer,
        frontend="gluon",
    )(_masked_safe_kernel)

    inp = torch.arange(4, dtype=torch.float32, device="cuda")
    out = torch.empty((8,), dtype=torch.float32, device="cuda")
    layout = gl.BlockedLayout([1], [32], [1], [0])
    ret = kernel[(1,)](inp, out, 4, 8, layout, num_warps=1)
    torch.cuda.synchronize()

    assert ret is None
    assert sanitizer.records == []


@pytest.mark.skipif(not _has_tma_device(), reason="CUDA TMA device required")
def test_gluon_tma_oob_example_reports(capsys):
    from examples.sanitizer import gluon_tma_oob

    with pytest.raises(SystemExit):
        gluon_tma_oob.run()

    output = capsys.readouterr().out
    assert "ILLEGAL MEMORY ACCESS" in output
    assert "gluon_tma_oob.py" in output
    assert "descriptor_access" in output
