"""End-to-end tests for Sanitizer(compile=True).

The analyzer-level path (parse + Z3) runs without a GPU; the trace-level
tests need a CUDA driver for the warmup compilation and are skipped without
one.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.compiled.client import CompiledSanitizer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warmup compilation needs a CUDA driver"
)


def test_factory_dispatch():
    eager = Sanitizer()
    compiled = Sanitizer(compile=True)
    assert type(eager).__name__ == "SymbolicSanitizer"
    assert isinstance(compiled, CompiledSanitizer)
    # compile kwarg must not leak into __init__ and must honor abort flag.
    assert Sanitizer(compile=True, abort_on_error=False).abort_on_error is False


def test_disable_flag_overrides_compile_mode():
    """ENABLE_SANITIZER=0 must disable compiled mode too: Sanitizer(compile=
    True) collapses to NullSanitizer (so trace() leaves the kernel
    untraced), instead of warming up and aborting on OOB."""
    from triton_viz.core.config import config as cfg

    saved = cfg.enable_sanitizer
    try:
        cfg.enable_sanitizer = False
        off = Sanitizer(compile=True, abort_on_error=False)
        assert type(off).__name__ == "NullSanitizer"
        cfg.enable_sanitizer = True
        on = Sanitizer(compile=True, abort_on_error=False)
        assert isinstance(on, CompiledSanitizer)
    finally:
        cfg.enable_sanitizer = saved


@requires_cuda
def test_correct_kernel_is_proven_in_bounds():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 4096
    x, y, out = torch.randn(n), torch.randn(n), torch.empty(n)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    assert det.last_status == "ok"
    assert det.records == []


@requires_cuda
def test_unmasked_tail_oob_is_reported():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)  # missing mask -> OOB on a ragged tail
        tl.store(out_ptr + offs, x)

    n = 3000  # not a multiple of BLOCK -> grid=3 covers [0,3072)
    x, out = torch.randn(n), torch.empty(n)
    add_nomask[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    assert det.last_status == "ok"
    assert len(det.records) >= 1
    kinds = {r.op_type.__name__ for r in det.records}
    assert "Load" in kinds
    # Report carries the offending source line and the kernel function name.
    tb = det.records[0].user_code_tracebacks[0]
    assert tb.func_name == "add_nomask"
    assert "tl.load" in tb.line_of_code


@requires_cuda
def test_abort_on_error_raises_system_exit():
    det = Sanitizer(compile=True, abort_on_error=True)

    @triton_viz.trace(det)
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, x)

    n = 3000
    x, out = torch.randn(n), torch.empty(n)
    with pytest.raises(SystemExit):
        add_nomask[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)


@requires_cuda
def test_indirect_gather_is_unsupported_not_silent():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def gather(idx_ptr, src_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask)
        vals = tl.load(src_ptr + idx, mask=mask)  # data-dependent address
        tl.store(out_ptr + offs, vals, mask=mask)

    n = 1024
    idx = torch.zeros(n, dtype=torch.int32)
    src, out = torch.randn(n), torch.empty(n)
    gather[(triton.cdiv(n, 256),)](idx, src, out, n, BLOCK=256)
    assert det.last_status == "unsupported"
    assert det.unsupported_reason is not None
    assert "data-dependent" in det.unsupported_reason
    assert det.records == []


@requires_cuda
def test_nested_loops_are_unsupported_not_silently_proven():
    """Nested loops carry independent induction variables the single-loop
    model can't represent. Regression: the nested-loop guard keyed on a flag
    only set at loop close, so an inner loop slipped through and the outer
    induction var leaked unbounded — a silent false "ok" proof for a real
    OOB."""
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def nested(in_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
        for i in range(0, M):
            for j in range(0, N):
                offs = (i * N + j) * BLOCK + tl.arange(0, BLOCK)
                x = tl.load(in_ptr + offs)  # no mask -> would be OOB
                tl.store(out_ptr + offs, x)

    inp, out = torch.randn(20), torch.empty(20)
    nested[(1,)](inp, out, 8, 2, BLOCK=4)  # real footprint needs numel 64
    assert det.last_status == "unsupported"
    assert "loop" in (det.unsupported_reason or "")
    assert det.records == []


@requires_cuda
def test_store_loop_without_accumulator_is_analyzed():
    """A pure side-effect loop (no yielded accumulator, so the TTIR scf.for
    has no '-> (types)') must still be recognized as a loop and checked, not
    skipped — its induction variable drives the address."""
    correct = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(correct)
    @triton.jit
    def store_loop(out_ptr, iters, BLOCK: tl.constexpr):
        for i in range(0, iters):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))

    out = torch.zeros(16)
    store_loop[(1,)](out, 4, BLOCK=4)  # 4*4 == 16, exactly fits
    assert correct.last_status == "ok"
    assert correct.records == []

    buggy = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(buggy)
    @triton.jit
    def store_loop2(out_ptr, iters, BLOCK: tl.constexpr):
        for i in range(0, iters):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))

    out2 = torch.zeros(16)
    store_loop2[(1,)](out2, 6, BLOCK=4)  # 6*4 == 24 > 16
    assert buggy.last_status == "ok"
    assert len(buggy.records) >= 1


@requires_cuda
def test_second_launch_recomputes_per_launch_metadata():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, x)

    # First launch: exact multiple -> in bounds.
    n1 = 4096
    x1, out1 = torch.randn(n1), torch.empty(n1)
    add_nomask[(triton.cdiv(n1, 1024),)](x1, out1, n1, BLOCK=1024)
    assert det.last_status == "ok"
    assert det.records == []

    # Second launch: ragged tail -> OOB. Per-launch metadata must be fresh.
    n2 = 3000
    x2, out2 = torch.randn(n2), torch.empty(n2)
    add_nomask[(triton.cdiv(n2, 1024),)](x2, out2, n2, BLOCK=1024)
    assert det.last_status == "ok"
    assert len(det.records) >= 1
