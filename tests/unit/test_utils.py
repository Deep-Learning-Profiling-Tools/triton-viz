from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from z3 import Int

from triton_viz.clients.sanitizer.sanitizer import SymbolicSanitizer
from triton_viz.clients.symbolic_engine import LoopContext
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.data import Load
from triton_viz.core.trace import trace_source
from triton_viz.utils import traceback_utils
from triton_viz.utils.traceback_utils import (
    TracebackInfo,
    extract_user_frames,
    location_to_traceback_info,
)


class DummyExpr:
    """Minimal symbolic expression stub for sanitizer source-location tests."""

    op = "load"

    def eval(self):
        """Return a stable z3 address expression with no extra constraints."""
        return Int("addr"), None


def _identity(fn):
    """Return function unchanged for optional decorator wiring."""
    return fn


@pytest.fixture(autouse=True)
def isolate_active_code_keys():
    """Keep active trace-source keys isolated per test."""
    keys = traceback_utils.CODE_KEYS
    original = set(keys)
    keys.clear()
    yield
    keys.clear()
    keys.update(original)


def _capture_sanitizer_traceback(
    with_oob_trace_source: bool, use_decorator: bool
) -> TracebackInfo:
    """Capture deferred sanitizer source location from an oob helper call."""
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    sanitizer.loop_stack.append(
        LoopContext(
            lineno=1,
            length=1,
            idx=0,
            idx_z3=Int("loop_idx"),
            start=0,
            stop=1,
            step=1,
        )
    )
    expr = DummyExpr()

    oob_trace = trace_source if with_oob_trace_source else _identity
    decorate_oob = oob_trace if use_decorator else _identity
    decorate_gemm = trace_source if use_decorator else _identity

    @decorate_oob
    def oob():
        sanitizer._handle_access_check(expr)

    @decorate_gemm
    def gemm_kernel():
        oob()

    if not use_decorator:
        oob = oob_trace(oob)
        gemm_kernel = trace_source(gemm_kernel)

    gemm_kernel()

    pending_check = sanitizer.loop_stack[-1].pending_checks[0]
    assert pending_check.source_location is not None
    return location_to_traceback_info(pending_check.source_location)


def _capture_tracer_frame(
    with_oob_trace_source: bool, use_decorator: bool
) -> TracebackInfo:
    """Capture tracer call path for a load issued through an oob helper call."""
    tracer = Tracer()

    tensor = MagicMock()
    tensor.data_ptr = MagicMock(return_value=4096)
    tracer.arg_callback("x", tensor, None)
    tracer.grid_callback((1, 1, 1))

    load_callback = tracer.register_op_callback(Load).before_callback
    assert load_callback is not None

    ptr = SimpleNamespace(data=np.array([4096], dtype=np.int64))
    mask = SimpleNamespace(data=np.array([True]))

    oob_trace = trace_source if with_oob_trace_source else _identity
    decorate_oob = oob_trace if use_decorator else _identity
    decorate_gemm = trace_source if use_decorator else _identity

    @decorate_oob
    def oob():
        load_callback(ptr, mask, None)

    @decorate_gemm
    def gemm_kernel():
        oob()

    if not use_decorator:
        oob = oob_trace(oob)
        gemm_kernel = trace_source(gemm_kernel)

    gemm_kernel()

    record = tracer.records[-1]
    return record.call_path[0]


@pytest.mark.parametrize("use_decorator", [True, False])
def test_sanitizer_trace_source_uses_oob_line(use_decorator: bool):
    """Sanitizer should attribute deferred checks to oob helper when traced."""
    tb_info = _capture_sanitizer_traceback(
        with_oob_trace_source=True, use_decorator=use_decorator
    )
    assert "oob" in tb_info.func_name
    assert "_handle_access_check" in tb_info.line_of_code


@pytest.mark.parametrize("use_decorator", [True, False])
def test_sanitizer_without_oob_trace_source_uses_oob_line(use_decorator: bool):
    """Boundary-marker: undecorated helpers are auto-captured, same as decorated."""
    tb_info = _capture_sanitizer_traceback(
        with_oob_trace_source=False, use_decorator=use_decorator
    )
    # With boundary-marker, oob is captured even without @trace_source
    assert "oob" in tb_info.func_name
    assert "_handle_access_check" in tb_info.line_of_code


@pytest.mark.parametrize("use_decorator", [True, False])
def test_tracer_trace_source_uses_oob_line(use_decorator: bool):
    """Tracer should record oob helper as innermost user frame when traced."""
    tb_info = _capture_tracer_frame(
        with_oob_trace_source=True, use_decorator=use_decorator
    )
    assert tb_info.func_name.endswith("oob")
    assert "load_callback" in tb_info.line_of_code


@pytest.mark.parametrize("use_decorator", [True, False])
def test_tracer_without_oob_trace_source_uses_oob_line(use_decorator: bool):
    """Boundary-marker: undecorated helpers are auto-captured, same as decorated."""
    tb_info = _capture_tracer_frame(
        with_oob_trace_source=False, use_decorator=use_decorator
    )
    # With boundary-marker, oob is captured even without @trace_source
    assert tb_info.func_name.endswith("oob")
    assert "load_callback" in tb_info.line_of_code


def test_undecorated_helper_captured_via_boundary_marker():
    """Verify that a completely undecorated helper between framework code and
    the @trace-registered kernel is automatically included in extract_user_frames."""
    captured: list[TracebackInfo] = []

    # helper_a calls helper_b which captures frames — neither is decorated
    def helper_b():
        captured.extend(extract_user_frames())

    def helper_a():
        helper_b()

    # Only the kernel is registered via trace_source (as @trace would do)
    @trace_source
    def my_kernel():
        helper_a()

    my_kernel()

    func_names = [tb.func_name for tb in captured]
    # All three should appear: kernel, helper_a, helper_b
    assert any("my_kernel" in fn for fn in func_names)
    assert any("helper_a" in fn for fn in func_names)
    assert any("helper_b" in fn for fn in func_names)
