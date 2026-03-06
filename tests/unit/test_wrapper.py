import pytest
from unittest.mock import MagicMock, patch

import triton_viz
from triton_viz.core.config import config as cfg
from triton_viz.core.trace import TraceInterface
from triton_viz.wrapper import (
    create_patched_jit,
    create_patched_autotune,
    sanitizer_wrapper,
    profiler_wrapper,
    apply_sanitizer,
    apply_profiler,
)


# ======== Wrapper Function Tests ===========


def test_sanitizer_wrapper_applies_trace():
    mock_kernel = MagicMock()
    mock_kernel.__name__ = "test_kernel"

    with patch("triton_viz.wrapper.triton_viz.trace") as mock_trace:
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = "wrapped_kernel"

        result = sanitizer_wrapper(mock_kernel)

        mock_trace.assert_called_once()
        mock_decorator.assert_called_once_with(mock_kernel)
        assert result == "wrapped_kernel"


def test_profiler_wrapper_applies_trace():
    mock_kernel = MagicMock()
    mock_kernel.__name__ = "test_kernel"

    with patch("triton_viz.wrapper.triton_viz.trace") as mock_trace:
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = "wrapped_kernel"

        result = profiler_wrapper(mock_kernel)

        mock_trace.assert_called_once()
        mock_decorator.assert_called_once_with(mock_kernel)
        assert result == "wrapped_kernel"


# ======== create_patched_jit Tests ===========


def test_create_patched_jit_direct_decorator():
    """Test @triton.jit without parentheses"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_original_jit = MagicMock(return_value="jit_kernel")

    with patch("triton_viz.wrapper._original_jit", mock_original_jit):
        patched_jit = create_patched_jit(mock_wrapper)

        mock_fn = MagicMock()
        result = patched_jit(mock_fn)

        mock_original_jit.assert_called_once_with(mock_fn)
        mock_wrapper.assert_called_once_with("jit_kernel")
        assert result == "final_kernel"


def test_create_patched_jit_with_kwargs():
    """Test @triton.jit(**opts) with parentheses"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_jit_decorator = MagicMock(return_value="jit_kernel")
    mock_original_jit = MagicMock(return_value=mock_jit_decorator)

    with patch("triton_viz.wrapper._original_jit", mock_original_jit):
        patched_jit = create_patched_jit(mock_wrapper)

        decorator = patched_jit(do_not_specialize=["n"])
        assert callable(decorator)

        mock_fn = MagicMock()
        result = decorator(mock_fn)

        mock_original_jit.assert_called_once_with(do_not_specialize=["n"])
        mock_jit_decorator.assert_called_once_with(mock_fn)
        mock_wrapper.assert_called_once_with("jit_kernel")
        assert result == "final_kernel"


# ======== create_patched_autotune Tests ===========


def test_create_patched_autotune_with_kwargs():
    """Test @triton.autotune(**opts) with parentheses"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_autotune_decorator = MagicMock(return_value="autotune_kernel")
    mock_original_autotune = MagicMock(return_value=mock_autotune_decorator)

    with patch("triton_viz.wrapper._original_autotune", mock_original_autotune):
        patched_autotune = create_patched_autotune(mock_wrapper)

        decorator = patched_autotune(configs=[], key=["n"])
        assert callable(decorator)

        mock_fn = MagicMock()
        result = decorator(mock_fn)

        mock_original_autotune.assert_called_once_with(configs=[], key=["n"])
        mock_autotune_decorator.assert_called_once_with(mock_fn)
        mock_wrapper.assert_called_once_with("autotune_kernel")
        assert result == "final_kernel"


def test_create_patched_autotune_direct_decorator():
    """Test @triton.autotune without parentheses (if fn is passed directly)"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_original_autotune = MagicMock(return_value="autotune_kernel")

    with patch("triton_viz.wrapper._original_autotune", mock_original_autotune):
        patched_autotune = create_patched_autotune(mock_wrapper)

        mock_fn = MagicMock()
        result = patched_autotune(mock_fn)

        mock_original_autotune.assert_called_once_with(mock_fn)
        mock_wrapper.assert_called_once_with("autotune_kernel")
        assert result == "final_kernel"


# ======== CLI Active Guard Tests ===========


def test_trace_decorator_raises_when_cli_active():
    """trace() should raise RuntimeError on an already-wrapped kernel when CLI is active."""
    original = cfg.cli_active
    try:
        cfg.cli_active = True
        mock_kernel = MagicMock(spec=TraceInterface)
        decorator = triton_viz.trace("tracer")
        with pytest.raises(RuntimeError, match="CLI wrapper"):
            decorator(mock_kernel)
    finally:
        cfg.cli_active = original


def test_trace_decorator_allows_cli_own_wrapping():
    """trace() called by the CLI on a raw kernel should NOT raise, even when cli_active."""
    from triton.runtime.interpreter import InterpretedFunction

    original = cfg.cli_active
    try:
        cfg.cli_active = True
        # Simulate a raw kernel (not already wrapped by TraceInterface)
        mock_kernel = MagicMock(spec=InterpretedFunction)
        mock_kernel.fn = lambda: None
        mock_kernel.arg_names = []
        decorator = triton_viz.trace("sanitizer")
        # Should not raise — this is the CLI's own first-time wrapping
        result = decorator(mock_kernel)
        assert isinstance(result, TraceInterface)
    finally:
        cfg.cli_active = original


def test_trace_decorator_works_when_cli_inactive():
    """trace() should work normally when CLI is not active."""
    from triton.runtime.interpreter import InterpretedFunction

    original = cfg.cli_active
    try:
        cfg.cli_active = False
        mock_kernel = MagicMock(spec=InterpretedFunction)
        mock_kernel.fn = lambda: None
        mock_kernel.arg_names = []
        decorator = triton_viz.trace("tracer")
        result = decorator(mock_kernel)
        assert isinstance(result, TraceInterface)
    finally:
        cfg.cli_active = original


def test_apply_wrapper_rejects_non_cli_invocation():
    """apply_sanitizer/apply_profiler should raise when called from Python, not CLI."""
    with pytest.raises(RuntimeError, match="must be used as a CLI tool"):
        apply_sanitizer()
    with pytest.raises(RuntimeError, match="must be used as a CLI tool"):
        apply_profiler()
