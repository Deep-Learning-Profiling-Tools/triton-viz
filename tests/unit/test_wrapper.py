from unittest.mock import MagicMock, patch

from triton_viz.wrapper import (
    create_patched_jit,
    create_patched_autotune,
    sanitizer_wrapper,
    profiler_wrapper,
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
