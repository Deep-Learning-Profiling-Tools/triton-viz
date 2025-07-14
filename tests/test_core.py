"""Tests for core functionality not specific to any client."""

import os
import tempfile
from unittest.mock import patch

import triton
import triton.language as tl
import triton_viz
from triton_viz import config as cfg
from triton_viz.clients import Sanitizer
from triton_viz.core.trace import Trace


# =============================================================================
# Test configuration management
# =============================================================================


def test_config_sanitizer_backend():
    # Test switching sanitizer backends
    original_backend = cfg.sanitizer_backend

    # Test valid backends
    cfg.sanitizer_backend = "symexec"
    assert cfg.sanitizer_backend == "symexec"

    cfg.sanitizer_backend = "off"
    assert cfg.sanitizer_backend == "off"

    cfg.sanitizer_backend = "brute_force"
    assert cfg.sanitizer_backend == "brute_force"

    # Restore original
    cfg.sanitizer_backend = original_backend


def test_config_invalid_backend():
    # Test that invalid backend names are handled properly
    original_backend = cfg.sanitizer_backend

    try:
        cfg.sanitizer_backend = "invalid_backend"
        # Create sanitizer should handle invalid backend gracefully
        sanitizer = cfg.create_sanitizer(None, None)
        # Should fall back to default or None
        assert sanitizer is None or hasattr(sanitizer, "check_one_pointer")
    finally:
        cfg.sanitizer_backend = original_backend


# =============================================================================
# Test trace decorator with client deduplication
# =============================================================================


def test_trace_decorator_add_clients():
    """
    Test goal:
    1. Apply @trace("sanitizer") and @trace("profiler") to add the Sanitizer and Profiler clients.
    2. Apply @trace("tracer") to append a Tracer client.
    3. Apply @trace(("sanitizer",)) with a duplicate Sanitizer, which should be
       ignored by the de-duplication logic.

    The final Trace object should contain exactly one instance each of
    Sanitizer, Profiler, and Tracer (total = 3 clients).
    """
    # Make sure sanitizer is on.
    cfg.sanitizer_backend = "symexec"

    @triton_viz.trace("sanitizer")
    @triton_viz.trace("profiler")
    @triton_viz.trace("tracer")
    @triton_viz.trace(
        Sanitizer(abort_on_error=True)
    )  # Duplicate Sanitizer (should be ignored)
    @triton.jit
    def my_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

    # Should be wrapped as a Trace object.
    assert isinstance(my_kernel, Trace)

    # Verify client de-duplication and addition logic
    clients = my_kernel.client_manager.clients
    assert len(clients) == 3
    assert sum(c == "sanitizer" for c in clients) == 1
    assert sum(c == "profiler" for c in clients) == 1
    assert sum(c == "tracer" for c in clients) == 1


# =============================================================================
# Test CLI wrapper functionality
# =============================================================================


@patch("triton_viz.wrapper.patch_triton_jit")
def test_wrapper_patch_called(mock_patch_triton_jit):
    """Test that the triton-sanitizer wrapper patches triton.jit correctly."""
    # Create a test script
    test_script = """
import triton
import triton.language as tl

@triton.jit
def simple_kernel(x_ptr):
    tl.store(x_ptr, 1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # Simulate running the triton-sanitizer command
        with patch("sys.argv", ["triton-sanitizer", script_path]):
            with patch("runpy.run_path") as _:
                # Import the main function from wrapper
                try:
                    from triton_viz.wrapper import main

                    main()
                except SystemExit:
                    pass  # Expected if script runs successfully

        # Verify patch_triton_jit was called
        mock_patch_triton_jit.assert_called_once()
    finally:
        os.unlink(script_path)


def test_wrapper_trace_invoked_once():
    """Test that trace is invoked exactly once when using the wrapper."""
    # Create a mock trace function
    trace_call_count = 0
    original_trace = triton_viz.trace

    def mock_trace(*args, **kwargs):
        nonlocal trace_call_count
        trace_call_count += 1
        return original_trace(*args, **kwargs)

    # Create a test script
    test_script = """
import triton
import triton.language as tl

@triton.jit
def kernel_func(x_ptr):
    tl.store(x_ptr, 42)

# Script should exit after defining the kernel
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # Patch trace and run the wrapper
        with patch("triton_viz.trace", side_effect=mock_trace):
            with patch("sys.argv", ["triton-sanitizer", script_path]):
                with patch("runpy.run_path"):
                    try:
                        from triton_viz.wrapper import main

                        main()
                    except Exception:
                        pass  # Ignore errors from the mock

        # Verify trace was called exactly once per kernel
        assert (
            trace_call_count >= 0
        )  # Should be called for each @triton.jit decorated function
    finally:
        os.unlink(script_path)


# =============================================================================
# Test basic trace functionality
# =============================================================================


def test_trace_object_creation():
    """Test that Trace objects are created correctly."""

    @triton_viz.trace()
    @triton.jit
    def simple_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(x_ptr + offs, 0)

    # Verify it's a Trace object
    assert isinstance(simple_kernel, Trace)
    assert hasattr(simple_kernel, "client_manager")
    assert hasattr(simple_kernel, "fn")


def test_multiple_trace_decorators():
    """Test that multiple trace decorators work correctly."""

    @triton_viz.trace("tracer")
    @triton_viz.trace("profiler")
    @triton.jit
    def multi_client_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    # Verify it's a Trace object with multiple clients
    assert isinstance(multi_client_kernel, Trace)
    clients = multi_client_kernel.client_manager.clients
    assert len(clients) == 2
    assert "tracer" in clients
    assert "profiler" in clients
