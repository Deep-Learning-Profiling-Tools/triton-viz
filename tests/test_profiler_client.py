"""Tests for the Profiler client functionality."""

import triton
import triton.language as tl
import triton_viz
from triton_viz.core.trace import Trace


def test_profiler_decorator():
    """Test that profiler can be added via @trace decorator."""

    @triton_viz.trace("profiler")
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    # Verify the kernel is wrapped as a Trace object
    assert isinstance(simple_kernel, Trace)

    # Verify profiler client was added
    clients = simple_kernel.client_manager.clients
    assert "profiler" in clients


def test_profiler_with_sanitizer():
    """Test that profiler works alongside sanitizer client."""

    @triton_viz.trace("profiler")
    @triton_viz.trace("sanitizer")
    @triton.jit
    def kernel_with_profiler_and_sanitizer(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(x_ptr + offs)
        result = value * 2 + 3
        tl.store(y_ptr + offs, result)

    # Verify the kernel is wrapped as a Trace object
    assert isinstance(kernel_with_profiler_and_sanitizer, Trace)

    # Verify both clients were added
    clients = kernel_with_profiler_and_sanitizer.client_manager.clients
    assert "profiler" in clients
    assert "sanitizer" in clients
    assert len(clients) == 2


def test_profiler_all_clients():
    """Test that all three clients can work together."""

    @triton_viz.trace("sanitizer")
    @triton_viz.trace("profiler")
    @triton_viz.trace("tracer")
    @triton.jit
    def kernel_with_all_clients(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

    # Verify the kernel is wrapped as a Trace object
    assert isinstance(kernel_with_all_clients, Trace)

    # Verify all three clients were added
    clients = kernel_with_all_clients.client_manager.clients
    assert len(clients) == 3
    assert "sanitizer" in clients
    assert "profiler" in clients
    assert "tracer" in clients
