"""Tests for the Tracer client functionality."""

import triton
import triton.language as tl
import triton_viz
from triton_viz.core.trace import Trace


def test_tracer_decorator():
    """Test that tracer can be added via @trace decorator."""

    @triton_viz.trace("tracer")
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    # Verify the kernel is wrapped as a Trace object
    assert isinstance(simple_kernel, Trace)

    # Verify tracer client was added
    clients = simple_kernel.client_manager.clients
    assert "tracer" in clients


def test_tracer_default_decorator():
    """Test that tracer is the default client when using @trace() without arguments."""

    @triton_viz.trace()
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    # Verify the kernel is wrapped as a Trace object
    assert isinstance(simple_kernel, Trace)

    # Verify tracer client was added by default
    clients = simple_kernel.client_manager.clients
    assert "tracer" in clients


def test_tracer_multiple_decorators():
    """Test that tracer works with multiple client decorators."""

    @triton_viz.trace("tracer")
    @triton_viz.trace("profiler")
    @triton.jit
    def kernel_with_multiple_clients(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs) + 1)

    # Verify the kernel is wrapped as a Trace object
    assert isinstance(kernel_with_multiple_clients, Trace)

    # Verify both clients were added
    clients = kernel_with_multiple_clients.client_manager.clients
    assert "tracer" in clients
    assert "profiler" in clients
    assert len(clients) == 2
