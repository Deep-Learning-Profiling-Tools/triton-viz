import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer, Profiler, Tracer
from triton_viz import config as cfg


# Make sure sanitizer is on.
cfg.sanitizer_backend = "symexec"

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
    @triton_viz.trace(("sanitizer", "profiler"))
    @triton_viz.trace("tracer")
    @triton_viz.trace(("sanitizer",))    # Duplicate Sanitizer (should be ignored)
    @triton.jit
    def my_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offs,
                 tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

    # Should be wrapped as a Trace object.
    from triton_viz.core.trace import Trace
    assert isinstance(my_kernel, Trace)

    # Verify client de-duplication and addition logic
    clients = my_kernel.client_manager.clients
    assert len(clients) == 3
    assert sum(isinstance(c, Sanitizer) for c in clients) == 1
    assert sum(isinstance(c, Profiler) for c in clients) == 1
    assert sum(isinstance(c, Tracer) for c in clients) == 1
