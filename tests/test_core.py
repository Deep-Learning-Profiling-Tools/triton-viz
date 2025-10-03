import pytest
import torch
import sys
import tempfile
from pathlib import Path
import textwrap
import os
import subprocess

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz import config as cfg


cfg.sanitizer_backend = "symexec"


# ======== Trace Decorator Tests =========
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
    from triton_viz.core.trace import TritonTrace

    assert isinstance(my_kernel, TritonTrace)

    # Verify client de-duplication and addition logic
    clients = my_kernel.client_manager.clients
    assert len(clients) == 3
    assert sum(c == "sanitizer" for c in clients) == 1
    assert sum(c == "profiler" for c in clients) == 1
    assert sum(c == "tracer" for c in clients) == 1


# ======== Wrapper Tests =========
# Test that triton_viz.wrapper works correctly:
# 1. It should patch triton.jit / triton.language.jit /
#   triton.runtime.interpreter.jit with wrapper._patched_jit
# 2. The first use of @triton.jit must invoke triton_viz.trace(Sanitizer)
def test_cli_invocation():
    """
    Simulate running:
        $ triton-sanitizer dummy_program.py
    and assert that triton_viz.trace is invoked exactly once.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # --- 1) Create a temporary Triton script under an ad-hoc tmp_path
        (tmp_path / "dummy_program.py").write_text(
            textwrap.dedent(
                """
                import triton
                import triton.language as tl
                @triton.jit
                def dummy_kernel(x_ptr, y_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
                    pass
                dummy_kernel[(1,)](None, None, None, BLOCK_SIZE=1)
                """
            )
        )

        # --- 2) Patch triton_viz.trace to count invocations
        (tmp_path / "sitecustomize.py").write_text(
            textwrap.dedent(
                """
                import triton_viz

                def fake_trace(*args, **kwargs):
                    print("TRACE_CALLED")
                    def _decorator(fn):
                        return fn
                    return _decorator

                triton_viz.trace = fake_trace
                """
            )
        )
        sys.modules.pop("triton_viz.wrapper", None)

        # --- 3) Start Subprocess to simulate CLI invocation ---
        # load sitecustomize.py
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        env["TRITON_SANITIZER_BACKEND"] = "symexec"
        env["TRITON_INTERPRET"] = "1"

        # run the dummy program using triton-sanitizer
        cmd = ["triton-sanitizer", str(tmp_path / "dummy_program.py")]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        # --- 4) Assertion: trace must be called exactly once
        # Check if the process exited successfully
        assert proc.returncode == 0, f"CLI exited with {proc.returncode}\n{proc.stderr}"
        # Check if trace was called once and only once
        trace_count = proc.stdout.count("TRACE_CALLED")
        assert (
            trace_count == 1
        ), "triton_viz.trace should be invoked exactly once via CLI path"


# ======== Autotuner Compatibility =========
if torch.cuda.is_available():  # Only test if CUDA is available

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        ],
        key=["n_elements"],
    )
    @triton_viz.trace(clients=Sanitizer(abort_on_error=True))
    @triton.jit
    def add_kernel_no_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """
        A Triton kernel that loads and stores values without boundary checks (mask).
        This can lead to out-of-bound access if n_elements exceeds the buffer size.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # No mask is applied here, so loading/storing beyond the valid range can occur.
        x_val = tl.load(x_ptr + offsets)
        y_val = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x_val + y_val)

    def test_autotune_add_inrange():
        """
        This test uses n_elements = 128, matching the size of the input tensors.
        It should NOT cause any out-of-bound access.
        """
        cfg.sanitizer_backend = "symexec"
        x = torch.randn(128)
        y = torch.randn(128)
        out = torch.empty_like(x)

        # The kernel launch uses n_elements=128, aligned with the tensor size.
        grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
        add_kernel_no_mask[grid](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=128)

    def test_autotune_add_out_of_bound():
        """
        This test deliberately sets n_elements = 256, exceeding the actual buffer size (128).
        It will likely cause out-of-bound reads/writes, which may trigger errors or warnings.
        """
        cfg.sanitizer_backend = "symexec"
        x = torch.randn(128)
        y = torch.randn(128)
        out = torch.empty_like(x)

        # The kernel launch uses n_elements=256, exceeding the valid tensor size.
        grid = lambda META: (triton.cdiv(256, META["BLOCK_SIZE"]),)
        with pytest.raises(ValueError):
            add_kernel_no_mask[grid](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=256)
