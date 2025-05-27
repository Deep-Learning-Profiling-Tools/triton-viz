"""
Test that triton_viz.wrapper works correctly:
1. It should patch triton.jit / triton.language.jit /
   triton.runtime.interpreter.jit with wrapper._patched_jit
2. The first use of @triton.jit must invoke triton_viz.trace(Sanitizer)
"""

import importlib
import sys
import tempfile
from pathlib import Path
import textwrap
import os
import subprocess

import triton
import triton.language as tl


def test_jit_patched_and_trace_called():
    """Main test: verify jit is patched and trace is called exactly once."""
    # Remove the already-loaded wrapper (relevant when pytest runs multiple times)
    sys.modules.pop("triton_viz.wrapper", None)

    # triton-viz must be imported before patching trace
    import triton_viz

    # A simple counter for how many times trace is called
    trace_calls = {"count": 0}

    def fake_trace(*args, **kwargs):
        """Return a decorator that leaves the function unchanged and count calls."""
        trace_calls["count"] += 1

        def _decorator(fn):
            return fn
        return _decorator

    # Monkey-patch triton_viz.trace and reload wrapper
    triton_viz.trace = fake_trace
    wrapper = importlib.import_module("triton_viz.wrapper")

    # Now start the actual test
    # --- 1) jit must be replaced ---
    assert triton.jit is wrapper._patched_jit
    assert tl.jit is wrapper._patched_jit
    import triton.runtime.interpreter as _interp
    assert _interp.jit is wrapper._patched_jit

    # --- 2) Define a dummy kernel to trigger @triton.jit once ---
    @triton.jit
    def dummy_kernel(x_ptr, y_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
        pass
    dummy_kernel[(1,)](None, None, None, BLOCK_SIZE=1)

    # --- 3) Verify that the fake trace was called exactly once ---
    assert trace_calls["count"] == 1, "triton_viz.trace was not invoked correctly"

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

        # run the dummy program using triton-sanitizer
        cmd = ["triton-sanitizer", str(tmp_path / "dummy_program.py")]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        # --- 4) Assertion: trace must be called exactly once
        assert proc.returncode == 0, f"CLI exited with {proc.returncode}\n{proc.stderr}"
        trace_count = proc.stdout.count("TRACE_CALLED")
        assert trace_count == 1, (
            "triton_viz.trace should be invoked exactly once via CLI path"
        )
