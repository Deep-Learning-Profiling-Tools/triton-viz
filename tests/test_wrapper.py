"""
Test that triton_viz.wrapper works correctly:
1. It should patch triton.jit / triton.language.jit /
   triton.runtime.interpreter.jit with wrapper._patched_jit
2. The first use of @triton.jit must invoke triton_viz.trace(Sanitizer)
"""
import sys
import tempfile
from pathlib import Path
import textwrap
import os
import subprocess


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
