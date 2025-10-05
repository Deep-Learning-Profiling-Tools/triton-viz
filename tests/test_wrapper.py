# tests/test_triton_sanitizer_trace_injection.py
import json
import os
import textwrap
import subprocess
from pathlib import Path
import shutil
from triton_viz.wrapper import SANITIZER_COMMAND, PROFILER_COMMAND


def test_triton_sanitizer_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification:
    - Pure @triton.jit kernels have exactly one @triton_viz.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch triton_viz.trace at subprocess startup,
      record each trace decorator application target (function name/type), write to JSON on exit.
    """

    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        textwrap.dedent(
            f"""
            import atexit, json, os
            # Import and replace triton_viz.trace here
            import triton_viz as _tv

            _orig_trace = _tv.trace
            _records = []

            def _spy_trace(*t_args, **t_kwargs):
                # Call original trace, get the actual decorator
                _decorator = _orig_trace(*t_args, **t_kwargs)
                def _apply(target):
                    # Which object is being applied to (usually function name, or autotune/jit product)
                    name = getattr(target, "__name__", None) or getattr(target, "__qualname__", None) or repr(target)
                    typ  = type(target).__name__
                    mod  = getattr(target, "__module__", None)

                    _records.append({{
                        "name": name,
                        "type": typ,
                        "module": mod,
                    }})

                    # Continue with original logic, keep functionality unchanged
                    return _decorator(target)
                return _apply

            # Replace
            _tv.trace = _spy_trace

            # Write records to disk on process exit
            _LOG_PATH = {json.dumps(str(trace_log))}
            @atexit.register
            def _dump_records():
                try:
                    with open(_LOG_PATH, "w") as f:
                        json.dump(_records, f)
                except Exception as e:
                    # Avoid affecting test process exit
                    pass
            """
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels
    #    - k1: pure jit
    #    - k2: autotune + jit
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        textwrap.dedent(
            """
            import triton
            import triton.language as tl

            # --- Pure jit ---
            @triton.jit
            def k1(X, n: tl.constexpr):
                return

            # --- autotune + jit ---
            @triton.autotune(configs=[triton.Config({})], key=['n'])
            @triton.jit
            def k2(X, n: tl.constexpr):
                return

            if __name__ == "__main__":
                # No need to actually launch kernel; we just need the decorator chain to be built
                pass
            """
        ),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    # Put tmp_path at the front, ensure sitecustomize.py is auto-imported
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find triton-sanitizer executable - fail if not found
    exe = shutil.which(SANITIZER_COMMAND)
    assert (
        exe is not None
    ), f"{SANITIZER_COMMAND} command not found. Please ensure the package is properly installed."
    cmd = [exe, str(my_program)]

    # 5) Run subprocess
    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Optional debug output (easier to locate on failure)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, f"{SANITIZER_COMMAND} run failed"

    # 6) Read trace records
    assert (
        trace_log.exists()
    ), "trace log not generated (sitecustomize may not have taken effect)"
    records = json.loads(trace_log.read_text(encoding="utf-8"))

    # Build counter by function name
    from collections import Counter

    counts = Counter(rec["name"] for rec in records)

    # Assertion: the 2 kernel names defined in our script should each appear exactly once
    # - k1: pure jit → should have trace injected once at outer layer
    # - k2: autotune + jit → should only appear once at autotune outer layer (should not trace inner jit again)
    for expected in ["k1", "k2"]:
        assert (
            counts[expected] == 1
        ), f"{expected} should be traced 1 time, actual count is {counts[expected]}; may have missing or duplicate injection"


def test_triton_profiler_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification for triton-profiler:
    - Pure @triton.jit kernels have exactly one @triton_viz.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch triton_viz.trace at subprocess startup,
      record each trace decorator application target (function name/type), write to JSON on exit.
    """

    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        textwrap.dedent(
            f"""
            import atexit, json, os
            # Import and replace triton_viz.trace here
            import triton_viz as _tv

            _orig_trace = _tv.trace
            _records = []

            def _spy_trace(*t_args, **t_kwargs):
                # Call original trace, get the actual decorator
                _decorator = _orig_trace(*t_args, **t_kwargs)
                def _apply(target):
                    # Which object is being applied to (usually function name, or autotune/jit product)
                    name = getattr(target, "__name__", None) or getattr(target, "__qualname__", None) or repr(target)
                    typ  = type(target).__name__
                    mod  = getattr(target, "__module__", None)

                    _records.append({{
                        "name": name,
                        "type": typ,
                        "module": mod,
                    }})

                    # Continue with original logic, keep functionality unchanged
                    return _decorator(target)
                return _apply

            # Replace
            _tv.trace = _spy_trace

            # Write records to disk on process exit
            _LOG_PATH = {json.dumps(str(trace_log))}
            @atexit.register
            def _dump_records():
                try:
                    with open(_LOG_PATH, "w") as f:
                        json.dump(_records, f)
                except Exception as e:
                    # Avoid affecting test process exit
                    pass
            """
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels
    #    - k1: pure jit
    #    - k2: autotune + jit
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        textwrap.dedent(
            """
            import triton
            import triton.language as tl

            # --- Pure jit ---
            @triton.jit
            def k1(X, n: tl.constexpr):
                return

            # --- autotune + jit ---
            @triton.autotune(configs=[triton.Config({})], key=['n'])
            @triton.jit
            def k2(X, n: tl.constexpr):
                return

            if __name__ == "__main__":
                # No need to actually launch kernel; we just need the decorator chain to be built
                pass
            """
        ),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    # Put tmp_path at the front, ensure sitecustomize.py is auto-imported
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find triton-profiler executable - fail if not found
    exe = shutil.which(PROFILER_COMMAND)
    assert (
        exe is not None
    ), f"{PROFILER_COMMAND} command not found. Please ensure the package is properly installed."
    cmd = [exe, str(my_program)]

    # 5) Run subprocess
    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Optional debug output (easier to locate on failure)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, f"{PROFILER_COMMAND} run failed"

    # 6) Read trace records
    assert (
        trace_log.exists()
    ), "trace log not generated (sitecustomize may not have taken effect)"
    records = json.loads(trace_log.read_text(encoding="utf-8"))

    # Build counter by function name
    from collections import Counter

    counts = Counter(rec["name"] for rec in records)

    # Assertion: the 2 kernel names defined in our script should each appear exactly once
    # - k1: pure jit → should have trace injected once at outer layer
    # - k2: autotune + jit → should only appear once at autotune outer layer (should not trace inner jit again)
    for expected in ["k1", "k2"]:
        assert (
            counts[expected] == 1
        ), f"{expected} should be traced 1 time, actual count is {counts[expected]}; may have missing or duplicate injection"
