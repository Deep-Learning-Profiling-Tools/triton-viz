import json
import os
import tempfile
import subprocess
from pathlib import Path
from string import Template
import shutil
from triton_viz.wrapper import SANITIZER_COMMAND, PROFILER_COMMAND


TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name: str, **kwargs) -> str:
    """Load a template file and substitute variables."""
    template_path = TEMPLATE_DIR / name
    template = Template(template_path.read_text())
    return template.substitute(**kwargs)


def test_triton_sanitizer_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification:
    - Pure @triton.jit kernels have exactly one @triton_viz.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer
      layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch triton_viz.trace at subprocess startup,
      record each trace decorator application target (function name/type),
      write to JSON on exit.
    """
    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        load_template(
            "sitecustomize_spy_trace.py.template",
            log_path=json.dumps(str(trace_log)),
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels (k1: pure jit, k2: autotune + jit)
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        load_template("kernel_jit_autotune.py.template"),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find triton-sanitizer executable - fail if not found
    exe = shutil.which(SANITIZER_COMMAND)
    assert exe is not None, (
        f"{SANITIZER_COMMAND} command not found. "
        "Please ensure the package is properly installed."
    )
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
    for expected in ["k1", "k2"]:
        assert counts[expected] == 1, (
            f"{expected} should be traced 1 time, actual count is {counts[expected]}; "
            "may have missing or duplicate injection"
        )


def test_triton_profiler_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification for triton-profiler:
    - Pure @triton.jit kernels have exactly one @triton_viz.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer
      layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch triton_viz.trace at subprocess startup,
      record each trace decorator application target (function name/type),
      write to JSON on exit.
    """
    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        load_template(
            "sitecustomize_spy_trace.py.template",
            log_path=json.dumps(str(trace_log)),
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels (k1: pure jit, k2: autotune + jit)
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        load_template("kernel_jit_autotune.py.template"),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find triton-profiler executable - fail if not found
    exe = shutil.which(PROFILER_COMMAND)
    assert exe is not None, (
        f"{PROFILER_COMMAND} command not found. "
        "Please ensure the package is properly installed."
    )
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
    for expected in ["k1", "k2"]:
        assert counts[expected] == 1, (
            f"{expected} should be traced 1 time, actual count is {counts[expected]}; "
            "may have missing or duplicate injection"
        )


def test_cli_invocation():
    """
    Simulate running:
        $ triton-sanitizer dummy_program.py
    and assert that triton_viz.trace is invoked exactly once.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1) Create a temporary Triton script
        (tmp_path / "dummy_program.py").write_text(
            load_template("kernel_dummy.py.template")
        )

        # 2) Patch triton_viz.trace to count invocations
        (tmp_path / "sitecustomize.py").write_text(
            load_template("sitecustomize_fake_trace.py.template")
        )

        # 3) Start Subprocess to simulate CLI invocation
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        env["TRITON_INTERPRET"] = "1"

        # Run the dummy program using triton-sanitizer
        cmd = ["triton-sanitizer", str(tmp_path / "dummy_program.py")]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        # 4) Assertion: trace must be called exactly once
        assert proc.returncode == 0, f"CLI exited with {proc.returncode}\n{proc.stderr}"
        trace_count = proc.stdout.count("TRACE_CALLED")
        assert (
            trace_count == 1
        ), "triton_viz.trace should be invoked exactly once via CLI path"
