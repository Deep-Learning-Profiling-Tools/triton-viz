"""The D10b version gate, on every Triton release (D29).

IR mode runs only on the Triton releases in
``tilelens.core.config.TESTED_TRITON_VERSIONS`` unless
``TILELENS_IR_ALLOW_UNTESTED_TRITON=1``. Outside that window the IR-mode test
modules skip (tests/conftest.py marks them ``ir_mode``); this module is
deliberately not one of them. It checks that the gate refuses correctly
wherever it runs, on a release in the window or not, with or without the
override in the environment: each test that expects a refusal pins the
override off (``gate``), and nothing here compiles on a release the gate
refuses, since the refusal comes first. It also checks the conftest's own
gating of the IR-mode tests.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.clients import Sanitizer
from tilelens.core.client import Client, ClientManager, LaunchCall
from tilelens.core.config import (
    DEFAULT_IR_TARGET,
    TESTED_TRITON_VERSIONS,
    Config,
    config as tilelens_config,
    untested_triton_version,
)
from tilelens.core.host_compile import HostCompiler
from tilelens.ir import IRClient, IRVerdict

trace_module = importlib.import_module("tilelens.core.trace")
TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
OVERRIDE_VARS = (
    "TILELENS_IR_ALLOW_UNTESTED_TRITON",
    "TRITON_VIZ_IR_ALLOW_UNTESTED_TRITON",
)
# A release outside the window: older than any release the window will hold.
UNTESTED = "3.5.1"


@pytest.fixture(autouse=True)
def _real_jit(monkeypatch):
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time; pin the knob off so @triton.jit builds real JITFunctions.
    from triton import knobs

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    yield
    if previous is missing:
        knobs.runtime.__dict__.pop("interpret", None)
    else:
        knobs.runtime.__dict__["interpret"] = previous


@pytest.fixture(autouse=True)
def _default_ir_target(monkeypatch):
    # Whatever TILELENS_IR_TARGET the caller has set.
    monkeypatch.setattr(tilelens_config, "ir_target", DEFAULT_IR_TARGET)


@pytest.fixture
def gate(monkeypatch):
    """The gate as it stands without the override, whatever the caller's
    environment says; ``gate(version)`` pretends ``version`` is installed."""
    monkeypatch.setattr(tilelens_config, "ir_allow_untested_triton", False)
    return lambda version: monkeypatch.setattr(triton, "__version__", version)


@pytest.fixture
def no_compile(monkeypatch):
    """Fail any host compile: a refused launch compiles nothing."""

    def compile(self, jit_fn, *args, **kwargs):
        raise AssertionError(f"host compile of {jit_fn!r} on a refused Triton")

    monkeypatch.setattr(HostCompiler, "compile", compile)


@pytest.fixture
def no_driver(unreachable_driver):
    """IR mode never asks Triton's driver (D25), refused or not."""
    unreachable_driver("IR mode queried Triton's driver")


def _make_copy():
    @triton.jit
    def copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    return copy


def _window() -> str:
    return ", ".join(f"{release}.x" for release in TESTED_TRITON_VERSIONS)


# ======== the window =========


def test_the_window_is_by_minor_release(gate, monkeypatch):
    for version, expected in (
        ("3.6.0", None),
        ("3.6.1+git1234abc", None),
        ("3.8.0", None),
        ("3.8.1", None),
        ("3.5.1", "3.5.1"),
        ("3.7.0", "3.7.0"),
        ("3.9.0", "3.9.0"),
        ("3.60.0", "3.60.0"),
        ("4.0.0", "4.0.0"),
    ):
        gate(version)
        assert untested_triton_version() == expected
    monkeypatch.setattr(tilelens_config, "ir_allow_untested_triton", True)
    assert untested_triton_version() is None


@pytest.mark.parametrize("prefix", ["TILELENS_", "TRITON_VIZ_"])
def test_the_override_is_read_like_every_tilelens_env_flag(monkeypatch, prefix):
    for name in OVERRIDE_VARS:
        monkeypatch.delenv(name, raising=False)
    assert Config().ir_allow_untested_triton is False
    monkeypatch.setenv(f"{prefix}IR_ALLOW_UNTESTED_TRITON", "1")
    assert Config().ir_allow_untested_triton is True


# ======== the core: nothing is captured =========


class _RecordingIR(Client):
    """An IR client that records the lifecycle; nothing interprets."""

    NAME = "recording_ir"
    NEEDS_INTERPRETER = False
    IR_STAGES = frozenset({"ttir"})
    LAUNCH = "skip"

    def __init__(self):
        super().__init__()
        self.log: list = []
        self.calls: list[LaunchCall] = []

    def begin_launch(self, call):
        self.log.append("begin")
        self.calls.append(call)

    def before_launch(self, event):
        self.log.append("before")

    def compile_failed(self, event):
        self.log.append("compile_failed")

    def finalize(self):
        self.log.append("finalize")
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass

    def _unreachable(self, *args, **kwargs):
        raise AssertionError("an interpreter hook reached an IR client")

    pre_run_callback = post_run_callback = arg_callback = _unreachable
    grid_callback = grid_idx_callback = _unreachable
    register_op_callback = register_for_loop_callback = _unreachable


def test_the_core_captures_nothing_on_an_untested_triton(gate, monkeypatch):
    """Outside the window LaunchCall.capture is False: no host compile, no
    IR event, and a skipped launch runs nothing; the override captures."""
    compiles: list = []

    class _Kernel:
        hash = "hash"
        asm = {"ttir": "// ttir"}

    def compile(self, jit_fn, args, kwargs, *, target, stages=()):
        compiles.append(jit_fn)
        return _Kernel()

    monkeypatch.setattr(HostCompiler, "compile", compile)
    gate(UNTESTED)
    ir = _RecordingIR()
    traced = tilelens.trace(ir)(_make_copy())
    x, out = torch.ones(8), torch.zeros(8)

    assert traced[(2,)](x, out, 8, BLOCK=4) is None
    assert compiles == [] and ir.log == ["begin", "finalize"]
    (call,) = ir.calls
    assert call.jit_fn is traced.jit_fn and call.capture is False
    assert torch.equal(out, torch.zeros(8))

    monkeypatch.setattr(tilelens_config, "ir_allow_untested_triton", True)
    traced[(2,)](x, out, 8, BLOCK=4)
    assert compiles == [traced.jit_fn] and ir.calls[-1].capture is True
    assert ir.log[2:] == ["begin", "before", "finalize"]


# ======== IR clients: refused before any analysis =========


class _ToyIR(IRClient):
    NAME = "toy_ir"
    LAUNCH = "skip"
    IR_STAGES = frozenset({"ttir"})

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def analyze_launch(self, log):
        self.calls.append("analyze")
        return [], IRVerdict(self.NAME, "ok")

    def on_analysis_error(self, exc):
        self.calls.append(("error", exc))
        return IRVerdict(self.NAME, "error", notes=[repr(exc)])

    def on_refusal(self, refusal):
        self.calls.append(("refusal", refusal))
        return IRVerdict(self.NAME, "unsupported", refusal=refusal)


def _toy_launch(ir):
    manager = ClientManager([ir])
    call = LaunchCall(jit_fn=None, args=(), kwargs={}, grid=(1,), capture=True)
    manager.begin_launch(call)
    manager.finalize()
    return manager


def test_an_ir_client_refuses_before_any_analysis(gate):
    gate(UNTESTED)
    ir = _ToyIR()
    manager = _toy_launch(ir)

    ((hook, refusal),) = ir.calls
    assert hook == "refusal" and refusal.kind == "untested-triton-version"
    assert refusal.message == (
        f"IR mode is tested on Triton {_window()}, not {UNTESTED}; "
        "set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to run it anyway"
    )
    assert manager.launch.records == [ir.last_verdict]
    assert ir.last_verdict == IRVerdict("toy_ir", "unsupported", refusal=refusal)


def test_an_ir_client_analyzes_under_the_override(gate, monkeypatch):
    gate(UNTESTED)
    monkeypatch.setattr(tilelens_config, "ir_allow_untested_triton", True)
    ir = _ToyIR()
    _toy_launch(ir)
    assert ir.calls == ["analyze"] and ir.last_verdict.status == "ok"


# ======== the compiled sanitizer =========


def test_the_compiled_sanitizer_refuses_an_untested_triton(gate, no_compile, no_driver):
    gate(UNTESTED)
    det = Sanitizer(compile=True, abort_on_error=False)
    x, out = torch.ones(8), torch.zeros(8)

    assert tilelens.trace(det)(_make_copy())[(2,)](x, out, 8, BLOCK=4) is None

    verdict = det.last_verdict
    assert (verdict.status, verdict.refusal.kind) == (
        "unsupported",
        "untested-triton-version",
    )
    assert f"not {UNTESTED}" in verdict.refusal.message
    assert det.records == [] and trace_module.launches[-1].records == [verdict]
    assert torch.equal(out, torch.zeros(8))  # nothing ran


def test_a_call_that_does_not_bind_is_not_bound_on_an_untested_triton(
    gate, no_compile, no_driver
):
    """D28 holds where IR mode runs: on an untested release the JIT's
    binder (private API) is never reached, so a call that does not bind
    the kernel's parameters is one more refused launch, not a TypeError
    (the untraced call would raise it)."""
    gate(UNTESTED)
    det = Sanitizer(compile=True, abort_on_error=False)
    x, out = torch.ones(8), torch.zeros(8)

    assert tilelens.trace(det)(_make_copy())[(2,)](x, out, BLOCK=4) is None

    assert (det.last_status, det.last_verdict.refusal.kind) == (
        "unsupported",
        "untested-triton-version",
    )


def test_the_installed_triton_is_gated_as_the_window_says(gate, no_driver):
    """The installed release, not a pretended one: refused exactly when it
    is outside the window (then nothing compiles), analyzed otherwise."""
    det = Sanitizer(compile=True, abort_on_error=False)
    x, out = torch.ones(8), torch.zeros(8)

    tilelens.trace(det)(_make_copy())[(2,)](x, out, 8, BLOCK=4)

    refusal = det.last_verdict.refusal
    kind = None if refusal is None else refusal.kind
    if untested_triton_version() is None:
        assert kind != "untested-triton-version"
    else:
        assert kind == "untested-triton-version"
        assert f"not {triton.__version__};" in refusal.message


_CLI_SCRIPT = """\
import torch, triton, triton.language as tl

triton.__version__ = {version!r}  # an untested release, as far as IR mode knows


@triton.jit
def copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))  # unmasked: OOB if checked


x, out = torch.ones(6), torch.zeros(6)
copy[(1,)](x, out, 6, BLOCK=8)
print("launch returned", out.sum().item())
"""


def test_the_cli_reports_an_untested_triton_and_goes_on(tmp_path):
    """tile-sanitizer --compile on an untested release: each launch is
    reported as not checked, the kernel does not run, the script goes on."""
    script = tmp_path / "untested.py"
    script.write_text(_CLI_SCRIPT.format(version=UNTESTED))
    cli = (
        f"import sys; sys.argv = ['tile-sanitizer', '--compile', {str(script)!r}]; "
        "from tilelens.wrapper import apply_sanitizer; apply_sanitizer()"
    )
    # The override unset: the gate as it stands; this checkout first on the
    # path, then whatever the caller put there (e.g. another Triton release).
    unset = ("TRITON_INTERPRET", *OVERRIDE_VARS)
    env = {k: v for k, v in os.environ.items() if k not in unset}
    path = os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])])
    env.update(PYTHONPATH=path, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-c", cli], capture_output=True, text=True, env=env, cwd=REPO
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "[CompiledSanitizer] not checked: untested-triton-version: IR mode is "
        f"tested on Triton {_window()}, not {UNTESTED}; set "
        "TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to run it anyway",
        "launch returned 0.0",
    ]


# ======== the IR-mode tests' own gate (tests/conftest.py) =========


@pytest.fixture
def tests_conftest(request):
    """tests/conftest.py, as pytest loaded it."""
    path = TESTS / "conftest.py"
    (module,) = [
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) and Path(plugin.__file__).resolve() == path
    ]
    return module


def test_the_ir_mode_modules_are_the_d29_list(tests_conftest):
    ir_mode = [
        "unit/ir/test_host_compile.py",
        "unit/ir/test_ir_capture.py",
        "unit/ir/test_mlir_walk.py",
        "unit/ir/test_ttir_reader.py",
        "unit/ir/test_verdict_io.py",
        "unit/sanitizer_compiled/test_client.py",
        "unit/sanitizer_compiled/test_oob.py",
        "unit/test_ir_lifecycle.py",
        "end_to_end/test_ir_client.py",
        "end_to_end/test_ir_lifecycle_compiled.py",
        "end_to_end/test_ir_smoke.py",
        "end_to_end/test_compiled_sanitizer.py",
        "end_to_end/test_host_compile.py",
    ]
    others = [
        "unit/test_ir_version_gate.py",  # this module: runs everywhere
        "conformance/test_reader_conformance.py",  # a non-strict xfail instead
        "unit/test_client_manager.py",
        "unit/test_wrapper.py",
        "unit/test_sanitizer.py",
        "end_to_end/test_sanitizer.py",
    ]
    for relative in ir_mode:
        assert (TESTS / relative).is_file(), relative
        assert tests_conftest.is_ir_mode_module(TESTS / relative), relative
    for relative in others:
        assert not tests_conftest.is_ir_mode_module(TESTS / relative), relative


def test_the_skip_reason_names_the_release_and_the_window(
    tests_conftest, gate, monkeypatch
):
    gate(UNTESTED)
    assert tests_conftest.ir_mode_skip_reason() == (
        f"IR mode is not tested on the installed Triton {UNTESTED}: the tested "
        "window (tilelens.core.config.TESTED_TRITON_VERSIONS) is Triton "
        f"{_window()}; set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to run the "
        "IR-mode tests anyway (D29)"
    )
    monkeypatch.setattr(tilelens_config, "ir_allow_untested_triton", True)
    assert tests_conftest.ir_mode_skip_reason() is None
    monkeypatch.setattr(tilelens_config, "ir_allow_untested_triton", False)
    gate(f"{TESTED_TRITON_VERSIONS[0]}.0")
    assert tests_conftest.ir_mode_skip_reason() is None


def test_this_session_gates_the_ir_mode_tests(tests_conftest, request):
    """Every IR-mode test collected with this one skips with the conftest's
    reason exactly when the installed Triton (and the override as the
    environment sets it) says so; this module's tests never do."""
    reason = tests_conftest.ir_mode_skip_reason()
    for item in request.session.items:
        skipifs = list(item.iter_markers("skipif"))
        gated = [m for m in skipifs if "(D29)" in str(m.kwargs.get("reason"))]
        if item.get_closest_marker(tests_conftest.IR_MODE) is None or reason is None:
            assert gated == [], item.nodeid
        else:
            # The first skipif pytest evaluates: its reason is the one shown.
            assert gated == skipifs[:1], item.nodeid
            assert gated[0].args == (True,) and gated[0].kwargs == {"reason": reason}
    assert request.node.get_closest_marker(tests_conftest.IR_MODE) is None
