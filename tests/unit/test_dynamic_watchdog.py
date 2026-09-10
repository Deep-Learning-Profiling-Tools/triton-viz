"""A lost signal must not turn an over-budget interpretation into a proof."""

from contextlib import nullcontext
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation import harness


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX interval timer")
def test_ordinary_fallback_cannot_swallow_deadline_cancellation():
    root = Path(__file__).resolve().parents[2]
    code = """
import time
from evaluation.harness import _watchdog, _DynamicDeadlineExceeded

swallowed = False
try:
    with _watchdog(0.05):
        try:
            time.sleep(1)
        except Exception:
            swallowed = True
        time.sleep(1)
except _DynamicDeadlineExceeded:
    assert not swallowed
else:
    raise AssertionError('ordinary fallback swallowed cancellation')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX interval timer")
@pytest.mark.parametrize("during_unwind", [False, True])
def test_finalizer_finishes_without_swallowing_or_reinjecting_cancellation(
    during_unwind,
):
    root = Path(__file__).resolve().parents[2]
    code = f"""
import sys
import time
import z3
from evaluation.harness import _watchdog, _DynamicDeadlineExceeded

unraisable = []
sys.unraisablehook = unraisable.append
finished = []
class SlowNativeFinalizer:
    def __init__(self):
        self.ast = z3.Int('owned_ast') + 1
    def __del__(self):
        # Deliberately outlive several alarms while owning a native Z3 AST.
        time.sleep(0.18)
        del self.ast
        finished.append(True)

timing = None
try:
    with _watchdog(0.04) as timing:
        owned = SlowNativeFinalizer()
        if {during_unwind!r}:
            try:
                while True:
                    time.sleep(0.001)
            finally:
                del owned
        else:
            del owned
            time.sleep(1)
except _DynamicDeadlineExceeded:
    pass
else:
    raise AssertionError('deadline did not escape the trace')
assert finished == [True], finished
assert not unraisable, unraisable
assert timing['first_signal_at'] <= timing['cancellation_at'] <= timing['scope_exit_at']
assert (timing['finalizer_deferrals'] > 0) == (not {during_unwind!r}), timing
# No background interruption may leak into the next analysis.
control = z3.Solver()
control.add(z3.Int('next_phase') == 3)
assert control.check() == z3.sat
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Exception ignored" not in result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX interval timer")
def test_native_solver_is_interrupted_without_python_signal_delivery():
    root = Path(__file__).resolve().parents[2]
    code = """
import z3
from evaluation import harness

# Isolate the native interruption from Python's alarm mechanism.
harness.signal.signal = lambda *args: None
harness.signal.setitimer = lambda *args: (0.0, 0.0)
query = z3.Solver()
values = z3.Ints(' '.join('pigeon_%d' % i for i in range(40)))
query.add(z3.Distinct(values))
query.add(*[z3.And(v >= 0, v < 39) for v in values])
with harness._watchdog(0.03):
    result = query.check()
assert result == z3.unknown, (result, query.reason_unknown())
assert 'cancel' in query.reason_unknown() or 'interrupt' in query.reason_unknown()
# The joined interrupter must not cancel the following analysis.
control = z3.Solver()
control.add(z3.Int('following_phase') == 7)
assert control.check() == z3.sat
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX interval timer")
def test_native_cancellation_restores_existing_trace_and_solver_context():
    root = Path(__file__).resolve().parents[2]
    code = """
import sys
import z3
from evaluation.harness import _watchdog, _DynamicDeadlineExceeded

def original_trace(frame, event, arg):
    return original_trace
sys.settrace(original_trace)
query = z3.Solver()
values = z3.Ints(' '.join('native_pigeon_%d' % i for i in range(40)))
query.add(z3.Distinct(values))
query.add(*[z3.And(v >= 0, v < 39) for v in values])
try:
    with _watchdog(0.03) as timing:
        query.check()
except _DynamicDeadlineExceeded:
    pass
else:
    raise AssertionError('native solver did not cancel at a Python checkpoint')
assert sys.gettrace() is original_trace
assert timing['native_interrupts'] >= 1, timing
assert timing['first_signal_at'] <= timing['cancellation_at'] <= timing['scope_exit_at']
sys.settrace(None)
control = z3.Solver()
control.add(z3.Int('following_native') == 7)
assert control.check() == z3.sat
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Exception ignored" not in result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX interval timer")
@pytest.mark.parametrize("start_fails", [False, True])
def test_watchdog_restores_enclosing_timer_and_handler(monkeypatch, start_fails):
    old_handler = object()
    handler_calls = []
    timer_calls = []
    ticks = iter((10.0, 10.25))

    def set_handler(sig, handler):
        handler_calls.append((sig, handler))
        return old_handler

    def set_timer(which, delay, interval=0.0):
        timer_calls.append((which, delay, interval))
        return (2.0, 0.4) if len(timer_calls) == 1 else (0.0, 0.0)

    monkeypatch.setattr(harness.signal, "signal", set_handler)
    monkeypatch.setattr(harness.signal, "setitimer", set_timer)
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks))
    if start_fails:

        def fail_start(self):
            raise RuntimeError("cannot start new thread")

        monkeypatch.setattr(harness.threading.Thread, "start", fail_start)
    expected = (
        pytest.raises(RuntimeError, match="cannot start new thread")
        if start_fails
        else nullcontext()
    )
    with expected:
        with harness._watchdog(1.0):
            pass
    assert timer_calls == [
        (signal.ITIMER_REAL, 1.0, 0.0),
        (signal.ITIMER_REAL, 0, 0.0),
        (signal.ITIMER_REAL, 1.75, 0.4),
    ]
    assert handler_calls[-1] == (signal.SIGALRM, old_handler)


@pytest.mark.parametrize(
    "elapsed, expected", [(0.25, "ok"), (60.0, "timeout"), (90.8, "timeout")]
)
def test_dynamic_completion_checks_deadline_even_without_an_exception(
    monkeypatch, elapsed, expected
):
    import triton_viz
    from triton_viz import clients

    detector = SimpleNamespace(
        last_status="ok", last_reports=[], unsupported_reason=None, last_premises=()
    )
    monkeypatch.setattr(clients, "RaceDetector", lambda **kwargs: detector)

    class Launcher:
        def __getitem__(self, grid):
            return lambda **kwargs: None

    monkeypatch.setattr(triton_viz, "trace", lambda det: lambda fn: Launcher())
    monkeypatch.setattr(harness, "_watchdog", lambda seconds: nullcontext())
    ticks = iter((10.0, 10.0 + elapsed))
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(ticks))
    spec = SimpleNamespace(
        make_args=lambda seed: (),
        kernel_fn=SimpleNamespace(arg_names=()),
        grid=(1,),
        constexprs={},
    )
    result = harness._dynamic_track_local(spec, 0)
    assert result["status"] == expected
    assert result["time_s"] == elapsed
    assert (result["error"] is not None) == (expected == "timeout")


def test_deadline_discards_partial_reports_and_premises(monkeypatch):
    import triton_viz
    from triton_viz import clients

    detector = SimpleNamespace(
        last_status="ok",
        last_reports=[object()],
        unsupported_reason=None,
        last_premises=("partial snapshot",),
    )
    monkeypatch.setattr(clients, "RaceDetector", lambda **kwargs: detector)

    class Launcher:
        def __getitem__(self, grid):
            def run(**kwargs):
                raise harness._DynamicDeadlineExceeded("expired")

            return run

    monkeypatch.setattr(triton_viz, "trace", lambda det: lambda fn: Launcher())
    monkeypatch.setattr(harness, "_watchdog", lambda seconds: nullcontext())
    spec = SimpleNamespace(
        make_args=lambda seed: (),
        kernel_fn=SimpleNamespace(arg_names=()),
        grid=(1,),
        constexprs={},
    )
    result = harness._dynamic_track_local(spec, 0)
    assert result["status"] == "timeout"
    assert result["n_reports"] == 0
    assert result["witnesses"] == []
    assert result["premises"] == []
