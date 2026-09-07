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
def test_swallowed_first_alarm_does_not_disable_the_deadline():
    root = Path(__file__).resolve().parents[2]
    code = """
import time
from evaluation.harness import _watchdog

swallowed = False
try:
    with _watchdog(0.05):
        try:
            time.sleep(1)
        except TimeoutError:
            swallowed = True
        time.sleep(1)
except TimeoutError:
    assert swallowed
else:
    raise AssertionError('the first swallowed alarm disabled the watchdog')
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
def test_watchdog_restores_enclosing_timer_and_handler(monkeypatch):
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
    with harness._watchdog(1.0):
        pass
    assert timer_calls == [
        (signal.ITIMER_REAL, 1.0, 0.1),
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
    result = harness._dynamic_track(spec, 0)
    assert result["status"] == expected
    assert result["time_s"] == elapsed
    assert (result["error"] is not None) == (expected == "timeout")
