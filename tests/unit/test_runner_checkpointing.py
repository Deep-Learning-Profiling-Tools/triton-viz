"""Fresh-process cancellation and the timing boundary used by row checkpoints.

Tiny Python subprocesses exercise the real pipe/process machinery without
running a detector corpus or spending any of the pinned experiment's budget.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from evaluation import runner


SPEC = SimpleNamespace(name="checkpoint_probe", expected="race-free", pattern="probe")
ROW = {
    "name": SPEC.name,
    "corpus": "checkpoint_probe",
    "verdict": "race-free",
    "terminal": "proved",
}


def _worker(monkeypatch, code, *, spawn_delay=0):
    """Replace only the harness command, retaining actual Popen and pipes."""
    real_popen = subprocess.Popen

    def popen(cmd, **kwargs):
        output = cmd[cmd.index("--out") + 1]
        time.sleep(spawn_delay)
        return real_popen([sys.executable, "-c", code, output], **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)


def _run(timeout, **kwargs):
    return runner._run_one(SPEC, "checkpoint_probe", 0, timeout, False, **kwargs)


def _write_row_code(delay=0):
    return (
        "import json, sys, time; "
        f"time.sleep({delay!r}); "
        f"open(sys.argv[1], 'w').write({json.dumps(ROW)!r})"
    )


def _alive(pid):
    try:
        # A killed grandchild may briefly await reaping by the host's init;
        # a zombie is no longer consuming CPU or running experiment work.
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


def test_slow_spawn_counts_in_wall_but_not_timeout(monkeypatch, tmp_path):
    _worker(monkeypatch, _write_row_code(0.08), spawn_delay=0.3)
    row = _run(0.2, output_dir=tmp_path, cancel_requested=lambda: False)
    assert row["terminal"] == "proved"
    assert row["wall_s"] >= 0.38
    assert not list(tmp_path.iterdir())


def test_timeout_remains_timeout_when_pause_arrives_at_deadline(monkeypatch, tmp_path):
    _worker(monkeypatch, "import time; time.sleep(30)")
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) > 1

    row = _run(0.05, output_dir=tmp_path, cancel_requested=cancelled)
    assert row["terminal"] == "timeout"
    assert row["harness_error"] == "exceeded 0.05s"
    assert len(checks) == 1  # deadline was observed before a second pause check
    assert not list(tmp_path.iterdir())


def test_completion_wins_over_unobserved_pause(monkeypatch, tmp_path):
    _worker(monkeypatch, _write_row_code())
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) > 1

    row = _run(5, output_dir=tmp_path, cancel_requested=cancelled)
    assert row["terminal"] == "proved"
    assert len(checks) == 1


def test_completion_during_pause_callback_is_preserved(monkeypatch, tmp_path):
    _worker(monkeypatch, _write_row_code(0.2))
    processes = []

    def cancelled():
        if not processes:  # initial check before spawn
            return False
        processes[0].wait(timeout=3)
        return True

    def forbidden_cleanup(proc):
        pytest.fail("a completed row must not trigger process-group cleanup")

    monkeypatch.setattr(runner, "_kill_row_group", forbidden_cleanup)
    row = _run(
        5,
        output_dir=tmp_path,
        on_spawn=processes.append,
        cancel_requested=cancelled,
    )
    assert row["terminal"] == "proved"
    assert processes[0].returncode == 0
    assert not list(tmp_path.iterdir())


def test_deadline_crossed_inside_pause_callback_remains_timeout(monkeypatch, tmp_path):
    _worker(monkeypatch, "import time; time.sleep(30)")
    checks = []

    def cancelled():
        checks.append(True)
        if len(checks) == 1:
            return False
        time.sleep(0.55)
        return True

    row = _run(0.5, output_dir=tmp_path, cancel_requested=cancelled)
    assert row["terminal"] == "timeout"
    assert row["harness_error"] == "exceeded 0.5s"
    assert len(checks) == 2
    assert not list(tmp_path.iterdir())


def test_cancel_before_spawn_does_not_create_worker(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("cancelled attempt must not spawn")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(runner.RowInterrupted, match="before process creation"):
        _run(5, output_dir=tmp_path, cancel_requested=lambda: True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stop", ["cancel", "timeout", "registration_error"])
def test_abnormal_exit_kills_descendants_and_reaps_child(monkeypatch, tmp_path, stop):
    child_file = tmp_path / "grandchild.pid"
    code = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    while True: time.sleep(1)\n"
        f"open({str(child_file)!r}, 'w').write(str(pid))\n"
        "while True: time.sleep(1)\n"
    )
    _worker(monkeypatch, code)
    processes = []

    def registered(proc):
        processes.append(proc)
        assert os.getpgid(proc.pid) == proc.pid
        if stop == "registration_error":
            deadline = time.monotonic() + 5
            while not child_file.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            raise RuntimeError("registration failed")

    options = {
        "on_spawn": registered,
        "output_dir": tmp_path,
        "cancel_requested": lambda: stop == "cancel" and child_file.exists(),
    }
    try:
        if stop == "timeout":
            assert _run(0.25, **options)["terminal"] == "timeout"
        else:
            expected = runner.RowInterrupted if stop == "cancel" else RuntimeError
            with pytest.raises(expected):
                _run(5, **options)
        assert processes and processes[0].returncode is not None
        assert not _alive(processes[0].pid)
        assert child_file.exists()
        grandchild = int(child_file.read_text())
        deadline = time.monotonic() + 5
        while _alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _alive(grandchild)
        assert list(tmp_path.iterdir()) == [child_file]
    finally:
        # Ensure that a failed assertion cannot leave the test's own children.
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


def test_large_diagnostics_are_drained_without_deadlock(monkeypatch, tmp_path):
    code = "import sys; sys.stderr.write('x' * 300000); " + _write_row_code()
    _worker(monkeypatch, code)
    row = _run(5, output_dir=tmp_path, cancel_requested=lambda: False)
    assert row["terminal"] == "proved"


def test_crash_keeps_existing_row_shape(monkeypatch, tmp_path):
    _worker(monkeypatch, "import sys; sys.stderr.write('worker crashed'); sys.exit(3)")
    row = _run(5, output_dir=tmp_path, cancel_requested=lambda: False)
    assert row["verdict"] == "error"
    assert row["terminal"] == "crash"
    assert row["harness_error"] == "worker crashed"
    assert row["name"] == SPEC.name and row["expected"] == SPEC.expected
    assert row["ladder_level"] == "L0" and "fence_order" in row


def test_spawn_callback_is_inside_wall_envelope(monkeypatch, tmp_path):
    _worker(monkeypatch, _write_row_code())
    row = _run(5, output_dir=tmp_path, on_spawn=lambda proc: time.sleep(0.2))
    assert row["wall_s"] >= 0.2
    assert row["terminal"] == "proved"


@pytest.mark.parametrize("operator_cancel", [False, True])
def test_signaled_worker_requires_operator_cancel_to_be_interruption(
    monkeypatch, tmp_path, operator_cancel
):
    from evaluation.pinned_resume import Control

    _worker(monkeypatch, "import time; time.sleep(30)")
    control = Control(tmp_path)
    processes = []

    def stop_worker(proc):
        processes.append(proc)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        if operator_cancel:
            # Model the same signal delivered to the service controller.
            os.kill(os.getpid(), signal.SIGTERM)

    with control.signals():
        if operator_cancel:
            with pytest.raises(runner.RowInterrupted, match="signaled worker"):
                _run(
                    5,
                    output_dir=tmp_path,
                    cancel_requested=control.immediate,
                    on_spawn=stop_worker,
                )
        else:
            row = _run(
                5,
                output_dir=tmp_path,
                cancel_requested=control.immediate,
                on_spawn=stop_worker,
            )
            assert row["verdict"] == "error" and row["terminal"] == "crash"
    assert processes[0].returncode == -signal.SIGTERM
    assert not list(tmp_path.iterdir())


def test_signaled_worker_at_deadline_remains_timeout(monkeypatch, tmp_path):
    from evaluation.pinned_resume import Control

    _worker(monkeypatch, "import time; time.sleep(30)")
    control = Control(tmp_path)

    def stop_after_deadline(proc):
        time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        os.kill(os.getpid(), signal.SIGTERM)

    with control.signals():
        row = _run(
            0.05,
            output_dir=tmp_path,
            cancel_requested=control.immediate,
            on_spawn=stop_after_deadline,
        )
    assert row["terminal"] == "timeout"
    assert not list(tmp_path.iterdir())


def test_worker_signaled_inside_cancel_callback_is_interrupted(monkeypatch, tmp_path):
    _worker(monkeypatch, "import time; time.sleep(30)")
    processes = []

    def cancelled():
        if not processes:
            return False
        processes[0].send_signal(signal.SIGTERM)
        processes[0].wait(timeout=5)
        return True

    with pytest.raises(runner.RowInterrupted, match="signaled worker"):
        _run(
            5,
            output_dir=tmp_path,
            cancel_requested=cancelled,
            on_spawn=processes.append,
        )
    assert processes[0].returncode == -signal.SIGTERM
    assert not list(tmp_path.iterdir())
