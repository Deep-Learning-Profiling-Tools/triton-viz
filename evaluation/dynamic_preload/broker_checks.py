"""Linux containment helpers and explicit P18 diagnostic checks (never auto-run).

This experiment requires Linux pidfds, /proc, PR_SET_PDEATHSIG and subreapers.
The module has no Torch import or atexit registration at import time.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import triton.language as tl

PROTOCOL = "dynamic-preload-normal-exit-v1"
MAX_MESSAGE = 4 * 1024 * 1024


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def script_hashes():
    directory = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("broker_server.py", "broker_adapter.py", "broker_checks.py")
    }


def proc_identity(pid):
    """Read identities without treating a recycled PID as its previous owner."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": int(pid),
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgrp": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def same_process(identity):
    now = proc_identity(identity["pid"])
    return now is not None and now["start_ticks"] == identity["start_ticks"]


def checked_pidfd(identity):
    descriptor = os.pidfd_open(identity["pid"])
    if not same_process(identity):
        os.close(descriptor)
        raise ProcessLookupError("PID identity changed before pidfd acquisition")
    return descriptor


def signal_identity(identity, number):
    try:
        descriptor = checked_pidfd(identity)
    except ProcessLookupError:
        return False
    try:
        signal.pidfd_send_signal(descriptor, number)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)


def exited(descriptor):
    return bool(select.select([descriptor], [], [], 0)[0])


def _prctl(option, argument=0):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(option, argument, 0, 0, 0)
    if result != 0:
        value = ctypes.get_errno()
        raise OSError(value, os.strerror(value))


def parent_death_signal(number, expected_parent):
    _prctl(1, number)  # PR_SET_PDEATHSIG
    if os.getppid() != expected_parent:
        # Never fork / analyze after the controller or broker already disappeared.
        raise RuntimeError("parent died before parent-death protection was armed")


def set_subreaper(enabled):
    value = ctypes.c_int()
    _prctl(37, ctypes.byref(value))  # PR_GET_CHILD_SUBREAPER
    _prctl(36, int(enabled))  # PR_SET_CHILD_SUBREAPER
    return bool(value.value)


def direct_children(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        return [
            int(value)
            for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        ]
    except FileNotFoundError:
        return []


def descendants(pid=None):
    """Only descend from the specified owner, including children that setsid()."""
    pending = direct_children(pid)
    found = {}
    while pending:
        child = pending.pop()
        if child in found:
            continue
        identity = proc_identity(child)
        if identity is not None:
            found[child] = identity
            pending.extend(direct_children(child))
    return list(found.values())


def reap_children(timeout_s=5.0, kill=True, progress=None):
    """Caller must own the entire child tree and be a subreaper.

    Return genuine waitpid receipts. This is used only for diagnostic shutdown
    or after a leader has exited; it never replaces successful process exit.
    """
    deadline = time.monotonic() + timeout_s
    progress = {} if progress is None else progress
    waits = progress.setdefault("waitpid", [])
    signalled_list = progress.setdefault("signalled", [])
    signalled = {item["pid"]: item for item in signalled_list}
    progress["remaining_children"] = direct_children()
    while True:
        members = descendants()
        if kill:
            for member in reversed(members):
                if member["state"] != "Z" and signal_identity(member, signal.SIGKILL):
                    if member["pid"] not in signalled:
                        signalled_list.append(member)
                        signalled[member["pid"]] = member
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            waits.append(
                {
                    "pid": pid,
                    "wait_status": status,
                    "returncode": os.waitstatus_to_exitcode(status),
                    "reaped_monotonic_s": time.monotonic(),
                }
            )
        progress["remaining_children"] = direct_children()
        if not progress["remaining_children"] and not descendants():
            return progress
        if time.monotonic() >= deadline:
            raise RuntimeError(f"unreaped diagnostic descendants: {descendants()}")
        time.sleep(0.005)


def peer_pid(connection):
    pid, uid, _ = struct.unpack(
        "3i",
        connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        ),
    )
    if uid != os.getuid():
        raise PermissionError("broker peer uid differs")
    return pid


def send_message(connection, message):
    raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(raw) > MAX_MESSAGE:
        raise ValueError("oversized broker message")
    connection.sendall(raw)


def receive_message(connection, deadline=None):
    raw = bytearray()
    while not raw.endswith(b"\n"):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("broker reply exceeded its absolute deadline")
            connection.settimeout(remaining)
        value = connection.recv(1)
        if not value:
            raise ConnectionError("broker connection closed before a reply")
        raw.extend(value)
        if len(raw) > MAX_MESSAGE:
            raise ValueError("oversized broker message")
    result = json.loads(raw)
    if not result.get("ok", True):
        raise RuntimeError(result.get("error", "broker rejected request"))
    return result


class FaultObserver:
    """Observers run exclusively in disposable analysis children."""

    def __init__(self, mode):
        self.mode = mode
        self.grandchild_pid = None

    def begin(self):
        if self.mode == "late-exit":
            import atexit

            atexit.register(time.sleep, 2)
        elif self.mode in ("cleanup", "hold", "hold-descendant"):
            import triton_viz
            import z3

            if self.mode == "hold-descendant":
                # Deliberately malformed observer: a descendant escapes the
                # analysis process group. Subreaper cleanup must still find it.
                self.grandchild_pid = os.fork()
                if self.grandchild_pid == 0:
                    os.setsid()
                    while True:
                        time.sleep(1)

            class NativeOwner:
                def __init__(self):
                    self.ast = z3.Int("p18_cleanup_owned") + 1

                def __del__(self):
                    time.sleep(2)
                    del self.ast

            class Launcher:
                def __getitem__(self, grid):
                    def launch(**kwargs):
                        owned = NativeOwner()
                        try:
                            while True:
                                time.sleep(0.001)
                        finally:
                            del owned

                    return launch

            def trace_for_fault(detector):
                def decorate(function):
                    return Launcher()

                return decorate

            triton_viz.trace = trace_for_fault

    def snapshot(self):
        return {
            "partial": True,
            "mode": self.mode,
            "grandchild_pid": self.grandchild_pid,
        }

    def finish(self):
        return {"partial": False, "mode": self.mode}


def _make_observer(mode):
    return FaultObserver(mode)


def _copy_kernel_source(x, out, N: tl.constexpr):
    i = tl.program_id(0)
    tl.store(out + i, tl.load(x + i), i < N)


def worker(root, socket_path, mode, output):
    sys.path.insert(0, str(Path(root).resolve()))
    import torch
    import triton
    import triton.language as tl
    from evaluation import dynamic_subprocess as transport
    from evaluation.spec import LaunchSpec
    from triton_viz.clients.race_detector.ladder import LadderLevel
    from .broker_adapter import install

    globals()["tl"] = tl
    copy_kernel = triton.jit(_copy_kernel_source)
    spec = LaunchSpec(
        name="p18_containment_copy",
        kernel_fn=copy_kernel,
        signature={"x": "*fp32", "out": "*fp32", "N": "constexpr"},
        constexprs={"N": 4},
        grid=(4,),
        make_args=lambda seed: (torch.arange(4, dtype=torch.float32), torch.zeros(4)),
    )
    install(transport, socket_path)
    mismatch = mode in ("input-mismatch", "source-mismatch", "config-mismatch")
    registration_failure = mode.startswith("registration-")
    if mode == "input-mismatch":
        original = transport.input_identity

        def bad_inputs(*args, **kwargs):
            value = original(*args, **kwargs)
            value["x"]["sha256"] = "0" * 64
            return value

        transport.input_identity = bad_inputs
    elif mode == "source-mismatch":
        original = transport.source_identity

        def bad_source():
            value = original()
            value["detector_tree_sha256"] = "0" * 64
            return value

        transport.source_identity = bad_source
    elif mode == "config-mismatch":
        from triton_viz.core.config import config

        config.p18_deliberately_invalid_schema = True
    budget = (
        20.0
        if mode in ("normal", "hold", "hold-descendant")
        or mismatch
        or registration_failure
        else 0.4
    )
    hooks = (
        ()
        if mode == "normal" or mismatch or registration_failure
        else (
            {
                "name": "diagnostic",
                "module": "evaluation.dynamic_preload.broker_checks",
                "factory": "_make_observer",
                "kwargs": {"mode": mode},
            },
        )
    )
    try:
        result = transport.run_dynamic(spec, 0, LadderLevel.L2, budget, hooks=hooks)
    except (transport.DynamicSubprocessError, RuntimeError, OSError) as exc:
        if not mismatch and not registration_failure:
            raise
        expected = (
            (
                "P18 injected post-fork registration failure: "
                + mode.removeprefix("registration-")
            )
            if registration_failure
            else {
                "input-mismatch": "identity mismatch",
                "source-mismatch": "source differs",
                "config-mismatch": "configuration schema differs",
            }[mode]
        )
        assert expected in str(exc), exc
        atomic_json(output, {"expected_rejection": mode, "error": str(exc)})
        return
    assert (
        not mismatch and not registration_failure
    ), "invalid or failed launch was admitted"
    atomic_json(output, result)
    if mode == "normal":
        assert result["status"] == "ok", result
        assert result["execution"]["child_exit_code"] == 0, result
    elif mode not in ("hold", "hold-descendant"):
        execution = result["execution"]
        assert result["status"] == "timeout", result
        assert (
            result["n_reports"] == 0
            and not result["premises"]
            and not result["witnesses"]
        )
        assert 0.4 <= execution["parent_ready_to_reap_s"] < 0.9, execution
        assert execution["child_exit_code"] != 0, execution
        if mode == "late-exit":
            assert execution["child_result_available"], execution
        else:
            assert execution["kill_sent_s"] is not None, execution


def audit(run_dir):
    run_dir = Path(run_dir)
    cost = json.loads((run_dir / "run-cost.json").read_text())
    assert cost["protocol"] == PROTOCOL and cost["remaining_children"] == [], cost
    assert cost["closed"] and cost["broker_reaped"], cost
    assert not cost["cleanup_issues"], cost
    starts = sorted((run_dir / "launches").glob("*.launch.json"))
    for path in starts:
        launch = json.loads(path.read_text())
        receipt_path = path.with_name(path.name.replace(".launch.json", ".reap.json"))
        assert receipt_path.exists(), path
        receipt = json.loads(receipt_path.read_text())
        assert receipt["child"]["start_ticks"] == launch["child"]["start_ticks"]
        assert receipt["launch_id"] == launch["launch_id"]
        assert receipt["waitpid_returned_pid"] == launch["child"]["pid"]
        assert receipt["remaining_children"] == []
        assert not same_process(launch["child"]), launch
    print(
        json.dumps({"protocol": PROTOCOL, "launches": len(starts), "audit": "passed"})
    )


def fault_suite(root, run_dir, python):
    """Explicit serialized correctness controls; call under host admission.

    Never used by import/install. Contains no representative performance rows.
    The SIGSTOP control stops only the broker after GO, proving that the row's
    original hard deadline continues to kill the child without broker RPC.
    """
    from .broker_adapter import BrokerRun, rpc

    run_dir.mkdir(parents=True, exist_ok=True)
    script = "evaluation.dynamic_preload.broker_checks"
    results = []

    def launch_worker(broker, mode, label):
        output = broker.run_dir / (label + ".result.json")
        command = [
            python,
            "-m",
            script,
            "worker",
            "--root",
            str(root),
            "--broker-socket",
            broker.socket_path,
            "--mode",
            mode,
            "--out",
            str(output),
        ]
        stdout = (broker.run_dir / (label + ".stdout.log")).open("x")
        stderr = (broker.run_dir / (label + ".stderr.log")).open("x")
        proc = subprocess.Popen(
            command,
            env=broker.env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        return proc, stdout, stderr

    def wait_worker(proc, timeout=30):
        descriptor = checked_pidfd(proc_identity(proc.pid))
        try:
            if not select.select([descriptor], [], [], timeout)[0]:
                raise TimeoutError(
                    "fault worker did not finish within its controller cap"
                )
            return proc.wait()
        finally:
            os.close(descriptor)

    with BrokerRun(
        root, run_dir / "controls", python=python, allow_fault_injection=True
    ) as broker:
        modes = (
            "normal",
            "late-exit",
            "cleanup",
            "input-mismatch",
            "source-mismatch",
            "config-mismatch",
            "registration-setpgid",
            "registration-identity",
            "registration-pidfd",
            "registration-selector",
            "registration-receipt",
            "normal-after-registration-failures",
        )
        for mode in modes:
            begin = time.monotonic()
            if mode.startswith("registration-"):
                rpc(
                    broker.socket_path,
                    {
                        "op": "inject-next-launch-fault",
                        "stage": mode.removeprefix("registration-"),
                    },
                )
            worker_mode = (
                "normal" if mode == "normal-after-registration-failures" else mode
            )
            proc, stdout, stderr = launch_worker(broker, worker_mode, mode)
            try:
                code = wait_worker(proc)
            finally:
                stdout.close()
                stderr.close()
            broker.record_worker(mode, time.monotonic() - begin, code)
            assert code == 0, (mode, code)
            broker.assert_quiescent()
            pidfds = []
            for descriptor in Path(f"/proc/{broker.proc.pid}/fd").iterdir():
                try:
                    if os.readlink(descriptor) == "anon_inode:[pidfd]":
                        pidfds.append(descriptor.name)
                except FileNotFoundError:
                    pass
            assert len(pidfds) == 1, (
                "owner/child pidfd leak; only controller lease should remain",
                pidfds,
            )
            results.append({"control": mode, "passed": True})
    audit(run_dir / "controls")

    for mode in ("owner-death", "broker-crash", "broker-stop"):
        with BrokerRun(root, run_dir / mode, python=python) as broker:
            begin = time.monotonic()
            worker_mode = "cleanup" if mode == "broker-stop" else "hold-descendant"
            proc, stdout, stderr = launch_worker(broker, worker_mode, mode)
            try:
                deadline = time.monotonic() + 15
                launched = None
                while time.monotonic() < deadline:
                    paths = list((broker.run_dir / "launches").glob("*.launch.json"))
                    if paths:
                        launched = json.loads(paths[0].read_text())
                        directory = Path(launched["request_dir"])
                        if mode == "broker-stop" and (directory / "go.json").exists():
                            break
                        live = directory / "live.json"
                        if mode != "broker-stop" and live.exists():
                            snapshot = json.loads(live.read_text())
                            if snapshot["hooks"]["diagnostic"].get("grandchild_pid"):
                                break
                    if proc.poll() is not None:
                        raise RuntimeError("fault worker exited before injection point")
                    time.sleep(0.001)
                else:
                    raise TimeoutError("fault child did not reach GO")
                injected = time.monotonic()
                if mode == "owner-death":
                    signal_identity(proc_identity(proc.pid), signal.SIGKILL)
                elif mode == "broker-crash":
                    signal_identity(broker.broker_identity, signal.SIGKILL)
                else:
                    signal_identity(broker.broker_identity, signal.SIGSTOP)
                code = wait_worker(proc, 5)
                elapsed = time.monotonic() - injected
                assert code != 0, (mode, code)
                if mode == "broker-stop":
                    assert elapsed < 1.5, (
                        "broker-stop blocked original deadline loop",
                        elapsed,
                    )
                    assert (
                        not same_process(launched["child"])
                        or proc_identity(launched["child"]["pid"])["state"] == "Z"
                    )
                    signal_identity(broker.broker_identity, signal.SIGCONT)
                if mode != "broker-crash":
                    broker.assert_quiescent()
                broker.record_worker(mode, time.monotonic() - begin, code)
                results.append(
                    {
                        "control": mode,
                        "passed": True,
                        "worker_returncode": code,
                        "injection_to_worker_exit_s": elapsed,
                    }
                )
            finally:
                # A failed assertion must not strand a stopped broker.
                signal_identity(broker.broker_identity, signal.SIGCONT)
                stdout.close()
                stderr.close()
        audit(run_dir / mode)
    results.extend(close_fault_checks(root, run_dir, python))
    atomic_json(
        run_dir / "fault-checks.json", {"protocol": PROTOCOL, "checks": results}
    )


def close_fault_checks(root, run_dir, python):
    """Inject interrupted/failed shutdown without discarding historical errors."""
    import broker_adapter as adapter

    results = []

    def extra_owned_child():
        process = subprocess.Popen(
            [python, "-c", "import time; time.sleep(30)"], start_new_session=True
        )
        return process, proc_identity(process.pid)

    def assert_closed(broker, process, identity):
        assert (
            broker.closed and broker.cost["closed"] and broker.cost["broker_reaped"]
        ), broker.cost
        assert (
            not broker.cost["remaining_children"] and not direct_children()
        ), broker.cost
        assert not same_process(identity), identity
        waits = broker.cost["orphan_cleanup"]["waitpid"]
        observed = [value for value in waits if value["pid"] == identity["pid"]]
        assert len(observed) == 1, (identity, broker.cost)
        process.returncode = observed[0]["returncode"]
        assert broker.cost[
            "cleanup_issues"
        ], "injected cleanup fault was erased from history"

    for stage in ("shutdown-rpc", "broker-wait"):
        broker = adapter.BrokerRun(
            root, run_dir / ("close-interrupt-" + stage), python=python
        )
        original_rpc = adapter.rpc
        original_wait = None
        fired = False
        primary = ValueError(
            "P18 intentional primary exception before interrupted close"
        )
        process = identity = None
        try:
            try:
                with broker:
                    process, identity = extra_owned_child()
                    if stage == "shutdown-rpc":

                        def interrupt_rpc(path, message):
                            nonlocal fired
                            if message["op"] == "shutdown" and not fired:
                                fired = True
                                raise KeyboardInterrupt(
                                    "P18 injected shutdown RPC interrupt"
                                )
                            return original_rpc(path, message)

                        adapter.rpc = interrupt_rpc
                    else:
                        original_wait = broker.proc.wait

                        def interrupt_wait(*args, **kwargs):
                            nonlocal fired
                            if not fired:
                                fired = True
                                raise KeyboardInterrupt(
                                    "P18 injected broker wait interrupt"
                                )
                            return original_wait(*args, **kwargs)

                        broker.proc.wait = interrupt_wait
                    raise primary
            except ValueError as observed:
                assert observed is primary, "cleanup replaced the primary exception"
                assert getattr(
                    observed, "__notes__", None
                ), "cleanup interruption was not retained"
            else:
                raise AssertionError("primary exception was suppressed")
        finally:
            adapter.rpc = original_rpc
            if original_wait is not None:
                broker.proc.wait = original_wait
            if not broker.closed:
                broker.close()
        assert fired, "shutdown injection did not execute"
        assert_closed(broker, process, identity)
        results.append(
            {
                "control": "close-interrupt-" + stage,
                "passed": True,
                "primary_exception_preserved": True,
                "cleanup_history_retained": True,
            }
        )

    broker = adapter.BrokerRun(root, run_dir / "close-retry", python=python).__enter__()
    process, identity = extra_owned_child()
    original_reap = adapter.reap_children
    failures = 0
    try:

        def fail_twice(*args, **kwargs):
            nonlocal failures
            if failures < 2:
                failures += 1
                raise RuntimeError("P18 injected temporary orphan reap failure")
            return original_reap(*args, **kwargs)

        adapter.reap_children = fail_twice
        try:
            broker.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed close did not report its cleanup error")
        assert failures == 2 and not broker.closed, broker.cost
        assert same_process(
            identity
        ), "control did not leave a real child for the explicit retry"
    finally:
        adapter.reap_children = original_reap
        broker.close()
    assert_closed(broker, process, identity)
    assert len(broker.cost["cleanup_attempts"]) == 2, broker.cost
    results.append(
        {
            "control": "close-retry",
            "passed": True,
            "incomplete_close_was_retryable": True,
            "cleanup_history_retained": True,
        }
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit")
    check.add_argument("--run-dir", type=Path, required=True)
    test = sub.add_parser("worker")
    test.add_argument("--root", type=Path, required=True)
    test.add_argument("--broker-socket", required=True)
    test.add_argument(
        "--mode",
        choices=(
            "normal",
            "late-exit",
            "cleanup",
            "hold",
            "hold-descendant",
            "input-mismatch",
            "source-mismatch",
            "config-mismatch",
            "registration-setpgid",
            "registration-identity",
            "registration-pidfd",
            "registration-selector",
            "registration-receipt",
        ),
        required=True,
    )
    test.add_argument("--out", type=Path, required=True)
    faults = sub.add_parser("faults")
    faults.add_argument("--root", type=Path, required=True)
    faults.add_argument("--run-dir", type=Path, required=True)
    faults.add_argument("--python", default=sys.executable)
    options = parser.parse_args()
    if options.command == "audit":
        audit(options.run_dir)
    elif options.command == "faults":
        fault_suite(options.root, options.run_dir, options.python)
    else:
        worker(options.root, options.broker_socket, options.mode, options.out)


if __name__ == "__main__":
    main()
