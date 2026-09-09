"""P18 transport adapter and charged, independently reaped broker controller.

Use BrokerRun inside the existing host experiment admission. Start one context
for a serial run, pass its socket_path to each fresh row's install(), call
assert_quiescent() after each row, and close the context once. Its run-cost.json
includes setup, all elapsed run time, shutdown, actual broker reap and orphan
cleanup. No startup or warmup cost is erased or charged to a hypothetical run.

CLI plan commands can contain the literal {broker_socket}. The CLI does not
acquire host admission itself: launch it as one run_serial.py job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from .broker_checks import (
    PROTOCOL,
    atomic_json,
    checked_pidfd,
    descendants,
    direct_children,
    exited,
    proc_identity,
    reap_children,
    receive_message,
    same_process,
    script_hashes,
    send_message,
    set_subreaper,
    signal_identity,
)

REAP_RECEIPT_TIMEOUT_S = 0.1


def connect(socket_path):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(10)
    try:
        connection.connect(str(socket_path))
    except BaseException:
        connection.close()
        raise
    return connection


def rpc(socket_path, message):
    with connect(socket_path) as connection:
        send_message(connection, message)
        return receive_message(connection)


class ProcessProxy:
    """Only the narrow Popen API used by the unchanged dynamic transport.

    poll()/wait() return a code only after a real waitpid in the broker. A
    result file, process disappearance, or broker EOF never counts as success.
    """

    def __init__(self, arguments, *, env, socket_path, transport_file, hashes):
        if len(arguments) != 5 or arguments[1:4] != [
            "-m",
            "evaluation.dynamic_subprocess",
            "--child",
        ]:
            raise ValueError(f"unexpected P18 transport command: {arguments!r}")
        self.args = arguments
        self.returncode = None
        self.receipt = None
        self.connection = connect(socket_path)
        self.pidfd = None
        self.launch = None
        self.closed = False
        self.broken = None
        owner = proc_identity(os.getpid())
        try:
            send_message(
                self.connection,
                {
                    "op": "launch",
                    "protocol": PROTOCOL,
                    "python": arguments[0],
                    "request_dir": str(Path(arguments[4]).resolve()),
                    "owner_start_ticks": owner["start_ticks"],
                    "env": env,
                    "sys_path": list(sys.path),
                    "cwd": os.getcwd(),
                    "transport_file": str(transport_file),
                    "script_sha256": hashes,
                },
            )
            reply = receive_message(self.connection)
            self.launch = reply["launch"]
            self.pid = self.launch["child"]["pid"]
            self.pidfd = checked_pidfd(self.launch["child"])
        except BaseException:
            self.connection.close()
            raise

    def _exchange(self, operation, **fields):
        if self.broken is not None:
            raise RuntimeError(self.broken)
        try:
            deadline = time.monotonic() + REAP_RECEIPT_TIMEOUT_S
            self.connection.settimeout(REAP_RECEIPT_TIMEOUT_S)
            send_message(
                self.connection,
                {"op": operation, "launch_id": self.launch["launch_id"], **fields},
            )
            reply = receive_message(self.connection, deadline=deadline)
        except (OSError, ConnectionError, RuntimeError) as exc:
            # The row is not the parent and cannot manufacture a reap receipt.
            # Kill its exact known child; BrokerRun will reap adopted orphans.
            if self.pidfd is not None:
                try:
                    signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.broken = f"P18 broker failed; no authoritative reap receipt: {exc}"
            self._close_handles()
            raise RuntimeError(self.broken) from exc
        receipt = reply.get("receipt")
        if receipt is not None:
            if (
                receipt["launch_id"] != self.launch["launch_id"]
                or receipt["child"] != self.launch["child"]
                or receipt["waitpid_returned_pid"] != self.pid
                or os.waitstatus_to_exitcode(receipt["wait_status"])
                != receipt["actual_returncode"]
                or (
                    receipt["returncode"] == 0
                    and (
                        receipt["actual_returncode"] != 0
                        or receipt.get("cleanup_reason")
                        or receipt.get("descendants_at_exit")
                    )
                )
                or receipt["remaining_children"]
            ):
                raise RuntimeError("invalid broker waitpid receipt")
            self.receipt = receipt
            self.returncode = receipt["returncode"]
            self._close_handles()
        return self.returncode

    def _close_handles(self):
        if not self.closed:
            self.connection.close()
            if self.pidfd is not None:
                os.close(self.pidfd)
                self.pidfd = None
            self.closed = True

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.broken is not None:
            raise RuntimeError(self.broken)
        # This path is inside the original hard-deadline supervisory loop.
        # A stopped or wedged broker must not block checking/sending deadlines.
        if not exited(self.pidfd):
            return None
        return self._exchange("poll")

    def wait(self, timeout=None):
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.005)
        return self.returncode

    def send_signal(self, number):
        if self.returncode is None:
            if self.pidfd is None:
                raise RuntimeError(self.broken or "P18 child pidfd is closed")
            signal.pidfd_send_signal(self.pidfd, int(number))

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def __del__(self):
        # Closing a live lease makes the independent broker kill and reap it.
        # No Python shutdown success is inferred by this destructor.
        try:
            self._close_handles()
        except (AttributeError, OSError):
            pass


def install(transport, socket_path):
    """Replace ONLY transport.subprocess, never the shared subprocess module.

    Return a restoration callback. The native transport protocol and all its
    input/kernel/config/source checks, READY/GO files and deadlines stay intact.
    Candidate code identity and real reap receipts live beside run-cost.json.
    """
    if getattr(transport, "_p18_installed", False):
        raise RuntimeError("P18 transport adapter is already installed")
    info = rpc(socket_path, {"op": "info"})["info"]
    hashes = script_hashes()
    if info["protocol"] != PROTOCOL or info["script_sha256"] != hashes:
        raise RuntimeError("P18 broker/adapter protocol or script identity mismatch")
    transport_file = Path(transport.__file__).resolve()
    if transport_file != Path(info["root"]) / "evaluation/dynamic_subprocess.py":
        raise RuntimeError("P18 broker and row use different detector source roots")
    original = transport.subprocess

    def popen(arguments, *, env):
        process = ProcessProxy(
            arguments,
            env=env,
            socket_path=socket_path,
            transport_file=transport_file,
            hashes=hashes,
        )
        launch_id = process.launch["launch_id"]
        transport._p18_launcher_info["launches"].append(
            {
                "launch_id": launch_id,
                "owner": process.launch["owner"],
                "child": process.launch["child"],
                "reap_receipt": str(
                    Path(info["run_dir"]) / "launches" / (launch_id + ".reap.json")
                ),
            }
        )
        return process

    transport.subprocess = SimpleNamespace(Popen=popen)
    transport._p18_installed = True
    transport._p18_launcher_info = {
        "protocol": PROTOCOL,
        "script_sha256": hashes,
        "broker": info["broker"],
        "run_dir": info["run_dir"],
        "reap_receipt_timeout_s": REAP_RECEIPT_TIMEOUT_S,
        "native_transport_protocol": transport.PROTOCOL,
        "running_child_poll_and_signals": "local pidfd; no broker RPC",
        "launches": [],
    }

    def restore():
        transport.subprocess = original
        del transport._p18_installed
        del transport._p18_launcher_info

    restore.launcher_info = dict(transport._p18_launcher_info)
    return restore


class BrokerRun:
    """A run-level controller. Enter with no existing child processes.

    This context must remain above all fresh row subprocesses. It is a Linux
    subreaper so a failed broker's descendants are adopted and actually waited.
    Do not put the broker outside the enclosing experiment service/cgroup.
    """

    def __init__(
        self,
        root,
        run_dir,
        *,
        python=None,
        socket_dir=None,
        env=None,
        startup_timeout_s=60,
        shutdown_timeout_s=10,
        allow_fault_injection=False,
    ):
        self.root = Path(root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.python = sys.executable if python is None else str(python)
        self.socket_dir = socket_dir
        self.env = dict(os.environ if env is None else env)
        self.allow_fault_injection = allow_fault_injection
        self.row_process = None
        self.row_identity = None
        self.startup_timeout_s = startup_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.proc = None
        self.broker_identity = None
        self.socket_temporary = None
        self.socket_path = None
        self.previous_subreaper = None
        self.closed = False
        self.rows = []
        self.started = None
        self.ready = None
        self.error = None
        self._close_started = None
        self._close_attempts = []
        self._cleanup_adopted = {}
        self._cleanup_waits = {}
        self._cleanup_signalled = {}
        self._broker_reaped = False
        self._broker_returncode = None

    def __enter__(self):
        if direct_children():
            raise RuntimeError("BrokerRun must start with no existing children")
        self.started = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("broker-ready.json", "broker-close.json", "run-cost.json"):
            if (self.run_dir / name).exists():
                raise FileExistsError(
                    f"use a fresh broker run directory: {self.run_dir}"
                )
        self.previous_subreaper = set_subreaper(True)
        try:
            self.socket_temporary = tempfile.TemporaryDirectory(
                prefix="p18-", dir=self.socket_dir
            )
            self.socket_path = str(Path(self.socket_temporary.name) / "broker.sock")
            if len(os.fsencode(self.socket_path)) >= 104:
                raise ValueError("socket path too long; use --socket-dir /tmp")
            controller = proc_identity(os.getpid())
            self.stdout = (self.run_dir / "broker.stdout.log").open("x")
            self.stderr = (self.run_dir / "broker.stderr.log").open("x")
            command = [
                self.python,
                "-m",
                "evaluation.dynamic_preload.broker_server",
                "--root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--socket",
                self.socket_path,
                "--controller-pid",
                str(controller["pid"]),
                "--controller-start",
                str(controller["start_ticks"]),
            ]
            if self.allow_fault_injection:
                command.append("--allow-fault-injection")
            self.proc = subprocess.Popen(
                command,
                cwd=self.root,
                env=self.env,
                stdout=self.stdout,
                stderr=self.stderr,
                start_new_session=True,
            )
            self.broker_identity = proc_identity(self.proc.pid)
            while not (self.run_dir / "broker-ready.json").exists():
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"broker preload failed; inspect {self.run_dir / 'broker.stderr.log'}"
                    )
                if time.monotonic() - self.started >= self.startup_timeout_s:
                    raise TimeoutError(
                        "P18 broker startup exceeded its charged setup cap"
                    )
                time.sleep(0.01)
            self.info = rpc(self.socket_path, {"op": "info"})["info"]
            self.ready = time.monotonic()
            self.assert_quiescent()
            return self
        except BaseException as exc:
            self.error = repr(exc)
            self._close_preserving_error(exc)
            raise

    def assert_quiescent(self, timeout_s=5.0):
        if self.proc.poll() is not None:
            raise RuntimeError("broker exited before controller shutdown")
        deadline = time.monotonic() + timeout_s
        while True:
            state = rpc(self.socket_path, {"op": "status"})
            if not state["active"] and not state["children"]:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        if state["active"] or state["children"]:
            raise RuntimeError(
                f"broker has an active or unreaped analysis child: {state}"
            )
        unexpected = set(direct_children()) - {self.proc.pid}
        if unexpected:
            raise RuntimeError(
                f"controller has unexpected/unreaped children: {sorted(unexpected)}"
            )
        return state

    def record_worker(self, label, wall_s, returncode):
        self.rows.append(
            {"label": label, "worker_wall_s": wall_s, "returncode": returncode}
        )
        atomic_json(self.run_dir / "workers.json", self.rows)

    def validate_row(self, row, receipt):
        """Cross-check successful harness output with the controller's reaps."""
        failed = bool(row.get("harness_error")) or row.get("terminal") in (
            "timeout",
            "crash",
            "harness-error",
        )
        details = row.get("dynamic_launcher_info")
        if failed:
            # A killed/crashed row may not have serialized any launcher data.
            # Its independently verified actual launches remain in the audit.
            return
        if not isinstance(details, dict):
            raise RuntimeError("successful preload row lacks its launcher declaration")
        if (
            details.get("protocol") != PROTOCOL
            or details.get("script_sha256") != self.info["script_sha256"]
            or details.get("broker") != self.info["broker"]
            or details.get("run_dir") != str(self.run_dir)
        ):
            raise RuntimeError("row launcher declaration differs from its session")
        declared, observed = details.get("launches"), receipt["launches"]
        if not isinstance(declared, list) or len(declared) != len(observed):
            raise RuntimeError("row launcher count differs from actual reaped children")
        for claimed, actual in zip(declared, observed):
            if (
                claimed.get("launch_id") != actual["launch_id"]
                or claimed.get("child") != actual["child"]
                or not isinstance(claimed.get("owner"), dict)
                or any(
                    claimed["owner"].get(key) != receipt["row_owner"][key]
                    for key in ("pid", "start_ticks")
                )
                or claimed.get("reap_receipt")
                != str(self.run_dir / "launches" / (actual["launch_id"] + ".reap.json"))
            ):
                raise RuntimeError(
                    "row launcher identity differs from its reaped child"
                )
        dynamic = row.get("dynamic", {})
        if dynamic.get("status") == "not-run" and observed:
            raise RuntimeError(
                "static-decided row unexpectedly launched a dynamic child"
            )
        execution = dynamic.get("execution")
        if execution:
            if (
                len(observed) != 1
                or execution.get("child_exit_code") != observed[0]["returncode"]
            ):
                raise RuntimeError(
                    "dynamic result exit differs from its authoritative wait"
                )

    def finish_row(self, attempt, elapsed_wall_s):
        """Bind a reaped row and its dynamic children before durable acceptance.

        The scheduler calls this outside the original row clock. All checking,
        owner-death cleanup and durable receipt writes stay in session run cost.
        This is also required for timeout, cancellation and exception paths.
        """
        process, owner = self.row_process, self.row_identity
        if process is None or owner is None:
            raise RuntimeError("row has no registered process identity")
        if process.pid != owner["pid"]:
            raise RuntimeError("row process and registered identity do not match")
        if process.returncode is None:
            raise RuntimeError("row worker has not been actually waited")
        if same_process(owner):
            raise RuntimeError("row worker still exists after reported wait")
        self.assert_quiescent()
        reply = rpc(self.socket_path, {"op": "row-receipts", "owner": owner})
        launches, files, seen = [], {}, set()
        for record in reply["launches"]:
            launch, receipt = record["launch"], record["receipt"]
            launch_id = launch["launch_id"]
            if (
                launch_id in seen
                or len(launch_id) != 32
                or any(c not in "0123456789abcdef" for c in launch_id)
            ):
                raise RuntimeError("invalid or duplicate dynamic launch ID")
            seen.add(launch_id)
            if (
                launch["protocol"] != PROTOCOL
                or launch["root"] != str(self.root)
                or launch["script_sha256"] != self.info["script_sha256"]
                or launch["broker"] != self.info["broker"]
                or any(
                    launch["owner"][key] != owner[key] for key in ("pid", "start_ticks")
                )
            ):
                raise RuntimeError("dynamic launch does not belong to this row/session")
            child = launch["child"]
            if (
                any(receipt.get(key) != value for key, value in launch.items())
                or receipt["waitpid_returned_pid"] != child["pid"]
                or os.waitstatus_to_exitcode(receipt["wait_status"])
                != receipt["actual_returncode"]
                or receipt["remaining_children"]
                or same_process(child)
            ):
                raise RuntimeError(
                    "dynamic child lacks a valid authoritative wait receipt"
                )
            if receipt["returncode"] == 0 and (
                receipt["actual_returncode"] != 0
                or receipt.get("cleanup_reason")
                or receipt.get("descendants_at_exit")
                or not receipt.get("launch_admitted")
            ):
                raise RuntimeError(
                    "dynamic cleanup was incorrectly reported as success"
                )
            for suffix, expected in (("launch", launch), ("reap", receipt)):
                path = self.run_dir / "launches" / (launch_id + "." + suffix + ".json")
                raw = path.read_bytes()
                if json.loads(raw) != expected:
                    raise RuntimeError(
                        "on-disk dynamic receipt differs from broker record"
                    )
                files[str(path)] = hashlib.sha256(raw).hexdigest()
            launches.append(
                {
                    "launch_id": launch_id,
                    "child": child,
                    "returncode": receipt["returncode"],
                    "actual_returncode": receipt["actual_returncode"],
                    "wait_status": receipt["wait_status"],
                    "waitpid_returned_pid": receipt["waitpid_returned_pid"],
                }
            )
        result = {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "broker": self.info["broker"],
            "row_owner": owner,
            "row_returncode": process.returncode,
            "launch_count": len(launches),
            "launches": launches,
            "files": files,
            "worker_wall_s": elapsed_wall_s,
            "quiescent": True,
        }
        path = self.run_dir / (attempt + ".row.json")
        if Path(attempt).name != attempt or path.exists():
            raise ValueError("attempt must be unique and filename-safe")
        atomic_json(path, result)
        result["audit_file"] = str(path)
        result["audit_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.record_worker(attempt, elapsed_wall_s, process.returncode)
        self.row_process = self.row_identity = None
        return result

    def close(self):
        if self.closed or self.started is None:
            return
        begin = time.monotonic()
        if self._close_started is None:
            self._close_started = begin
        issues = []
        primary = None
        primary_traceback = None
        remaining = None
        attempt = {"started_monotonic_s": begin, "issues": issues, "closed": False}
        self._close_attempts.append(attempt)

        def record_error(stage, exc):
            nonlocal primary, primary_traceback
            issues.append(f"{stage}: {type(exc).__name__}: {exc}")
            if primary is None:
                primary, primary_traceback = exc, exc.__traceback__

        def perform(stage, function, tries=1):
            # A second interrupt or an I/O failure in one cleanup operation
            # must not skip the other mandatory operations in the finally.
            for _ in range(tries):
                try:
                    function()
                    return True
                except BaseException as exc:
                    record_error(stage, exc)
            return False

        def wait_broker(force=False):
            if self.proc is None or self._broker_reaped:
                return
            if force and self.proc.returncode is None:
                identity = self.broker_identity or proc_identity(self.proc.pid)
                if identity is None:
                    raise RuntimeError(
                        "broker identity unavailable before authoritative wait"
                    )
                self.broker_identity = identity
                signal_identity(identity, signal.SIGKILL)
            self._broker_returncode = self.proc.wait(
                timeout=5 if force else self.shutdown_timeout_s
            )
            self._broker_reaped = True

        def cleanup_orphans():
            # Retain identities and receipts across close retries. A receipt
            # write failure must never turn a real wait into guessed success.
            for member in descendants():
                self._cleanup_adopted[member["pid"]] = member
            progress = {"waitpid": [], "signalled": [], "remaining_children": None}
            try:
                reap_children(timeout_s=5, progress=progress)
            finally:
                # The helper updates progress after each actual wait, including
                # partial progress before a later interrupt or cleanup failure.
                for item in progress["waitpid"]:
                    self._cleanup_waits[item["pid"]] = item
                for item in progress["signalled"]:
                    self._cleanup_signalled[item["pid"]] = item
                if self.proc is not None and self.proc.pid in self._cleanup_waits:
                    identity = self._cleanup_adopted.get(self.proc.pid)
                    if (
                        self.broker_identity is None
                        or identity is None
                        or identity["start_ticks"]
                        != self.broker_identity["start_ticks"]
                    ):
                        raise RuntimeError(
                            "adopted broker wait has no matching recorded identity"
                        )
                    waited = self._cleanup_waits[self.proc.pid]
                    self._broker_returncode = waited["returncode"]
                    self.proc.returncode = self._broker_returncode
                    self._broker_reaped = True

        def recover_receipts():
            for launch_path in (self.run_dir / "launches").glob("*.launch.json"):
                receipt_path = launch_path.with_name(
                    launch_path.name.replace(".launch.json", ".reap.json")
                )
                if receipt_path.exists():
                    continue
                launch = json.loads(launch_path.read_text())
                child = launch["child"]
                identity = self._cleanup_adopted.get(child["pid"])
                waited = self._cleanup_waits.get(child["pid"])
                if not (
                    identity
                    and identity["start_ticks"] == child["start_ticks"]
                    and waited
                ):
                    raise RuntimeError(
                        f"no authoritative wait receipt recoverable for {launch['launch_id']}"
                    )
                atomic_json(
                    receipt_path,
                    {
                        **launch,
                        "reaper": "controller-after-broker-failure",
                        "waitpid_returned_pid": waited["pid"],
                        "wait_status": waited["wait_status"],
                        "actual_returncode": waited["returncode"],
                        "returncode": -signal.SIGKILL,
                        "reaped_monotonic_s": waited["reaped_monotonic_s"],
                        "cleanup_reason": "broker-failed",
                        "remaining_children": direct_children(),
                    },
                )

        try:
            if self.proc is not None and not self._broker_reaped:
                if self.proc.poll() is None:
                    rpc(self.socket_path, {"op": "shutdown"})
                wait_broker()
        except BaseException as exc:
            record_error("graceful broker shutdown", exc)
        finally:
            # These stages run even after KeyboardInterrupt in shutdown/wait.
            # Failed forced waits do not prevent the subreaper from collecting
            # the broker itself and every adopted descendant in the next stage.
            broker_done = perform(
                "forced broker kill/wait", lambda: wait_broker(force=True), tries=2
            )
            orphans_done = perform("adopted descendant reap", cleanup_orphans, tries=2)
            receipts_done = perform("authoritative receipt recovery", recover_receipts)
            resources_done = True
            for name in ("stdout", "stderr"):
                handle = getattr(self, name, None)
                if handle is not None:
                    resources_done = (
                        perform(f"close {name}", handle.close) and resources_done
                    )
            if self.socket_temporary is not None:
                resources_done = (
                    perform(
                        "remove broker socket directory", self.socket_temporary.cleanup
                    )
                    and resources_done
                )

            def inspect_remaining():
                nonlocal remaining
                remaining = direct_children()

            inspected = perform("inspect remaining children", inspect_remaining)
            # Do not relinquish adopted-child responsibility when cleanup failed.
            processes_done = (
                inspected
                and not remaining
                and orphans_done
                and (self.proc is None or self._broker_reaped)
            )
            if self.previous_subreaper is not None and processes_done:
                resources_done = (
                    perform(
                        "restore controller subreaper",
                        lambda: set_subreaper(self.previous_subreaper),
                    )
                    and resources_done
                )
            ended = time.monotonic()
            setup = (
                self.ready if self.ready is not None else self._close_started
            ) - self.started
            workers = sum(row["worker_wall_s"] for row in self.rows)
            terminal = bool(processes_done and receipts_done and resources_done)
            attempt.update(
                ended_monotonic_s=ended,
                remaining_children=remaining,
                broker_wait_stage_completed=broker_done,
                closed=terminal,
            )
            self.cost = {
                "protocol": PROTOCOL,
                "diagnostic_only": False,
                "root": str(self.root),
                "script_sha256": None,
                "closed": terminal,
                "broker_reaped": self._broker_reaped,
                "broker_returncode": self._broker_returncode,
                "shared_setup_wall_s": setup,
                "shared_shutdown_wall_s": ended - self._close_started,
                "run_wall_s": ended - self.started,
                "sum_worker_wall_s": workers,
                "other_controller_wall_s": max(
                    0.0,
                    self._close_started - (self.ready or self._close_started) - workers,
                ),
                "worker_count": len(self.rows),
                "workers": self.rows,
                "cost_accounting": "run_wall_s includes actual setup, all rows/gaps, shutdown and reap; do not add it to worker times",
                "remaining_children": remaining,
                "orphan_cleanup": {
                    "waitpid": list(self._cleanup_waits.values()),
                    "signalled": list(self._cleanup_signalled.values()),
                    "remaining_children": remaining,
                },
                "error": self.error,
                "cleanup_attempts": self._close_attempts,
                "cleanup_issues": [],
            }

            def identify_scripts():
                self.cost["script_sha256"] = script_hashes()

            identified = perform("record script identities", identify_scripts)
            self.cost["closed"] = terminal and identified
            attempt["closed"] = self.cost["closed"]

            def write_cost():
                self.cost["cleanup_issues"] = [
                    item for entry in self._close_attempts for item in entry["issues"]
                ]
                atomic_json(self.run_dir / "run-cost.json", self.cost)

            written = perform("write controller cleanup audit", write_cost, tries=2)
            self.closed = bool(terminal and identified and written)
            attempt["closed"] = self.closed
            self.cost["closed"] = self.closed
            self.cost["cleanup_issues"] = [
                item for entry in self._close_attempts for item in entry["issues"]
            ]
        if primary is not None:
            raise primary.with_traceback(primary_traceback)
        if not self.closed:
            raise RuntimeError(
                f"P18 controller cleanup was not fully verified: {issues}; children={remaining}"
            )

    def _close_preserving_error(self, primary):
        try:
            self.close()
        except BaseException as cleanup_error:
            # An admission/row error remains the exception the caller sees.
            # close() records cleanup failures independently in run-cost.json.
            if hasattr(primary, "add_note"):
                primary.add_note(f"P18 cleanup also failed: {cleanup_error!r}")

    def __exit__(self, kind, value, traceback):
        if value is not None:
            self.error = repr(value)
            self._close_preserving_error(value)
        else:
            self.close()
        return False


def run_plan(options):
    jobs = json.loads(options.plan.read_text())
    with BrokerRun(
        options.root,
        options.run_dir,
        python=options.python,
        socket_dir=options.socket_dir,
    ) as broker:
        # The enclosing admission covers the batch, but each row still needs
        # the original pinned load gate. This import belongs to the controller.
        sys.path.insert(0, str(Path(options.root).resolve()))
        from evaluation.pinned_run import load_guard

        for index, job in enumerate(jobs):
            label = job.get("label", f"row-{index:04d}")
            load_guard(True, sys.stdout, label)
            command = [
                item.replace("{broker_socket}", broker.socket_path)
                for item in job["command"]
            ]
            env = dict(broker.env, **job.get("env", {}))
            begin = time.monotonic()
            with (broker.run_dir / (label + ".stdout.log")).open("x") as stdout, (
                broker.run_dir / (label + ".stderr.log")
            ).open("x") as stderr:
                proc = subprocess.Popen(
                    command,
                    cwd=job.get("cwd"),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                identity = proc_identity(proc.pid)
                descriptor = checked_pidfd(identity)
                try:
                    timeout = job.get("timeout_s", 210)
                    if not select.select([descriptor], [], [], timeout)[0]:
                        raise subprocess.TimeoutExpired(command, timeout)
                    code = proc.wait()
                except BaseException:
                    signal_identity(identity, signal.SIGKILL)
                    proc.wait()
                    raise
                finally:
                    os.close(descriptor)
            wall = time.monotonic() - begin
            broker.record_worker(label, wall, code)
            broker.assert_quiescent()
            if code != job.get("expected_returncode", 0):
                raise RuntimeError(f"diagnostic row {label} returned {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--socket-dir", type=Path, default=Path("/tmp"))
    run.add_argument("--python", default=sys.executable)
    options = parser.parse_args()
    run_plan(options)


if __name__ == "__main__":
    main()
