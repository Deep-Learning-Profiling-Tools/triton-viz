"""Session-owned clean preloader, single active analysis child, real waitpid.

Invoke through broker_adapter.BrokerRun so an independent controller is present.
No multiprocessing.Process / forkserver bootstrap and no os._exit is used.
The fork branch unwinds to this file's top level before normal Python exit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import sys
import time
import traceback
import uuid

from .broker_checks import (
    MAX_MESSAGE,
    PROTOCOL,
    atomic_json,
    checked_pidfd,
    descendants,
    direct_children,
    exited,
    parent_death_signal,
    peer_pid,
    proc_identity,
    reap_children,
    script_hashes,
    send_message,
    set_subreaper,
    signal_identity,
)


class ChildLaunch:
    def __init__(self, message, launch, broker_pid):
        self.message, self.launch, self.broker_pid = message, launch, broker_pid


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def loaded_files():
    result = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file():
            path = str(Path(path).resolve())
            result[path] = file_stamp(path)
    return result


def clean_state():
    import torch
    import z3.z3 as z3_impl

    threads = sorted(path.name for path in Path(f"/proc/{os.getpid()}/task").iterdir())
    result = {
        "thread_ids": threads,
        "thread_count": len(threads),
        "cuda_initialized": torch.cuda.is_initialized(),
        "z3_default_context_created": z3_impl._main_ctx is not None,
        "children": direct_children(),
    }
    if (
        len(threads) != 1
        or result["cuda_initialized"]
        or result["z3_default_context_created"]
        or result["children"]
    ):
        raise RuntimeError(f"broker is not clean before fork: {result}")
    return result


class Broker:
    def __init__(self, options):
        self.options = options
        self.pid = os.getpid()
        self.root = options.root.resolve()
        self.run_dir = options.run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "launches").mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.selector = selectors.DefaultSelector()
        self.connections = {}
        self.active = None
        self.records = {}
        self.next_launch_fault = None
        self.stop = False
        self.stop_reason = "requested"
        self.controller = proc_identity(options.controller_pid)
        if (
            self.controller is None
            or self.controller["start_ticks"] != options.controller_start
        ):
            raise RuntimeError("controller identity changed")
        self.controller_fd = checked_pidfd(self.controller)
        self.selector.register(
            self.controller_fd, selectors.EVENT_READ, ("controller", None)
        )
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        parent_death_signal(signal.SIGTERM, options.controller_pid)
        set_subreaper(True)
        # Capture before imports: preloaded libraries may derive state from env.
        self.preload_environment = dict(os.environ)
        sys.path.insert(0, str(self.root))
        begin = time.monotonic()
        # Imports are the preloader's work; keep their module cache side effects.
        for module in ("torch", "cloudpickle", "numpy"):
            importlib.import_module(module)
        self.preload_checks = {"base_imports": clean_state()}
        importlib.import_module("evaluation.harness")
        importlib.import_module("multiprocessing.util")
        self.preload_checks["harness_import"] = clean_state()
        self.preload_s = time.monotonic() - begin
        self.clean = clean_state()
        self.frozen_files = loaded_files()
        self.hashes = script_hashes()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(options.socket))
        os.chmod(options.socket, 0o600)
        self.socket.listen(8)
        self.socket.setblocking(False)
        self.selector.register(self.socket, selectors.EVENT_READ, ("listen", None))
        self.info = {
            "protocol": PROTOCOL,
            "broker": proc_identity(self.pid),
            "controller": self.controller,
            "root": str(self.root),
            "python": str(Path(sys.executable).resolve()),
            "script_sha256": self.hashes,
            "preload_s": self.preload_s,
            "preloaded": ["torch", "cloudpickle", "numpy", "evaluation.harness"],
            "detector_preloaded": False,
            "preload_checks": self.preload_checks,
            "clean_state": self.clean,
            "frozen_module_file_count": len(self.frozen_files),
            "startup_s": time.monotonic() - self.started,
            "socket": str(options.socket),
            "run_dir": str(self.run_dir),
            "normal_python_exit": True,
            "pidfd_required": True,
            "single_active_child": True,
            "fresh_child_per_launch": True,
        }
        atomic_json(self.run_dir / "broker-ready.json", self.info)

    def request_stop(self, number, frame):
        self.stop = True
        self.stop_reason = f"signal-{number}"

    def verify_preload(self, message):
        if message.get("protocol") != PROTOCOL:
            raise RuntimeError("launcher protocol differs")
        clean = clean_state()
        changed = [
            path
            for path, stamp in self.frozen_files.items()
            if not Path(path).exists() or file_stamp(path) != stamp
        ]
        if changed:
            raise RuntimeError(f"preloaded source/binary changed: {changed[:5]}")
        if message["script_sha256"] != self.hashes:
            raise RuntimeError("adapter and broker script identities differ")
        if script_hashes() != self.hashes:
            raise RuntimeError("experiment scripts changed after broker preload")
        if str(Path(message["python"]).resolve()) != self.info["python"]:
            raise RuntimeError("row and broker interpreters differ")
        if (
            Path(message["transport_file"]).resolve()
            != self.root / "evaluation/dynamic_subprocess.py"
        ):
            raise RuntimeError("row and broker detector roots differ")
        # sys.path is passed explicitly and checked by original source_identity.
        # Every other env key must match the environment used during preload.
        expected = {
            key: value
            for key, value in self.preload_environment.items()
            if key != "PYTHONPATH"
        }
        actual = {
            key: value for key, value in message["env"].items() if key != "PYTHONPATH"
        }
        if expected != actual:
            keys = sorted(
                key
                for key in set(expected) | set(actual)
                if expected.get(key) != actual.get(key)
            )
            raise RuntimeError(f"preload environment differs from row for keys: {keys}")
        return clean

    def launch(self, connection, state, message):
        if self.active is not None:
            raise RuntimeError(
                "broker permits only one active child; run rows serially"
            )
        if state["record"] is not None:
            raise RuntimeError("connection already owns a launch")
        owner = proc_identity(peer_pid(connection))
        if owner is None or owner["start_ticks"] != message["owner_start_ticks"]:
            raise RuntimeError("row owner identity changed")
        request_dir = Path(message["request_dir"]).resolve(strict=True)
        if request_dir.stat().st_uid != os.getuid():
            raise PermissionError("request directory is not owned by this user")
        for name in ("request.json", "spec.pkl", "inputs.pt"):
            if not (request_dir / name).is_file():
                raise RuntimeError(f"missing original transport payload: {name}")
        clean = self.verify_preload(message)
        owner_fd = checked_pidfd(owner)
        launch_id = uuid.uuid4().hex
        log_prefix = self.run_dir / "launches" / launch_id
        launch = {
            "protocol": PROTOCOL,
            "launch_id": launch_id,
            "owner": owner,
            "request_dir": str(request_dir),
            "script_sha256": self.hashes,
            "row_cwd": message["cwd"],
            "root": str(self.root),
            "broker": self.info["broker"],
            "fork_monotonic_s": time.monotonic(),
            "clean_state": clean,
            "stdout": str(log_prefix) + ".stdout.log",
            "stderr": str(log_prefix) + ".stderr.log",
        }
        injected_fault, self.next_launch_fault = self.next_launch_fault, None
        launch["injected_registration_fault"] = injected_fault

        def inject(stage):
            if injected_fault == stage:
                raise OSError(f"P18 injected post-fork registration failure: {stage}")

        sys.stdout.flush()
        sys.stderr.flush()
        try:
            child_pid = os.fork()
        except BaseException:
            os.close(owner_fd)
            raise
        if child_pid == 0:
            return ChildLaunch(message, launch, self.pid)
        child_fd = None
        try:
            inject("setpgid")
            os.setpgid(child_pid, child_pid)
            inject("identity")
            child = proc_identity(child_pid)
            if child is None:
                raise RuntimeError(
                    "forked child disappeared before identity registration"
                )
            launch["child"] = child
            inject("pidfd")
            child_fd = checked_pidfd(child)
            record = {
                "launch": launch,
                "child_fd": child_fd,
                "owner_fd": owner_fd,
                "receipt": None,
                "connection": connection,
                "cleanup_reason": None,
                "pidfd_acknowledged": False,
            }
            self.records[launch_id] = record
            self.active = record
            state["record"] = record
            inject("selector")
            self.selector.register(owner_fd, selectors.EVENT_READ, ("owner", record))
            inject("receipt")
            atomic_json(str(log_prefix) + ".launch.json", launch)
            atomic_json(self.run_dir / "active.json", launch)
            send_message(connection, {"ok": True, "launch": launch})
        except BaseException as exc:
            self.abort_registration(
                connection, state, launch, child_pid, child_fd, owner_fd, exc
            )
            raise
        return None

    def abort_registration(
        self, connection, state, launch, child_pid, child_fd, owner_fd, error
    ):
        """A successful fork is owned even if any subsequent registration fails."""
        try:
            if "child" not in launch:
                try:
                    child = proc_identity(child_pid)
                except OSError:
                    child = None
                launch["child"] = child or {
                    "pid": child_pid,
                    "start_ticks": None,
                    "identity_unavailable": "owned unreaped fork PID; registration failed",
                }
            # No wait has occurred and this single-threaded broker is the parent,
            # so this exact fork PID cannot yet have been recycled. This fallback
            # works even when obtaining its pidfd or /proc identity was the error.
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            cleanup = reap_children(kill=True)
            waits = [item for item in cleanup["waitpid"] if item["pid"] == child_pid]
            if len(waits) != 1:
                raise RuntimeError(
                    "registration failure has no actual child waitpid receipt"
                )
            waited = waits[0]
            receipt = {
                **launch,
                "waitpid_returned_pid": child_pid,
                "wait_status": waited["wait_status"],
                "actual_returncode": waited["returncode"],
                "returncode": -signal.SIGKILL,
                "reaped_monotonic_s": waited["reaped_monotonic_s"],
                "cleanup_reason": "post-fork-registration-failed",
                "registration_error": f"{type(error).__name__}: {error}",
                "descendant_cleanup": cleanup,
                "remaining_children": direct_children(),
                "launch_admitted": False,
                "pidfd_acknowledged": False,
            }
            record = {"launch": launch, "receipt": receipt, "connection": connection}
            self.records[launch["launch_id"]] = record
            state["record"] = record
            self.active = None
            atomic_json(
                self.run_dir / "launches" / (launch["launch_id"] + ".launch.json"),
                launch,
            )
            atomic_json(
                self.run_dir / "launches" / (launch["launch_id"] + ".reap.json"),
                receipt,
            )
            atomic_json(self.run_dir / "active.json", {"active": None})
        except BaseException:
            # Stop admitting work; main/independent controller must finish any
            # cleanup that failed. Never silently strand an unregistered child.
            self.stop, self.stop_reason = True, "post-fork-registration-cleanup-failed"
            self.active = None
            state["record"] = None
            raise
        finally:
            for descriptor in (owner_fd, child_fd):
                if descriptor is not None:
                    try:
                        self.selector.unregister(descriptor)
                    except KeyError:
                        pass
                    os.close(descriptor)

    def finish(self, record, reason=None):
        if record["receipt"] is not None:
            return record["receipt"]
        child = record["launch"]["child"]
        if reason:
            record["cleanup_reason"] = reason
            signal_identity(child, signal.SIGKILL)
        elif not record["pidfd_acknowledged"] or not exited(record["child_fd"]):
            return None
        # Keep the leader a zombie until descendants are identified. The leader
        # PID cannot be recycled while we establish and clean this owned tree.
        others = [member for member in descendants() if member["pid"] != child["pid"]]
        cleanup = reap_children(kill=True)
        leader_waits = [
            item for item in cleanup["waitpid"] if item["pid"] == child["pid"]
        ]
        if len(leader_waits) != 1:
            raise RuntimeError(
                "missing actual broker waitpid receipt for analysis child"
            )
        waited = leader_waits[0]
        # A nominal success with escaped/leftover descendants is a harness error.
        effective_code = waited["returncode"]
        if effective_code == 0 and (others or reason):
            effective_code = -signal.SIGKILL
        receipt = {
            **record["launch"],
            "waitpid_returned_pid": waited["pid"],
            "wait_status": waited["wait_status"],
            "actual_returncode": waited["returncode"],
            "returncode": effective_code,
            "reaped_monotonic_s": waited["reaped_monotonic_s"],
            "cleanup_reason": record["cleanup_reason"],
            "descendants_at_exit": others,
            "descendant_cleanup": cleanup,
            "remaining_children": direct_children(),
            "pidfd_acknowledged": record["pidfd_acknowledged"],
            "launch_admitted": True,
        }
        request_dir = Path(record["launch"]["request_dir"])
        receipt["transport_artifacts_at_reap"] = {
            name: (request_dir / name).exists()
            for name in ("ready.json", "go.json", "result.json", "error.json")
        }
        if receipt["remaining_children"]:
            raise RuntimeError("broker still has children after reap")
        record["receipt"] = receipt
        for descriptor in (record["owner_fd"], record["child_fd"]):
            try:
                self.selector.unregister(descriptor)
            except KeyError:
                pass
            os.close(descriptor)
        self.active = None
        atomic_json(
            self.run_dir / "launches" / (receipt["launch_id"] + ".reap.json"), receipt
        )
        atomic_json(self.run_dir / "active.json", {"active": None})
        return receipt

    def dispatch(self, connection, state, message):
        operation = message.get("op")
        if operation == "info":
            send_message(connection, {"ok": True, "info": self.info})
        elif operation in ("shutdown", "status", "inject-next-launch-fault"):
            if peer_pid(connection) != self.controller["pid"]:
                raise PermissionError(
                    "only the owning controller can manage broker lifetime"
                )
            if operation == "shutdown":
                self.stop = True
            elif operation == "inject-next-launch-fault":
                if not self.options.allow_fault_injection:
                    raise RuntimeError(
                        "fault injection is disabled for production runs"
                    )
                if self.active is not None or self.next_launch_fault is not None:
                    raise RuntimeError("cannot arm another post-fork fault")
                stage = message["stage"]
                if stage not in ("setpgid", "identity", "pidfd", "selector", "receipt"):
                    raise ValueError("unknown post-fork registration fault stage")
                self.next_launch_fault = stage
            send_message(
                connection,
                {
                    "ok": True,
                    "active": self.active is not None,
                    "launch_count": len(self.records),
                    "children": direct_children(),
                },
            )
        elif operation == "row-receipts":
            if peer_pid(connection) != self.controller["pid"]:
                raise PermissionError(
                    "only the owning controller can audit row receipts"
                )
            owner = message["owner"]
            records = [
                record
                for record in self.records.values()
                if all(
                    record["launch"]["owner"][key] == owner[key]
                    for key in ("pid", "start_ticks")
                )
            ]
            if any(record["receipt"] is None for record in records):
                raise RuntimeError("row has an unreaped dynamic launch")
            send_message(
                connection,
                {
                    "ok": True,
                    "launches": [
                        {"launch": record["launch"], "receipt": record["receipt"]}
                        for record in records
                    ],
                },
            )
        elif operation == "launch":
            return self.launch(connection, state, message)
        else:
            record = state["record"]
            if (
                record is None
                or message.get("launch_id") != record["launch"]["launch_id"]
            ):
                raise RuntimeError("operation does not match this connection's launch")
            if operation == "signal":
                number = int(message["signal"])
                if number not in (signal.SIGALRM, signal.SIGTERM, signal.SIGKILL):
                    raise ValueError("unsupported child signal")
                if record["receipt"] is None:
                    signal_identity(record["launch"]["child"], number)
            elif operation != "poll":
                raise ValueError("unknown broker operation")
            if operation == "poll" and record["receipt"] is None:
                # ProcessProxy can send poll only after checked_pidfd succeeded.
                # Keep even an early-failed child unreaped until this implicit
                # ACK, so the constructor cannot race automatic waitpid / reuse.
                record["pidfd_acknowledged"] = True
            receipt = self.finish(record)
            send_message(connection, {"ok": True, "receipt": receipt})
        return None

    def disconnect(self, connection):
        state = self.connections.pop(connection, None)
        if state is None:
            return
        if state["record"] is not None and state["record"]["receipt"] is None:
            self.finish(state["record"], "owner-connection-closed")
        self.selector.unregister(connection)
        connection.close()

    def serve(self):
        while not self.stop:
            if self.active is not None:
                self.finish(self.active)
            for key, _ in self.selector.select(0.005):
                kind, value = key.data
                if kind == "controller":
                    self.stop, self.stop_reason = True, "controller-exited"
                    break
                if kind == "owner":
                    self.finish(value, "owner-process-exited")
                elif kind == "listen":
                    connection, _ = self.socket.accept()
                    connection.setblocking(False)
                    state = {"buffer": bytearray(), "record": None}
                    self.connections[connection] = state
                    self.selector.register(
                        connection, selectors.EVENT_READ, ("connection", state)
                    )
                elif kind == "connection":
                    connection = key.fileobj
                    if connection not in self.connections:
                        continue
                    try:
                        chunk = connection.recv(65536)
                        if not chunk:
                            self.disconnect(connection)
                            continue
                        value["buffer"].extend(chunk)
                        if len(value["buffer"]) > MAX_MESSAGE:
                            raise ValueError("oversized broker message")
                        while b"\n" in value["buffer"]:
                            raw, _, remaining = value["buffer"].partition(b"\n")
                            value["buffer"] = bytearray(remaining)
                            outcome = self.dispatch(connection, value, json.loads(raw))
                            if isinstance(outcome, ChildLaunch):
                                return outcome
                    except (Exception, KeyboardInterrupt) as exc:
                        try:
                            send_message(
                                connection,
                                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                            )
                        except OSError:
                            pass
                        self.disconnect(connection)
        return None

    def close(self):
        if os.getpid() != self.pid:
            return
        begin = time.monotonic()
        if self.active is not None:
            self.finish(self.active, self.stop_reason)
        cleanup = reap_children()
        for connection in list(self.connections):
            self.disconnect(connection)
        self.socket.close()
        self.selector.close()
        os.close(self.controller_fd)
        self.options.socket.unlink(missing_ok=True)
        atomic_json(
            self.run_dir / "broker-close.json",
            {
                "protocol": PROTOCOL,
                "broker": self.info["broker"],
                "stop_reason": self.stop_reason,
                "shutdown_s": time.monotonic() - begin,
                "broker_lifetime_s": time.monotonic() - self.started,
                "launch_count": len(self.records),
                "remaining_children": direct_children(),
                "cleanup": cleanup,
                "script_sha256": self.hashes,
            },
        )

    def detach_in_child(self):
        """Invalidate inherited wrappers without modifying shared epoll entries."""
        if os.getpid() == self.pid:
            raise RuntimeError("cannot detach the live parent broker")
        for connection in self.connections:
            connection.close()
        self.socket.close()
        # EpollSelector.close closes its fd and clears its local Python map.
        # Do not unregister: epoll_ctl would modify the parent's shared epoll.
        self.selector.close()
        self.connections.clear()
        self.records.clear()
        self.active = None


def run_child(child, broker):
    """Return to top-level SystemExit; all ordinary Python exit handlers run."""
    parent_death_signal(signal.SIGKILL, child.broker_pid)
    os.setpgid(0, 0)
    broker.detach_in_child()
    # Drop every inherited broker socket, pidfd, selector and controller lease.
    # Clean preload contains no input tensors, accelerator handles or open jobs.
    for entry in list(Path("/proc/self/fd").iterdir()):
        descriptor = int(entry.name)
        if descriptor > 2:
            try:
                os.close(descriptor)
            except OSError:
                pass
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    for descriptor, name in ((1, "stdout"), (2, "stderr")):
        log = os.open(child.launch[name], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(log, descriptor)
        os.close(log)
    os.environ.clear()
    os.environ.update(child.message["env"])
    os.chdir(child.message["cwd"])
    sys.path[:] = child.message["sys_path"]
    sys.argv[:] = [
        "evaluation.dynamic_subprocess",
        "--child",
        child.message["request_dir"],
    ]
    import multiprocessing.util

    # os.register_at_fork hooks ran during os.fork; cover multiprocessing's
    # separate registry too (notably torch's SharedCache lock). Do not invoke
    # Process._bootstrap, which bypasses ordinary CPython finalization.
    multiprocessing.util._finalizer_registry.clear()
    for (_, _, callback), owner in sorted(
        multiprocessing.util._afterfork_registry.items()
    ):
        # Unlike _run_after_forkers(), fail closed if an inherited lock reset
        # raises; silently logging such a failure would admit an unsafe child.
        callback(owner)
    from evaluation import dynamic_subprocess as transport

    path = Path(child.message["request_dir"])
    try:
        transport._child(path)
    except BaseException as exc:
        transport._json(
            path / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        traceback.print_exc()
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--controller-start", type=int, required=True)
    parser.add_argument("--allow-fault-injection", action="store_true")
    options = parser.parse_args()
    broker = None
    child = None
    try:
        broker = Broker(options)
        child = broker.serve()
    finally:
        if broker is not None and os.getpid() == broker.pid:
            broker.close()
    if child is not None:
        return run_child(child, broker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
