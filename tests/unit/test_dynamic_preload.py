"""Preloader admission and receipt checks without creating any child process.

The separately invoked lifecycle controls cover real fork/waitpid behavior.
These tests cannot launch a broker, kernel, or subprocess, even accidentally.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import socket
import subprocess
import sys

import pytest

from evaluation.dynamic_preload import broker_adapter as adapter
from evaluation.dynamic_preload import broker_server as server


@pytest.fixture(autouse=True)
def no_process_or_socket_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mocked preloader unit test attempted a real process/socket")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    if hasattr(os, "fork"):
        monkeypatch.setattr(os, "fork", forbidden)


@pytest.fixture
def admission(monkeypatch, tmp_path):
    root = tmp_path / "detector"
    hashes = {
        name: name + "-sha256"
        for name in ("broker_adapter.py", "broker_checks.py", "broker_server.py")
    }
    original = SimpleNamespace(Popen=object())
    transport = SimpleNamespace(
        __file__=str(root / "evaluation/dynamic_subprocess.py"),
        subprocess=original,
        PROTOCOL="dynamic-spawn-v1",
    )
    info = {
        "protocol": adapter.PROTOCOL,
        "script_sha256": hashes,
        "root": str(root),
        "broker": {"pid": 123, "start_ticks": 456},
        "run_dir": str(tmp_path / "run"),
    }
    monkeypatch.setattr(adapter, "rpc", lambda *args: {"info": info})
    monkeypatch.setattr(adapter, "script_hashes", lambda: dict(hashes))
    return SimpleNamespace(
        transport=transport, original=original, info=info, hashes=hashes
    )


def test_install_is_local_and_restores_after_exception(admission):
    global_popen = subprocess.Popen
    restore = adapter.install(admission.transport, "/unused.sock")
    try:
        assert admission.transport.subprocess is not admission.original
        assert callable(admission.transport.subprocess.Popen)
        assert subprocess.Popen is global_popen
        assert restore.launcher_info["native_transport_protocol"] == "dynamic-spawn-v1"
        assert restore.launcher_info["script_sha256"] == admission.hashes
        with pytest.raises(ValueError, match="body failed"):
            try:
                raise ValueError("body failed")
            finally:
                restore()
    finally:
        # Avoid assuming that a restoration callback is idempotent.
        if admission.transport.subprocess is not admission.original:
            restore()
    assert admission.transport.subprocess is admission.original
    assert subprocess.Popen is global_popen
    assert not getattr(admission.transport, "_p18_installed", False)
    assert not hasattr(admission.transport, "_p18_launcher_info")


def test_duplicate_install_keeps_first_owner(admission):
    restore = adapter.install(admission.transport, "/unused.sock")
    installed = admission.transport.subprocess
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            adapter.install(admission.transport, "/unused.sock")
        assert admission.transport.subprocess is installed
    finally:
        restore()
    assert admission.transport.subprocess is admission.original


@pytest.mark.parametrize("changed", ["protocol", "hashes", "root"])
def test_install_rejects_unbound_broker_before_mutating_transport(admission, changed):
    if changed == "protocol":
        admission.info["protocol"] = "other-protocol"
    elif changed == "hashes":
        admission.info["script_sha256"] = {"broker_server.py": "wrong"}
    else:
        admission.info["root"] += "-other"
    with pytest.raises(RuntimeError, match="identity mismatch|source roots"):
        adapter.install(admission.transport, "/unused.sock")
    assert admission.transport.subprocess is admission.original
    assert not getattr(admission.transport, "_p18_installed", False)


@pytest.fixture
def verifier(monkeypatch, admission):
    broker = object.__new__(server.Broker)
    broker.root = Path(admission.info["root"])
    broker.hashes = admission.hashes
    broker.info = {"python": str(Path(sys.executable).resolve())}
    broker.frozen_files = {}
    broker.preload_environment = {
        "TRITON_INTERPRET": "0",
        "TORCHINDUCTOR_CACHE_DIR": "/shared/inductor",
        "TRITON_CACHE_DIR": "/shared/triton",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "/controller/path",
    }
    clean = {
        "thread_count": 1,
        "cuda_initialized": False,
        "z3_default_context_created": False,
        "children": [],
    }
    monkeypatch.setattr(server, "clean_state", lambda: dict(clean))
    monkeypatch.setattr(server, "script_hashes", lambda: dict(admission.hashes))
    message = {
        "protocol": server.PROTOCOL,
        "script_sha256": dict(admission.hashes),
        "python": sys.executable,
        "transport_file": str(broker.root / "evaluation/dynamic_subprocess.py"),
        "env": dict(broker.preload_environment),
    }
    return broker, message, clean


def test_preload_environment_allows_only_explicit_pythonpath_exception(verifier):
    broker, message, clean = verifier
    message["env"]["PYTHONPATH"] = "/row/path"
    assert broker.verify_preload(message) == clean


@pytest.mark.parametrize(
    "key",
    [
        "TRITON_INTERPRET",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "OMP_NUM_THREADS",
    ],
)
@pytest.mark.parametrize("change", ["different", "missing"])
def test_preload_rejects_changed_import_environment(verifier, key, change):
    broker, message, _ = verifier
    if change == "different":
        message["env"][key] = "different"
    else:
        del message["env"][key]
    with pytest.raises(RuntimeError, match="preload environment differs") as caught:
        broker.verify_preload(message)
    assert key in str(caught.value)


@pytest.mark.parametrize(
    "field", ["protocol", "python", "transport_file", "script_sha256"]
)
def test_preload_rejects_changed_launch_identity(verifier, field):
    broker, message, _ = verifier
    message[field] = {} if field == "script_sha256" else "/different"
    with pytest.raises(RuntimeError):
        broker.verify_preload(message)


def test_preload_rejects_module_modified_since_import(verifier, tmp_path):
    broker, message, _ = verifier
    module = tmp_path / "loaded.py"
    module.write_text("value = 1\n")
    broker.frozen_files[str(module)] = server.file_stamp(module)
    module.write_text("value = 100\n")
    with pytest.raises(RuntimeError, match="preloaded source/binary changed"):
        broker.verify_preload(message)


def test_preload_rejects_changed_broker_files(verifier, monkeypatch):
    broker, message, _ = verifier
    monkeypatch.setattr(server, "script_hashes", lambda: {"changed": "hash"})
    with pytest.raises(RuntimeError, match="scripts changed"):
        broker.verify_preload(message)


@pytest.fixture
def proxy(monkeypatch):
    process = object.__new__(adapter.ProcessProxy)
    process.args = ["unused"]
    process.pid = 123
    process.pidfd = 456
    process.launch = {
        "launch_id": "launch-one",
        "child": {"pid": 123, "start_ticks": 789},
    }
    process.returncode = None
    process.receipt = None
    process.broken = None
    process.closed = False
    process.connection = SimpleNamespace(settimeout=lambda timeout: None)
    process._close_handles = lambda: setattr(process, "closed", True)
    monkeypatch.setattr(adapter, "send_message", lambda *args: None)
    receipt = {
        "launch_id": "launch-one",
        "child": dict(process.launch["child"]),
        "waitpid_returned_pid": 123,
        "wait_status": 0,
        "actual_returncode": 0,
        "returncode": 0,
        "remaining_children": [],
    }
    return process, receipt


def test_running_child_poll_never_waits_for_broker(proxy, monkeypatch):
    process, _ = proxy
    monkeypatch.setattr(adapter, "exited", lambda descriptor: False)
    monkeypatch.setattr(
        process, "_exchange", lambda *args: pytest.fail("live child used broker RPC")
    )
    assert process.poll() is None
    assert process.returncode is None and process.receipt is None


def test_missing_reap_receipt_cannot_become_success(proxy, monkeypatch):
    process, _ = proxy
    monkeypatch.setattr(
        adapter, "receive_message", lambda *args, **kwargs: {"receipt": None}
    )
    assert process._exchange("poll") is None
    assert process.returncode is None and process.receipt is None
    assert not process.closed


@pytest.mark.parametrize(
    "field,value",
    [
        ("launch_id", "another-launch"),
        ("child", {"pid": 123, "start_ticks": 790}),
        ("waitpid_returned_pid", 124),
        ("wait_status", 9),
        ("actual_returncode", -9),
        ("cleanup_reason", "owner-process-exited"),
        ("descendants_at_exit", [{"pid": 125}]),
        ("remaining_children", [125]),
    ],
)
def test_invalid_reap_receipt_cannot_become_success(proxy, monkeypatch, field, value):
    process, receipt = proxy
    receipt[field] = value
    monkeypatch.setattr(
        adapter, "receive_message", lambda *args, **kwargs: {"receipt": receipt}
    )
    with pytest.raises(RuntimeError, match="invalid broker waitpid receipt"):
        process._exchange("poll")
    assert process.returncode is None and process.receipt is None


def test_valid_reap_receipt_is_required_for_exit_code(proxy, monkeypatch):
    process, receipt = proxy
    monkeypatch.setattr(
        adapter, "receive_message", lambda *args, **kwargs: {"receipt": receipt}
    )
    assert process._exchange("poll") == 0
    assert process.receipt is receipt and process.closed


def test_broker_eof_kills_exact_child_and_refuses_success(proxy, monkeypatch):
    process, _ = proxy
    signalled = []

    def eof(*args, **kwargs):
        raise ConnectionError("controller connection lost")

    monkeypatch.setattr(adapter, "receive_message", eof)
    monkeypatch.setattr(
        adapter.signal,
        "pidfd_send_signal",
        lambda fd, number: signalled.append((fd, number)),
    )
    with pytest.raises(RuntimeError, match="no authoritative reap receipt"):
        process._exchange("poll")
    assert signalled == [(456, adapter.signal.SIGKILL)]
    assert process.returncode is None and process.receipt is None and process.closed
    with pytest.raises(RuntimeError, match="no authoritative reap receipt"):
        process.poll()


@pytest.fixture
def row_audit(monkeypatch, admission, tmp_path):
    broker = object.__new__(adapter.BrokerRun)
    broker.root = Path(admission.info["root"])
    broker.run_dir = tmp_path / "session"
    (broker.run_dir / "launches").mkdir(parents=True)
    broker.info = admission.info
    broker.socket_path = "/unused.sock"
    broker.row_process = SimpleNamespace(pid=321, returncode=0)
    broker.row_identity = {"pid": 321, "start_ticks": 7}
    events = []
    broker.assert_quiescent = lambda: events.append("quiescent")
    broker.record_worker = lambda *args: events.append(("record", args))
    launch = {
        "launch_id": "a" * 32,
        "protocol": adapter.PROTOCOL,
        "root": str(broker.root),
        "script_sha256": admission.hashes,
        "broker": admission.info["broker"],
        "owner": dict(broker.row_identity),
        "child": {"pid": 322, "start_ticks": 8},
    }
    receipt = {
        **launch,
        "waitpid_returned_pid": 322,
        "wait_status": 0,
        "actual_returncode": 0,
        "returncode": 0,
        "remaining_children": [],
        "launch_admitted": True,
    }
    launches = [{"launch": launch, "receipt": receipt}]
    monkeypatch.setattr(adapter, "rpc", lambda *args: {"launches": launches})
    monkeypatch.setattr(adapter, "same_process", lambda identity: False)
    for suffix, value in (("launch", launch), ("reap", receipt)):
        (
            broker.run_dir / "launches" / (launch["launch_id"] + "." + suffix + ".json")
        ).write_text(json.dumps(value))
    return SimpleNamespace(
        broker=broker, launch=launch, receipt=receipt, launches=launches, events=events
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_finish_row_binds_actual_owner_and_disk_receipts(row_audit, dynamic):
    control = row_audit
    if not dynamic:
        control.launches.clear()
    result = control.broker.finish_row("attempt-1", 2.5)
    assert result["row_owner"] == {"pid": 321, "start_ticks": 7}
    assert result["launch_count"] == int(dynamic)
    assert result["quiescent"] and result["worker_wall_s"] == 2.5
    assert result["row_returncode"] == 0
    assert control.events == ["quiescent", ("record", ("attempt-1", 2.5, 0))]
    assert control.broker.row_process is None and control.broker.row_identity is None
    assert len(result["files"]) == 2 * int(dynamic)
    for filename, expected_hash in result["files"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected_hash
    assert (
        hashlib.sha256(Path(result["audit_file"]).read_bytes()).hexdigest()
        == result["audit_sha256"]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "unwaited-row",
        "wrong-row-pid",
        "row-still-alive",
        "foreign-owner",
        "wrong-wait-status",
        "false-success",
        "changed-disk-receipt",
        "missing-disk-receipt",
    ],
)
def test_finish_row_refuses_unverifiable_completion(row_audit, monkeypatch, failure):
    control = row_audit
    broker = control.broker
    receipt_path = (
        broker.run_dir / "launches" / (control.launch["launch_id"] + ".reap.json")
    )
    if failure == "unwaited-row":
        broker.row_process.returncode = None
    elif failure == "wrong-row-pid":
        broker.row_process.pid += 1
    elif failure == "row-still-alive":
        monkeypatch.setattr(
            adapter, "same_process", lambda identity: identity["pid"] == 321
        )
    elif failure == "foreign-owner":
        control.launch["owner"]["start_ticks"] += 1
    elif failure == "wrong-wait-status":
        control.receipt["wait_status"] = 9
    elif failure == "false-success":
        control.receipt["wait_status"] = 9
        control.receipt["actual_returncode"] = -9
    elif failure == "changed-disk-receipt":
        receipt_path.write_text("{}")
    else:
        receipt_path.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        broker.finish_row("attempt-1", 2.5)
    assert not (broker.run_dir / "attempt-1.row.json").exists()
    assert not any(
        isinstance(event, tuple) and event[0] == "record" for event in control.events
    )


def test_runner_forwards_socket_and_registers_before_original_callback(
    monkeypatch, tmp_path
):
    from evaluation import runner
    from evaluation.dynamic_preload import broker_checks

    broker = SimpleNamespace(
        socket_path="/unused.sock", row_process=None, row_identity=None
    )
    identity = {"pid": 123, "start_ticks": 45}
    process = SimpleNamespace(pid=123, returncode=0, stderr="", stdout="")
    callbacks = []
    spec = SimpleNamespace(name="mock-static", expected="race-free", pattern="unit")
    monkeypatch.setattr(broker_checks, "proc_identity", lambda pid: dict(identity))

    def original_callback(observed):
        assert observed is process and broker.row_process is process
        assert broker.row_identity == identity
        callbacks.append("original")

    def fake_run(command, timeout, cancel_requested, on_spawn):
        assert (
            command[command.index("--dynamic-broker-socket") + 1] == broker.socket_path
        )
        assert timeout == 200
        on_spawn(process)
        Path(command[command.index("--out") + 1]).write_text(
            json.dumps(
                {
                    "name": spec.name,
                    "verdict": "race-free",
                    "terminal": "proved@T1-launch+content",
                }
            )
        )
        return process

    monkeypatch.setattr(runner, "_run_cancellable", fake_run)
    result = runner._run_one(
        spec,
        "mock-corpus",
        0,
        200,
        False,
        on_spawn=original_callback,
        output_dir=tmp_path,
        dynamic_broker=broker,
    )
    assert result["terminal"] == "proved@T1-launch+content"
    assert callbacks == ["original"]
    assert broker.row_process is process and broker.row_identity == identity
    assert not list(tmp_path.iterdir())


@pytest.fixture
def declared_row(row_audit):
    broker = row_audit.broker
    launch = row_audit.launch
    receipt = {
        "row_owner": {**broker.row_identity, "state": "R"},
        "launches": [
            {
                "launch_id": launch["launch_id"],
                "child": launch["child"],
                "returncode": 0,
            }
        ],
    }
    row = {
        "verdict": "race-free",
        "terminal": "proved@T0",
        "dynamic": {"status": "ok", "execution": {"child_exit_code": 0}},
        "dynamic_launcher_info": {
            "protocol": adapter.PROTOCOL,
            "script_sha256": broker.info["script_sha256"],
            "broker": broker.info["broker"],
            "run_dir": str(broker.run_dir),
            "launches": [
                {
                    "launch_id": launch["launch_id"],
                    "child": launch["child"],
                    "owner": {**broker.row_identity, "state": "S"},
                    "reap_receipt": str(
                        broker.run_dir
                        / "launches"
                        / (launch["launch_id"] + ".reap.json")
                    ),
                }
            ],
        },
    }
    return broker, row, receipt


def test_validate_row_allows_owner_state_change_but_preserves_identity(declared_row):
    broker, row, receipt = declared_row
    assert row["dynamic_launcher_info"]["launches"][0]["owner"]["state"] == "S"
    assert receipt["row_owner"]["state"] == "R"
    broker.validate_row(row, receipt)


@pytest.mark.parametrize(
    "failure",
    [
        "owner-pid",
        "owner-start",
        "missing-declaration",
        "session-hash",
        "reap-path",
        "launch-count",
        "exit-code",
        "static-launched",
    ],
)
def test_validate_row_rejects_unbound_or_inconsistent_claim(declared_row, failure):
    broker, row, receipt = declared_row
    details = row["dynamic_launcher_info"]
    if failure == "owner-pid":
        details["launches"][0]["owner"]["pid"] += 1
    elif failure == "owner-start":
        details["launches"][0]["owner"]["start_ticks"] += 1
    elif failure == "missing-declaration":
        del row["dynamic_launcher_info"]
    elif failure == "session-hash":
        details["script_sha256"] = {}
    elif failure == "reap-path":
        details["launches"][0]["reap_receipt"] = "/different"
    elif failure == "launch-count":
        details["launches"].clear()
    elif failure == "exit-code":
        row["dynamic"]["execution"]["child_exit_code"] = -9
    else:
        row["dynamic"] = {"status": "not-run"}
    with pytest.raises(RuntimeError):
        broker.validate_row(row, receipt)


def test_validate_static_row_requires_no_launch_and_keeps_declaration(declared_row):
    broker, row, receipt = declared_row
    row["dynamic"] = {"status": "not-run"}
    row["dynamic_launcher_info"]["launches"].clear()
    receipt["launches"].clear()
    broker.validate_row(row, receipt)


def test_failed_row_can_lack_a_declaration_after_independent_reap(declared_row):
    broker, row, receipt = declared_row
    row.update(verdict="error", terminal="timeout", harness_error="exceeded 200s")
    del row["dynamic_launcher_info"]
    broker.validate_row(row, receipt)
