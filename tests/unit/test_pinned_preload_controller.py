"""Preload scheduling and preparation controls, without launching processes."""

import contextlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from evaluation import pinned_manifest as identity
from evaluation import pinned_resume as resume
from evaluation import pinned_service as service
from evaluation import runner
from evaluation.pinned_state import RunStore


@pytest.fixture
def preload_run(tmp_path, monkeypatch):
    events, brokers = [], []

    class Broker:
        def __init__(self, root, run_dir):
            self.run_dir = run_dir
            self.closed = False
            self.broker_identity = {"pid": 2, "start_ticks": 12}
            self.finish_error = self.close_error = None
            brokers.append(self)

        def __enter__(self):
            events.append("enter")
            self.run_dir.mkdir(parents=True)
            return self

        def __exit__(self, kind, value, tb):
            events.append("close")
            if self.close_error:
                if value is not None:
                    value.add_note(str(self.close_error))
                    return False
                raise self.close_error
            self.closed = True
            resume._json(
                self.run_dir / "run-cost.json",
                {
                    "protocol": "dynamic-preload-normal-exit-v1",
                    "closed": True,
                    "broker_reaped": True,
                    "broker_returncode": 0,
                    "remaining_children": [],
                    "cleanup_issues": [],
                    "error": repr(value) if value is not None else None,
                    "run_wall_s": 1.25,
                    "shared_setup_wall_s": 0.1,
                    "shared_shutdown_wall_s": 0.1,
                },
            )

        def finish_row(self, attempt, elapsed):
            events.append("finish")
            assert elapsed >= 0 and elapsed != 200.123
            assert self.row_process.returncode == 0
            if self.finish_error:
                raise self.finish_error
            path = self.run_dir / f"{attempt}.worker.json"
            child = self.run_dir / f"{attempt}.child-reap.json"
            resume._json(path, {"attempt": attempt})
            resume._json(child, {"waitpid_returned_pid": 3})
            return {
                "launch_count": 0,
                "elapsed_wall_s": elapsed,
                "waitpid_verified": True,
                "files": {str(child): identity.file_hash(child)},
                "audit_file": str(path),
                "audit_sha256": identity.file_hash(path),
            }

        def validate_row(self, row, receipt):
            assert row["name"] == "fast" and receipt["waitpid_verified"]

    checks = ModuleType("evaluation.dynamic_preload.broker_checks")
    checks.PROTOCOL = "dynamic-preload-normal-exit-v1"
    adapter = ModuleType("evaluation.dynamic_preload.broker_adapter")
    adapter.BrokerRun = Broker
    monkeypatch.setitem(sys.modules, checks.__name__, checks)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    config = {
        "ladder_level": "L2",
        "frontend_policy": "on-demand",
        "corpora": ["golden_smoke"],
        "seed": 0,
        "row_timeout_s": 200,
        "retry_timeout_s": 320,
        "rehearsal": True,
        "load_guard": False,
        "purpose": "definitive",
        "fence_order": True,
        "jobs": 1,
        "worker_reuse": False,
        **identity.launcher_metadata("preload"),
    }
    manifest = {
        "protocol_version": identity.PROTOCOL_VERSION,
        "run_id": "test",
        "config": config,
        "execution_commit": "a" * 40,
        "fingerprints": {},
        "rows": [{"corpus": "golden_smoke", "name": "fast", "spec_hash": "fast"}],
        "headers": {
            "golden_smoke": {
                "header": True,
                "corpus": "golden_smoke",
                "seed": 0,
                "ladder_level": "L2",
                "frontend_policy": "on-demand",
                "row_timeout_s": 200,
                "fence_order": True,
                "worker_reuse": False,
            }
        },
    }
    path = tmp_path / "run"
    with RunStore.create(path, manifest):
        pass
    monkeypatch.setattr(
        identity, "validate_manifest", lambda saved: {"golden_smoke": {"fast": "fast"}}
    )
    monkeypatch.setattr(service, "admission", lambda *a, **kw: contextlib.nullcontext())

    def quiescent(unit, *, allowed_resident=None):
        events.append("resident-check" if allowed_resident else "empty-check")
        if allowed_resident:
            assert allowed_resident == brokers[-1].broker_identity
        elif brokers:
            assert brokers[-1].closed

    monkeypatch.setattr(service, "assert_quiescent", quiescent)
    original_commit = RunStore.commit_result

    def commit(store, attempt, row):
        assert events[-2:] == ["finish", "resident-check"]
        events.append("commit")
        return original_commit(store, attempt, row)

    monkeypatch.setattr(RunStore, "commit_result", commit)

    def execute(*args, dynamic_broker, **kwargs):
        assert (
            dynamic_broker.row_process is None and dynamic_broker.row_identity is None
        )
        dynamic_broker.row_process = SimpleNamespace(returncode=0)
        dynamic_broker.row_identity = {"pid": 3, "start_ticks": 13}
        return {
            "name": "fast",
            "corpus": "golden_smoke",
            "ladder_level": "L2",
            "frontend_policy": "on-demand",
            "fence_order": True,
            "verdict": "race-free",
            "terminal": "proved@T0",
            "wall_s": 2.0,
        }

    monkeypatch.setattr(runner, "_run_one", execute)
    return path, manifest, events, brokers, execute


def test_preload_reap_before_commit_close_before_publication(preload_run):
    path, manifest, events, brokers, execute = preload_run
    output = resume.execute_run(path)
    assert output.is_file() and brokers[0].closed
    assert events.index("finish") < events.index("commit") < events.index("close")
    receipt = resume.verify_publication(path)
    assert receipt["dynamic_launcher"] == "preload"
    assert any(name.endswith("run-cost.json") for name in receipt["files"])
    assert any(name.endswith("preload-row.json") for name in receipt["files"])
    audit = next((path / "attempts").glob("*/preload-row.json"))
    audit.write_text("{}")
    with pytest.raises(ValueError, match="publication hash"):
        resume.verify_publication(path)


def test_cleanup_failure_never_commits(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run

    def failure(*args, **kwargs):
        row = execute(*args, **kwargs)
        kwargs["dynamic_broker"].finish_error = OSError("missing receipt")
        return row

    monkeypatch.setattr(runner, "_run_one", failure)
    with pytest.raises(OSError, match="missing receipt"):
        resume.execute_run(path)
    assert "commit" not in events and brokers[0].closed
    assert RunStore.inspect(path)["main_committed"] == 0


def test_cancel_cleans_before_recovery_and_resume_uses_new_broker(
    preload_run, monkeypatch
):
    path, manifest, events, brokers, execute = preload_run

    def cancel(*args, **kwargs):
        execute(*args, **kwargs)
        raise runner.RowInterrupted("operator")

    monkeypatch.setattr(runner, "_run_one", cancel)
    assert resume.execute_run(path) == path
    assert events == ["enter", "finish", "close", "empty-check"]
    assert RunStore.inspect(path)["interrupted_attempts"] == 1
    monkeypatch.setattr(runner, "_run_one", execute)
    resume.execute_run(path)
    assert len(brokers) == 2 and brokers[0].run_dir != brokers[1].run_dir
    costs = resume.verify_publication(path)["preload_session_costs"]
    assert len(costs) == 2 and costs[0]["status"] == "interrupted"
    assert costs[0]["error"] == "RowInterrupted('operator')"
    assert sum(cost["broker_run_wall_s"] for cost in costs) == 2.5
    assert all(cost["observed_session_wall_s"] >= 0 for cost in costs)


def test_primary_error_survives_row_cleanup_failure(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run
    original = ValueError("primary")

    def fail(*args, **kwargs):
        execute(*args, **kwargs)
        kwargs["dynamic_broker"].finish_error = KeyboardInterrupt("cleanup interrupt")
        raise original

    monkeypatch.setattr(runner, "_run_one", fail)
    with pytest.raises(ValueError) as raised:
        resume.execute_run(path)
    assert raised.value is original
    assert any("cleanup interrupt" in note for note in original.__notes__)
    assert "commit" not in events


def test_incomplete_close_does_not_acknowledge_recovery(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run
    original = ValueError("primary")

    def fail(*args, **kwargs):
        execute(*args, **kwargs)
        kwargs["dynamic_broker"].close_error = OSError("cannot reap")
        raise original

    monkeypatch.setattr(runner, "_run_one", fail)
    with pytest.raises(ValueError) as raised:
        resume.execute_run(path)
    assert raised.value is original
    assert RunStore.inspect(path)["interrupted_attempts"] == 0
    assert "commit" not in events


def test_publication_checks_nested_receipt_hashes(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run
    publish = resume.publish

    def corrupt(store, run_dir):
        next((path / "preload").rglob("*.child-reap.json")).write_text("{}")
        return publish(store, run_dir)

    monkeypatch.setattr(resume, "publish", corrupt)
    with pytest.raises(ValueError, match="preload audit hash"):
        resume.execute_run(path)
    assert not (path / "COMPLETE.json").exists()


def test_missing_session_cost_blocks_publication(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run
    publish = resume.publish

    def missing(store, run_dir):
        next((path / "preload").rglob("run-cost.json")).unlink()
        return publish(store, run_dir)

    monkeypatch.setattr(resume, "publish", missing)
    with pytest.raises(ValueError, match="cost is missing"):
        resume.execute_run(path)
    assert not (path / "COMPLETE.json").exists()


def test_failed_but_closed_session_is_retained_and_charged(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run

    def fail(*args, **kwargs):
        execute(*args, **kwargs)
        raise ValueError("failed first attempt")

    monkeypatch.setattr(runner, "_run_one", fail)
    with pytest.raises(ValueError, match="failed first attempt"):
        resume.execute_run(path)
    monkeypatch.setattr(runner, "_run_one", execute)
    resume.execute_run(path)
    costs = resume.verify_publication(path)["preload_session_costs"]
    assert costs[0]["status"] == "failed"
    assert costs[0]["session_reason"] == "controller error"
    assert costs[0]["error"] == "ValueError('failed first attempt')"
    assert sum(cost["broker_run_wall_s"] for cost in costs) == 2.5


def test_timeout_row_is_reaped_and_retried_with_same_broker(preload_run, monkeypatch):
    path, manifest, events, brokers, execute = preload_run
    calls = []

    def timeout(*args, **kwargs):
        row = execute(*args, **kwargs)
        calls.append(args[3])
        if args[3] == 200:
            row.update(verdict="error", terminal="timeout", wall_s=200.123)
        return row

    monkeypatch.setattr(runner, "_run_one", timeout)
    resume.execute_run(path)
    assert calls == [200, 320]
    assert len(brokers) == 1 and events.count("finish") == 2
    assert RunStore.inspect(path)["retry_committed"] == 1


def test_prepare_only_never_launches_or_executes(tmp_path, monkeypatch):
    path = tmp_path / "prepared"
    monkeypatch.setattr(resume, "official_config", lambda *args: {})
    monkeypatch.setattr(identity, "build_manifest", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(RunStore, "create", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(
        service, "admission", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(resume, "_load_guard", lambda *args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("prepare-only must not launch or execute")

    monkeypatch.setattr(service, "launch", forbidden)
    monkeypatch.setattr(resume, "execute_run", forbidden)
    assert resume.start_run(None, [], run_dir=path, prepare_only=True) == path


def test_allowed_resident_requires_identity_and_cgroup(monkeypatch):
    checks = ModuleType("evaluation.dynamic_preload.broker_checks")
    resident = {"pid": 77, "start_ticks": 12, "ppid": os.getpid(), "state": "S"}
    checks.proc_identity = lambda pid: dict(resident)
    monkeypatch.setitem(sys.modules, checks.__name__, checks)
    monkeypatch.setattr(service, "_show", lambda unit: {})
    members = [os.getpid(), 77]
    monkeypatch.setattr(service, "_members", lambda info: list(members))
    service.assert_quiescent("owned", allowed_resident=dict(resident))
    with pytest.raises(RuntimeError, match="remain"):
        service.assert_quiescent("owned")
    members.remove(77)
    with pytest.raises(RuntimeError, match="left"):
        service.assert_quiescent("owned", allowed_resident=dict(resident))
    members.append(77)
    with pytest.raises(RuntimeError, match="identity"):
        service.assert_quiescent(
            "owned", allowed_resident={"pid": 77, "start_ticks": 999}
        )


def test_preload_environment_is_explicit_and_does_not_create_run(tmp_path, monkeypatch):
    for key in ("TRITON_INTERPRET", "FLAGGEMS_SOURCE_DIR"):
        monkeypatch.delenv(key, raising=False)
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(key, "prior-cache")
    package = tmp_path / "installed" / "flag_gems"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    distribution = SimpleNamespace(
        files=[Path("flag_gems/__init__.py")],
        locate_file=lambda entry: package.parent / entry,
    )
    monkeypatch.setattr(
        identity.importlib.metadata, "distribution", lambda name: distribution
    )
    path = tmp_path / "new-run"
    identity.prepare_preload_environment(path)
    assert not path.exists()
    assert os.environ["FLAGGEMS_SOURCE_DIR"] == str(package)
    assert os.environ["TRITON_INTERPRET"] == "0"
    assert os.environ["TRITON_CACHE_DIR"] == str(path.parent / "new-run-cache-triton")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(
        path.parent / "new-run-cache-inductor"
    )
    # Model an import eagerly materializing both configured caches.
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir()
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir()
    assert not path.exists()
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    with pytest.raises(ValueError, match="TRITON_INTERPRET"):
        identity.prepare_preload_environment(path)


def test_outer_cost_includes_broker_exit_before_own_checkpoint(tmp_path, monkeypatch):
    events = []

    class Broker:
        run_dir = tmp_path / "preload" / "session"
        closed = False

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("broker-cost-fsync")
            self.closed = True

    ticks = iter([10.0, 15.5])

    def clock():
        events.append("clock")
        return next(ticks)

    def checkpoint(path, value):
        events.append("session-checkpoint")
        assert value["observed_session_wall_s"] == 5.5
        assert value["broker_closed"]

    monkeypatch.setattr(resume.time, "perf_counter", clock)
    monkeypatch.setattr(resume, "_json", checkpoint)
    with resume._preload_session_cost(Broker(), tmp_path, "session", "manifest"):
        events.append("rows-and-gaps")
    assert events == [
        "clock",
        "enter",
        "rows-and-gaps",
        "broker-cost-fsync",
        "clock",
        "session-checkpoint",
    ]
