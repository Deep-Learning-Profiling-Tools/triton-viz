"""Scheduler recovery invariants, independent of solver nondeterminism."""

import contextlib
import json
from collections import Counter

import pytest

from evaluation import pinned_manifest as identity
from evaluation import pinned_resume as resume
from evaluation import pinned_service as service
from evaluation import runner
from evaluation.pinned_state import RunStore, StateError


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    config = {
        "ladder_level": "L2",
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
    }
    manifest = {
        "protocol_version": identity.PROTOCOL_VERSION,
        "run_id": "test",
        "config": config,
        "execution_commit": "a" * 40,
        "fingerprints": {},
        "rows": [
            {"corpus": "golden_smoke", "name": name, "spec_hash": name}
            for name in ("fast", "slow")
        ],
        "headers": {
            "golden_smoke": {
                "header": True,
                "corpus": "golden_smoke",
                "ladder_level": "L2",
                "seed": 0,
                "row_timeout_s": 200,
                "fence_order": True,
                "worker_reuse": False,
            }
        },
    }
    path = tmp_path / "run"
    with RunStore.create(path, manifest):
        pass
    specs = {"golden_smoke": {name: name for name in ("fast", "slow")}}
    monkeypatch.setattr(identity, "validate_manifest", lambda saved: specs)
    monkeypatch.setattr(service, "admission", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(service, "assert_quiescent", lambda unit: None)
    calls = Counter()

    def execute(spec, corpus, seed, budget, mutate, level, **kw):
        calls[(spec, budget)] += 1
        timeout = spec == "slow" and budget == 200
        return {
            "name": spec,
            "corpus": corpus,
            "ladder_level": "L2",
            "fence_order": True,
            "verdict": "error"
            if timeout
            else "abstain"
            if budget == 320
            else "race-free",
            "terminal": "timeout"
            if timeout
            else "unsupported"
            if budget == 320
            else "proved@T0",
            "wall_s": 200.1 if timeout else 3.0,
        }

    monkeypatch.setattr(runner, "_run_one", execute)
    return path, manifest, calls, execute


def test_pause_after_commit_then_resume_never_reexecutes_saved(experiment, monkeypatch):
    path, manifest, calls, execute = experiment

    def pausing(*args, **kwargs):
        row = execute(*args, **kwargs)
        if args[0] == "fast":
            resume.request_pause(path)
        return row

    monkeypatch.setattr(runner, "_run_one", pausing)
    assert resume.execute_run(path) == path
    assert RunStore.inspect(path)["main_committed"] == 1
    monkeypatch.setattr(runner, "_run_one", execute)
    out = resume.execute_run(path)
    assert calls == {("fast", 200): 1, ("slow", 200): 1, ("slow", 320): 1}
    rows = [json.loads(line) for line in out.read_text().splitlines()][1:]
    assert rows[1]["verdict"] == "abstain" and rows[1]["pinned_pass"] == "retry"
    assert all("wall_s" not in row for row in rows)
    assert resume.execute_run(path) == out
    assert sum(calls.values()) == 3


def test_immediate_pause_repeats_only_uncommitted_attempt(experiment, monkeypatch):
    path, manifest, calls, execute = experiment

    def interrupted(*args, **kwargs):
        if args[0] == "slow":
            resume.request_pause(path, immediate=True)
            raise runner.RowInterrupted("operator")
        return execute(*args, **kwargs)

    monkeypatch.setattr(runner, "_run_one", interrupted)
    assert resume.execute_run(path) == path
    assert RunStore.inspect(path)["interrupted_attempts"] == 1
    monkeypatch.setattr(runner, "_run_one", execute)
    resume.execute_run(path)
    assert calls[("fast", 200)] == 1


def test_pause_during_admission_is_not_consumed(experiment, monkeypatch):
    path, manifest, calls, execute = experiment

    @contextlib.contextmanager
    def admission(*args, **kwargs):
        resume.request_pause(path)
        yield

    monkeypatch.setattr(service, "admission", admission)
    resume.execute_run(path, consumed_pause_sequence=0)
    assert not calls
    assert RunStore.inspect(path)["main_committed"] == 0


def test_execution_failure_is_not_masked_by_active_attempt(experiment, monkeypatch):
    path, manifest, calls, execute = experiment

    def fail(*args, **kwargs):
        raise OSError("original I/O failure")

    monkeypatch.setattr(runner, "_run_one", fail)
    with pytest.raises(OSError, match="original I/O failure"):
        resume.execute_run(path)
    assert RunStore.inspect(path)["interrupted_attempts"] == 1


def test_incomplete_retries_cannot_publish(experiment):
    path, manifest, calls, execute = experiment
    with RunStore.open(path) as store:
        session = store.new_session({})
        for name in ("fast", "slow"):
            attempt = store.begin_attempt("golden_smoke", name, "main", session, 200)
            store.commit_result(
                attempt, execute(name, "golden_smoke", 0, 200, False, None)
            )
        with pytest.raises(ValueError, match="incomplete retry"):
            resume.publish(store, path)
    assert not (path / "COMPLETE.json").exists()


def test_export_failure_only_reexports_on_resume(experiment, monkeypatch):
    path, manifest, calls, execute = experiment
    write = resume._json

    def fail(path, value):
        if path.name == "COMPLETE.json":
            raise OSError("receipt write failed")
        return write(path, value)

    monkeypatch.setattr(resume, "_json", fail)
    with pytest.raises(OSError, match="receipt write failed"):
        resume.execute_run(path)
    before = calls.copy()
    monkeypatch.setattr(resume, "_json", write)
    out = resume.execute_run(path)
    assert calls == before
    resume.verify_dataset(out)
    out.write_text(out.read_text() + "\n")
    with pytest.raises(ValueError, match="publication hash"):
        resume.verify_dataset(out)


def test_identity_drift_refuses_before_execution(experiment, monkeypatch):
    path, manifest, calls, execute = experiment

    def mismatch(saved):
        raise ValueError("resume identity mismatch: fingerprints")

    monkeypatch.setattr(identity, "validate_manifest", mismatch)
    with pytest.raises(ValueError, match="identity mismatch"):
        resume.execute_run(path)
    assert not calls


def test_pause_during_final_identity_check_prevents_publication(
    experiment, monkeypatch
):
    path, manifest, calls, execute = experiment
    count = 0

    def validate(saved):
        nonlocal count
        count += 1
        if count == 2:
            resume.request_pause(path)
        return {"golden_smoke": {"fast": "fast", "slow": "slow"}}

    monkeypatch.setattr(identity, "validate_manifest", validate)
    assert resume.execute_run(path) == path
    assert not (path / "COMPLETE.json").exists()
    assert RunStore.inspect(path)["main_committed"] == 2


def test_foreign_worker_identity_is_rejected(experiment, monkeypatch):
    path, manifest, calls, execute = experiment

    def wrong(*args, **kwargs):
        return dict(execute(*args, **kwargs), corpus="wrong")

    monkeypatch.setattr(runner, "_run_one", wrong)
    with pytest.raises(StateError, match="corpus"):
        resume.execute_run(path)
    assert RunStore.inspect(path)["main_committed"] == 0


def test_duplicate_service_launch_does_not_overwrite_record(experiment, monkeypatch):
    path, manifest, calls, execute = experiment
    marker = path / "launch.json"
    marker.write_text('{"original": true}')
    monkeypatch.setattr(
        service, "domain_status", lambda p: {"unit": "old", "active": True}
    )
    with pytest.raises(RuntimeError, match="already has an active"):
        service.launch(path, manifest)
    assert marker.read_text() == '{"original": true}'


def test_completed_export_requires_durable_receipt(experiment):
    path, manifest, calls, execute = experiment
    out = resume.execute_run(path)
    (path / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        resume.verify_dataset(out)


def test_start_cli_creates_default_directory_and_dispatches(
    experiment, monkeypatch, tmp_path
):
    _, template, _, _ = experiment
    output_root = tmp_path / "default-results"
    monkeypatch.setattr(runner, "RESULTS_DIR", output_root)
    launched = []

    def build(config, *, run_id, only_names):
        manifest = json.loads(json.dumps(template))
        manifest.update(config=config, run_id=run_id)
        return manifest, {}

    def launch(path, manifest):
        assert path.parent == output_root / "pinned-runs"
        assert path.name == manifest["run_id"]
        assert RunStore.inspect(path)["main_committed"] == 0
        launched.append(path)
        return "test.service"

    monkeypatch.setattr(identity, "build_manifest", build)
    monkeypatch.setattr(service, "launch", launch)
    assert (
        resume.main(
            ["start", "--rehearsal", "--no-load-guard", "--corpora", "golden_smoke"]
        )
        == 0
    )
    assert len(launched) == 1
