"""Crash recovery and protocol invariants of the durable pinned ledger."""

import copy
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.pinned_state import RunLocked, RunStore, StateError, atomic_write  # noqa: E402


def manifest():
    return {
        "protocol_version": "pinned-resume-v1",
        "run_id": "test-run",
        "fingerprints": {},
        "rehearsal": True,
        "config": {
            "ladder_level": "L2",
            "fence_order": True,
            "row_timeout_s": 200,
            "retry_timeout_s": 320,
            "seed": 0,
        },
        "rows": [
            {"corpus": "c", "name": name, "spec_hash": "frozen-" + name}
            for name in ("a", "b")
        ],
    }


def row(name="a", **extra):
    return {
        "corpus": "c",
        "name": name,
        "ladder_level": "L2",
        "fence_order": True,
        "wall_s": 1.25,
        "terminal": "proved@T1",
        "verdict": "race-free",
        **extra,
    }


def begin(store, name="a", slot="main", session=None):
    if session is None:
        session = store.new_session({"test": True})
    budget = 200 if slot == "main" else 320
    return store.begin_attempt("c", name, slot, session, budget), session


def test_saved_row_roundtrip_exact_replay_and_no_replacement(tmp_path):
    path = tmp_path / "run"
    raw = row(payload={"important": [None, True, 0.001], "unicode": "测试"})
    with RunStore.create(path, manifest()) as store:
        attempt, session = begin(store)
        metrics = store.commit_result(attempt, raw)
        assert 0 <= metrics["serialization_s"] <= metrics["total_s"]
        assert 0 <= metrics["commit_s"] <= metrics["total_s"]
        assert store.last_begin_metrics["total_s"] > 0
        assert store.commit_result(attempt, copy.deepcopy(raw))["replayed"]
        with pytest.raises(StateError, match="cannot be replaced"):
            store.commit_result(attempt, row(wall_s=0.1))
        with pytest.raises(StateError, match="accepted completion"):
            begin(store, session=session)
        assert store.result_records("main")[("c", "a")] == {
            "row": raw,
            "attempt_id": attempt,
            "session_id": session,
        }
        store.end_session(session, "paused")
    with RunStore.open(path) as store:
        assert store.results("main") == {("c", "a"): raw}
        assert store.recover_interrupted("crash") == 0
        assert store.status()["main_committed"] == 1
        assert store._con.execute("PRAGMA synchronous").fetchone()[0] == 3
        assert store._con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_retry_eligibility_budgets_and_interrupted_execution(tmp_path):
    with RunStore.create(tmp_path / "run", manifest()) as store:
        attempt, session = begin(store)
        with pytest.raises(StateError, match="one attempt"):
            begin(store, "b", session=session)
        store.commit_result(
            attempt, row(terminal="timeout", verdict="error", wall_s=200.01)
        )
        retry, _ = begin(store, slot="retry", session=session)
        assert store.recover_interrupted("operator immediate pause") == 1
        resumed, resumed_session = begin(store, slot="retry")
        assert retry != resumed
        store.commit_result(resumed, row(wall_s=210.25))
        assert store.status()["retry_committed"] == 1
        assert store.status()["interrupted_attempts"] == 1
        with pytest.raises(StateError, match="budget differs"):
            store.begin_attempt("c", "b", "main", resumed_session, 201)
        b, _ = begin(store, "b", session=resumed_session)
        store.commit_result(b, row("b"))
        with pytest.raises(StateError, match="not eligible"):
            begin(store, "b", "retry", resumed_session)
        with pytest.raises(StateError, match="only a started"):
            store.commit_result(retry, row(wall_s=210.25))


@pytest.mark.parametrize(
    "change",
    [
        {"name": "unexpected"},
        {"corpus": "other"},
        {"ladder_level": "L1"},
        {"fence_order": False},
        {"fence_order": 1},
        {"wall_s": float("nan")},
        {"wall_s": float("inf")},
        {"wall_s": -1},
        {"wall_s": True},
        {"wall_s": "1.0"},
        {"verdict": "invented"},
        {"terminal": ""},
        {"terminal": None},
        {"budget_s": 320},
        {"payload": float("nan")},
    ],
)
def test_malformed_rows_do_not_fill_slot(tmp_path, change):
    with RunStore.create(tmp_path / "run", manifest()) as store:
        attempt, _ = begin(store)
        with pytest.raises(StateError):
            store.commit_result(attempt, row(**change))
        assert store.results("main") == {}
        store.commit_result(attempt, row())


def test_start_is_exclusive_and_manifest_cannot_be_mutated(tmp_path):
    path = tmp_path / "run"
    with RunStore.create(path, manifest()) as store:
        edited = store.manifest
        edited["rows"][0]["name"] = "changed"
        assert store.manifest["rows"][0]["name"] == "a"
        with pytest.raises(StateError, match="already exists"):
            RunStore.create(path, manifest())
        with pytest.raises(RunLocked):
            RunStore.open(path)
        assert RunStore.inspect(path)["main_committed"] == 0
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from evaluation.pinned_state import RunStore; "
                "import sys; RunStore.open(sys.argv[1])",
                str(path),
            ],
            capture_output=True,
        )
        assert child.returncode != 0 and b"another controller" in child.stderr
    with RunStore.open(path):
        pass


@pytest.mark.parametrize(
    "corruption", ["manifest", "row", "wall", "schema", "version", "truncated"]
)
def test_corrupted_storage_fails_closed(tmp_path, corruption):
    path = tmp_path / "run"
    with RunStore.create(path, manifest()) as store:
        attempt, _ = begin(store)
        store.commit_result(attempt, row())
    if corruption == "manifest":
        with (path / "manifest.json").open("a") as f:
            f.write(" ")
    elif corruption == "truncated":
        (path / "checkpoint.sqlite").write_bytes(b"not a sqlite database")
    else:
        with sqlite3.connect(path / "checkpoint.sqlite") as con:
            if corruption in ("row", "wall"):
                con.execute("DROP TRIGGER immutable_results_update")
                if corruption == "row":
                    con.execute("UPDATE results SET row_json='{}'")
                else:
                    con.execute("UPDATE results SET wall_s=999")
                con.execute(
                    "CREATE TRIGGER immutable_results_update BEFORE UPDATE ON results BEGIN\n"
                    "    SELECT RAISE(ABORT,'accepted results are immutable');\nEND"
                )
            elif corruption == "schema":
                con.execute("DROP TRIGGER immutable_results_delete")
            else:
                con.execute("PRAGMA user_version=100")
    with pytest.raises(StateError):
        RunStore.open(path)


def test_sqlite_result_update_and_delete_prohibited(tmp_path):
    with RunStore.create(tmp_path / "run", manifest()) as store:
        attempt, _ = begin(store)
        store.commit_result(attempt, row())
        for sql in ("UPDATE results SET wall_s=0", "DELETE FROM results"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                with store._con:
                    store._con.execute(sql)


def test_failed_transaction_keeps_start_and_no_partial_result(tmp_path):
    with RunStore.create(tmp_path / "run", manifest()) as store:
        attempt, _ = begin(store)
        original_event = store._event

        def fail_event(kind, payload):
            if kind == "result_committed":
                raise RuntimeError("injected storage failure")
            original_event(kind, payload)

        store._event = fail_event
        with pytest.raises(RuntimeError, match="storage failure"):
            store.commit_result(attempt, row())
        assert store.results("main") == {}
        assert store.status()["active_attempts"][0]["state"] == "STARTED"
        store._event = original_event
        store.commit_result(attempt, row())


@pytest.mark.parametrize(
    "crash_point", ["after_begin", "during_commit", "after_commit"]
)
def test_sigkill_retains_prior_completion_and_recovers_open_transaction(
    tmp_path, crash_point
):
    path = tmp_path / "run"
    with RunStore.create(path, manifest()):
        pass
    ready = tmp_path / "ready"
    program = """
import json, os, sys, time
from pathlib import Path
from evaluation.pinned_state import RunStore
path, ready, raw_json, crash_point = sys.argv[1:]
store = RunStore.open(path)
session = store.new_session({"pid": os.getpid()})
a = store.begin_attempt("c", "a", "main", session, 200)
store.commit_result(a, json.loads(raw_json))
b = store.begin_attempt("c", "b", "main", session, 200)
def wait_for_kill():
    Path(ready).write_text("ready")
    time.sleep(60)
raw_b = dict(json.loads(raw_json), name="b")
if crash_point == "during_commit":
    original_event = store._event
    def crash_event(kind, payload):
        original_event(kind, payload)
        if kind == "result_committed":
            wait_for_kill()
    store._event = crash_event
    store.commit_result(b, raw_b)
elif crash_point == "after_commit":
    store.commit_result(b, raw_b)
    wait_for_kill()
else:
    wait_for_kill()
"""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            str(path),
            str(ready),
            json.dumps(row()),
            crash_point,
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.01)
        assert ready.exists()
        os.kill(proc.pid, signal.SIGKILL)
        assert proc.wait(timeout=5) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    with RunStore.open(path) as store:
        committed_b = crash_point == "after_commit"
        expected = {("c", "a"): row()}
        if committed_b:
            expected[("c", "b")] = row("b")
            assert store.status()["active_attempts"] == []
        else:
            assert store.status()["active_attempts"][0]["name"] == "b"
        assert store.results("main") == expected
        assert store.recover_interrupted("controller killed") == int(not committed_b)
        if not committed_b:
            attempt, _ = begin(store, "b")
            store.commit_result(attempt, row("b"))
        assert store.status()["main_committed"] == 2
        assert store.status()["interrupted_attempts"] == int(not committed_b)


def test_metadata_phase_and_atomic_publication(tmp_path):
    path = tmp_path / "run"
    with RunStore.create(path, manifest()) as store:
        store.set_metadata("publication", {"generation": "abc", "count": 2})
        assert store.get_metadata("publication")["count"] == 2
        store.set_phase("FINALIZING")
        with pytest.raises(StateError, match="backwards"):
            store.set_phase("MAIN")
        store.set_phase("COMPLETE")
        with pytest.raises(StateError, match="completed run"):
            begin(store)
    artifact = path / "exports" / "result.jsonl"
    atomic_write(artifact, b"first\n")
    atomic_write(artifact, b"replacement\n")
    assert artifact.read_bytes() == b"replacement\n"
