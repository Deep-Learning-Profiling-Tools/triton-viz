"""Keep on-demand timing and all-frontends measurements distinguishable."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import compare_runs, headline, pinned_manifest, pinned_resume
from evaluation import pinned_run, report, runner
from evaluation.frontend_policy import ALL_FRONTENDS_ENV
from evaluation.pinned_state import RunStore, StateError
from triton_viz.clients.race_detector.ladder import LadderLevel


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    monkeypatch.delenv(ALL_FRONTENDS_ENV, raising=False)
    monkeypatch.setattr(runner, "_versions", lambda: {})


def row(name="a", policy="on-demand"):
    result = {
        "corpus": "tutorials",
        "name": name,
        "ladder_level": "L2",
        "fence_order": True,
        "wall_s": 1.0,
        "verdict": "race-free",
        "terminal": "proved@T1",
        "static": {"status": "ok"},
        "dynamic": {"status": "not-run", "reason": "static-decided", "time_s": None},
    }
    if policy is not None:
        result["frontend_policy"] = policy
    if policy != "on-demand":
        result["dynamic"] = {"status": "ok", "time_s": 1.0, "n_reports": 0}
    return result


def write(path, rows, policy="on-demand"):
    header = {
        "header": True,
        "corpus": "tutorials",
        "ladder_level": "L2",
        "fence_order": True,
    }
    if policy is not None:
        header["frontend_policy"] = policy
    path.write_text("\n".join(json.dumps(value) for value in [header, *rows]) + "\n")
    return path


def manifest(policy="on-demand"):
    config = {
        "ladder_level": "L2",
        "fence_order": True,
        "row_timeout_s": 200,
        "retry_timeout_s": 320,
    }
    if policy is not None:
        config["frontend_policy"] = policy
    return {
        "protocol_version": "pinned-resume-v1",
        "run_id": "policy-test",
        "fingerprints": {},
        "config": config,
        "rows": [{"corpus": "tutorials", "name": "a", "spec_hash": "frozen"}],
    }


def test_header_and_synthetic_timeout_keep_actual_policy(monkeypatch):
    spec = SimpleNamespace(name="a", expected="race-free", pattern="probe")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 200)

    monkeypatch.setattr(runner, "_run_cancellable", timeout)
    for override, expected in ((None, "on-demand"), ("1", "all")):
        if override:
            monkeypatch.setenv(ALL_FRONTENDS_ENV, override)
        header = runner.results_header("tutorials", 0, {}, LadderLevel.L2)
        result = runner._run_one(spec, "tutorials", 0, 200, False, LadderLevel.L2)
        assert header["frontend_policy"] == result["frontend_policy"] == expected
        assert result["terminal"] == "timeout"


def test_runner_names_policies_separately(tmp_path, monkeypatch):
    from evaluation import kernels

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        kernels, "load", lambda name: SimpleNamespace(specs=[], provenance={})
    )
    lazy = runner.run_corpus("tutorials", None, 0, ladder_level=LadderLevel.L2)
    monkeypatch.setenv(ALL_FRONTENDS_ENV, "1")
    complete = runner.run_corpus("tutorials", None, 0, ladder_level=LadderLevel.L2)
    assert lazy.name == "tutorials_L2_on-demand.jsonl"
    assert complete.name == "tutorials_L2.jsonl"
    assert json.loads(lazy.read_text())["frontend_policy"] == "on-demand"
    assert json.loads(complete.read_text())["frontend_policy"] == "all"


def test_frozen_config_detects_execution_policy_drift(monkeypatch):
    config = pinned_resume.official_config(
        LadderLevel.L2, ("tutorials",), 0, None, 320, True, False, "definitive"
    )
    assert config["frontend_policy"] == "on-demand"
    assert ALL_FRONTENDS_ENV in pinned_manifest.ENV_KEYS
    monkeypatch.setenv(ALL_FRONTENDS_ENV, "1")
    with pytest.raises(ValueError, match="frontend policy differs"):
        pinned_manifest.build_manifest(config, run_id="changed")


@pytest.mark.parametrize("saved_policy", ["on-demand", None])
def test_checkpoint_rejects_cross_policy_rows_and_opens_historical_data(
    tmp_path, saved_policy
):
    directory = tmp_path / "run"
    with RunStore.create(directory, manifest(saved_policy)) as store:
        session = store.new_session({})
        attempt = store.begin_attempt("tutorials", "a", "main", session, 200)
        wrong = "all" if saved_policy == "on-demand" else "on-demand"
        with pytest.raises(StateError, match="frontend policy differs"):
            store.commit_result(attempt, row(policy=wrong))
        store.commit_result(attempt, row(policy=saved_policy))
    with RunStore.open(directory) as store:
        assert store.results("main")[("tutorials", "a")]["wall_s"] == 1.0


def test_merge_rejects_cross_policy_main_retry_and_corpus_files(tmp_path):
    path = write(tmp_path / "lazy.jsonl", [row()])
    files = {"tutorials": path}
    header, merged = pinned_run.merge(files, {}, "commit", LadderLevel.L2, 200, 320, 0)
    assert header["frontend_policy"] == merged[0]["frontend_policy"] == "on-demand"
    with pytest.raises(ValueError, match="retry.*frontend policy"):
        pinned_run.merge(
            files,
            {("tutorials", "a"): row(policy="all")},
            "commit",
            LadderLevel.L2,
            200,
            320,
            0,
        )
    write(path, [row(policy=None)])
    with pytest.raises(ValueError, match="row.*frontend policy"):
        pinned_run.merge(files, {}, "commit", LadderLevel.L2, 200, 320, 0)
    write(path, [row()])
    old = write(tmp_path / "old.jsonl", [row(policy=None)], policy=None)
    with pytest.raises(ValueError, match="frontend policy 'all'"):
        pinned_run.merge(
            dict(files, old=old), {}, "commit", LadderLevel.L2, 200, 320, 0
        )
    header, _ = pinned_run.merge(
        {"tutorials": old}, {}, "commit", LadderLevel.L2, 200, 320, 0
    )
    assert header["frontend_policy"] == "all"


def test_reports_never_count_skipped_dynamic_as_an_abstention(tmp_path):
    timeout = row("measured")
    timeout["static"] = {"status": "unsupported"}
    timeout["dynamic"] = {"status": "timeout", "time_s": 60.1}
    missing = row("missing")
    missing.pop("dynamic")
    path = write(tmp_path / "tutorials_L2_on-demand.jsonl", [row(), missing])
    cutile = row("cutile")
    cutile["frontend"] = "cutile"
    cutile["dynamic"] = {
        "status": "unsupported",
        "reason": "cuda.tile has no interpreter",
        "time_s": 0.0,
    }
    write(tmp_path / "liger.jsonl", [timeout, cutile])
    result = headline.headline(tmp_path)
    assert "2 not run" not in result
    assert (
        "1 rows with recorded dynamic results, 1 not run after a static decision, 1 without dynamic results, 1 with no interpreter frontend"
        in result
    )
    assert "**static verdict where the dynamic mode abstains**: 0 — []" in result
    assert "does not measure full frontend complementarity" in result
    assert "frontend policy on-demand" in report.render([path])
    dataset = compare_runs.load_dataset(path)
    assert dataset.stamps()["frontend_policy"] == "on-demand"
    old = compare_runs.load_dataset(
        write(tmp_path / "old.jsonl", [row(policy=None)], None)
    )
    assert old.stamps()["frontend_policy"] == "all"
    mixed = compare_runs.load_dataset(
        write(tmp_path / "mixed.jsonl", [row(policy=None)])
    )
    assert mixed.stamps()["frontend_policy_rows"] == "MIXED all:1"


def test_headline_rejects_policy_mixing_and_header_row_drift(tmp_path):
    write(tmp_path / "tutorials_L2_on-demand.jsonl", [row()])
    old = write(tmp_path / "tutorials_L2.jsonl", [row(policy=None)], None)
    with pytest.raises(
        ValueError, match="mixed frontend policies.*separate results directories"
    ):
        headline.headline(tmp_path)
    old.unlink()
    write(tmp_path / "tutorials_L2_on-demand.jsonl", [row(policy=None)])
    with pytest.raises(ValueError, match="frontend policy differs from header"):
        headline.headline(tmp_path)


def test_legacy_all_frontend_comparison_keeps_measured_abstention(tmp_path):
    measured = row(policy=None)
    measured["dynamic"] = {"status": "timeout", "time_s": 60.1}
    cutile = row("cutile", None)
    cutile["dynamic"] = {
        "status": "unsupported",
        "reason": "cuda.tile has no interpreter",
        "time_s": 0.0,
    }
    write(tmp_path / "tutorials.jsonl", [measured, cutile], None)
    result = headline.headline(tmp_path)
    assert "**static verdict where the dynamic mode abstains**: 1 — ['a']" in result
    assert "1 with no interpreter frontend" in result
