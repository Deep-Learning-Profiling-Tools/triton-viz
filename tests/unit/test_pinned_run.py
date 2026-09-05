"""Pins for the pinned-rerun driver (``evaluation/pinned_run.py``): the
merge rules (a retry that decided replaces the main row; one that did not
leaves a budget timeout with ``pinned_error`` and no verdict; every row
stamped), the section 6 overhead recipe, the section 12 counting (the L1
rung's proved@enum is an analyzed-launch proof), the refusal of a
debugging (worker-reuse) per-corpus file, the rehearsal naming, and an
end-to-end rehearsal on golden_smoke with a budget small enough to
exercise the retry pass.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import pinned_run as pr  # noqa: E402
from evaluation import runner as runner_mod  # noqa: E402
from evaluation.runner import results_header  # noqa: E402
from triton_viz.clients.race_detector.ladder import LadderLevel  # noqa: E402


def _row(name, corpus, verdict, terminal, wall, **extra):
    d = {
        "name": name,
        "corpus": corpus,
        "verdict": verdict,
        "terminal": terminal,
        "wall_s": wall,
    }
    d.update(extra)
    return d


def _write(path: Path, header: dict, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(x) for x in [header, *rows]) + "\n")
    return path


def test_budget_reached():
    assert pr.budget_reached(_row("a", "c", "error", "timeout", 12.0), 180)
    assert pr.budget_reached(_row("a", "c", "race-free", "proved@T1", 180.0), 180)
    assert not pr.budget_reached(_row("a", "c", "race-free", "proved@T1", 179.9), 180)


def test_merge_rules_and_stamps(tmp_path):
    header = results_header("c", 0, {"c_upstream": "u"})
    rows = [
        _row("ok", "c", "race-free", "proved@T1", 3.0),
        _row("slow", "c", "error", "timeout", 180.0, harness_error="exceeded 180s"),
        _row("dead", "c", "error", "timeout", 180.0, harness_error="exceeded 180s"),
    ]
    files = {"c": _write(tmp_path / "c_pinned.jsonl", header, rows)}
    retried = {
        ("c", "slow"): _row("slow", "c", "race-free", "proved@T1-launch", 250.0),
        ("c", "dead"): _row(
            "dead", "c", "error", "timeout", 320.0, harness_error="exceeded 320s"
        ),
    }
    hdr, merged = pr.merge(files, retried, "abc1234", LadderLevel.L0, 180, 320, 0)
    assert hdr["pinned_commit"] == "abc1234" and hdr["worker_reuse"] is False
    assert hdr["corpora"]["c"]["c_upstream"] == "u"
    by = {r["name"]: r for r in merged}
    assert all("wall_s" not in r and r["pinned_commit"] == "abc1234" for r in merged)
    assert by["ok"]["pinned_pass"] == "main" and by["ok"]["pinned_wall_s"] == 3.0
    assert by["slow"]["pinned_pass"] == "retry"
    assert (
        by["slow"]["terminal"] == "proved@T1-launch"
        and by["slow"]["pinned_wall_s"] == 250.0
    )
    assert by["dead"]["pinned_error"] is True and "verdict" not in by["dead"]
    assert (
        by["dead"]["pinned_wall_s"] == 320.0
        and by["dead"]["harness_error"] == "exceeded 320s"
    )


def test_merge_refuses_a_debugging_file_and_a_level_mismatch(tmp_path):
    debug = results_header("c", 0, {}, LadderLevel.L0, None, {"debugging_only": True})
    files = {
        "c": _write(
            tmp_path / "c_debug.jsonl",
            debug,
            [_row("a", "c", "race-free", "proved@T1", 1.0)],
        )
    }
    with pytest.raises(ValueError, match="DEBUGGING"):
        pr.merge(files, {}, "abc", LadderLevel.L0, 180, 320, 0)
    l1 = results_header("c", 0, {}, LadderLevel.L1)
    files = {
        "c": _write(
            tmp_path / "c_l1.jsonl",
            l1,
            [_row("a", "c", "race-free", "proved@enum", 1.0)],
        )
    }
    with pytest.raises(ValueError, match="ladder level"):
        pr.merge(files, {}, "abc", LadderLevel.L0, 180, 320, 0)


def test_overhead_stats_follow_the_section_6_recipe(tmp_path):
    header = results_header("fla", 0, {})
    rows = [
        _row(f"r{i}", "fla", "race-free", "proved@T1", w)
        for i, w in enumerate([1.0, 2.0, 3.0, 4.0, 100.0])
    ]
    rows.append(_row("t", "fla", "error", "timeout", 180.0))
    rows.append(
        _row("edge", "fla", "race-free", "proved@T1", 180.0)
    )  # wall >= budget counts as timeout
    files = {"fla": _write(tmp_path / "fla_pinned.jsonl", header, rows)}
    # litmus corpora are excluded from the statistic
    files["golden_smoke"] = _write(
        tmp_path / "gs.jsonl",
        results_header("golden_smoke", 0, {}),
        [_row("g", "golden_smoke", "race", "race-confirmed", 999.0)],
    )
    s = pr.overhead_stats(files, 180.0)
    assert (s["rows"], s["within_budget"], s["timeouts"]) == (7, 5, 2)
    assert s["median_s"] == 3.0 and s["max_s"] == 100.0
    assert abs(s["p95_s"] - (4.0 + (100.0 - 4.0) * 0.8)) < 1e-9


def test_verdict_table_counts_extents_including_the_l1_rung():
    merged = [
        dict(_row("a", "fla", "race-free", "proved@T0", 0), pinned_wall_s=1.0),
        dict(
            _row("b", "fla", "race-free", "proved@T1+assumes-termination", 0),
            pinned_wall_s=2.0,
        ),
        dict(_row("c", "fla", "race-free", "proved@T1-launch", 0), pinned_wall_s=3.0),
        dict(_row("d", "fla", "race-free", "proved@interp", 0), pinned_wall_s=4.0),
        dict(_row("e", "fla", "race-free", "proved@enum", 0), pinned_wall_s=5.0),
        dict(_row("j", "fla", "race-free", "proved@T1+content", 0), pinned_wall_s=9.0),
        dict(
            _row(
                "k",
                "fla",
                "race-free",
                "proved@T1-launch+assumes-termination+content",
                0,
            ),
            pinned_wall_s=10.0,
        ),
        dict(_row("f", "fla", "race", "race@enum", 0), pinned_wall_s=6.0),
        dict(_row("g", "fla", "abstain", "unsupported", 0), pinned_wall_s=7.0),
        dict(_row("h", "fla", "error", "compile-error", 0), pinned_wall_s=8.0),
        {"name": "i", "corpus": "fla", "pinned_error": True, "pinned_wall_s": 320.0},
    ]
    t = pr.verdict_table(merged)["fla"]
    assert (t["rows"], t["proof"], t["race"], t["abstain"], t["timeout"]) == (
        11,
        7,
        1,
        2,
        1,
    )
    assert (t["any"], t["input"], t["launch"], t["analyzed"]) == (1, 1, 1, 2)
    assert (t["input+content"], t["launch+content"]) == (1, 1)
    assert t["median_s"] == 5.5  # over rows with a verdict: 1..10
    md = pr.summary_markdown(
        {
            "pinned_commit": "x",
            "ladder_level": "L1",
            "seed": 0,
            "row_timeout_s": 200,
            "retry_timeout_s": 320,
        },
        merged,
        {
            "rows": 11,
            "within_budget": 10,
            "timeouts": 1,
            "budget_s": 200,
            "median_s": 4.5,
            "p95_s": 7.0,
            "max_s": 8.0,
        },
        pr.verdict_table(merged),
    )
    assert "| fla | 11 | 7 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 1 | 5.5 |" in md


def test_rehearsal_on_golden_smoke_exercises_the_retry_pass(tmp_path, monkeypatch):
    """End to end on the smallest corpus with a 1 s budget (under the
    per-row process start-up, so every row reaches it): every such row
    is retried at the retry budget and decides; the merged file is
    stamped and named REHEARSAL."""
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pr, "RESULTS_DIR", tmp_path)
    out = pr.run_pinned(
        LadderLevel.L1,
        ("golden_smoke",),
        seed=0,
        row_timeout=1,  # under the per-row start-up: every row is retried
        retry_timeout=120,
        rehearsal=True,
        guard=False,
        log=sys.stderr,
    )
    assert out.name.startswith("REHEARSAL_") and out.name.endswith("_L1.jsonl")
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    header, rows = lines[0], lines[1:]
    assert header["rehearsal"] is True and header["row_timeout_s"] == 1
    assert header["ladder_level"] == "L1" and header["worker_reuse"] is False
    assert len(rows) == 7
    assert all(r["pinned_commit"] == header["pinned_commit"] for r in rows)
    assert all("wall_s" not in r and "pinned_wall_s" in r for r in rows)
    retried = [r for r in rows if r["pinned_pass"] == "retry"]
    assert header["retried_rows"] == len(retried) >= 1
    # every retried row decided (no golden_smoke row needs 120 s)
    assert all("verdict" in r and not r.get("pinned_error") for r in retried)
    assert all(r.get("verdict") in ("race", "race-free") for r in rows)
    summary = out.with_name(out.stem + "_SUMMARY.md").read_text()
    assert "golden_smoke" in summary and "Real-code totals" not in summary
    # the per-corpus file the runner wrote is a protocol dataset
    per_corpus = tmp_path / "golden_smoke_L1_pinned-rehearsal.jsonl"
    assert runner_mod.assert_protocol_dataset(per_corpus)["ladder_level"] == "L1"


def test_pinned_mode_refuses_a_budget_override(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pr, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="rehearsal-only"):
        pr.run_pinned(
            LadderLevel.L0,
            ("golden_smoke",),
            row_timeout=5,
            rehearsal=False,
            guard=False,
        )
