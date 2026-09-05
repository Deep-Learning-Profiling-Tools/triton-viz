"""Pins for the runner's process reuse (``--reuse-workers``): served
workers give the SAME rows as per-row subprocesses, the parent keeps the
per-row budget (a silent worker is killed and the row is a timeout) and
crash containment (a dead worker is a crash row and the next row gets a
fresh worker), workers are recycled on the row bound, and the header
stamps the protocol.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import runner as runner_mod  # noqa: E402
from evaluation.runner import (  # noqa: E402
    _run_one,
    _run_one_reused,
    _Worker,
    results_header,
    row_timeout_s,
)
from triton_viz.clients.race_detector.ladder import LadderLevel  # noqa: E402


@pytest.fixture(scope="module")
def smoke_specs():
    from evaluation.kernels import load

    return load("golden_smoke").specs


def test_served_rows_match_subprocess_rows(tmp_path, smoke_specs):
    worker = _Worker(tmp_path / "worker.log")
    try:
        served = [
            _run_one_reused(
                worker,
                s,
                "golden_smoke",
                0,
                row_timeout_s(LadderLevel.L1),
                False,
                LadderLevel.L1,
            )
            for s in smoke_specs
        ]
    finally:
        worker.stop()
    assert worker.rows == len(smoke_specs)
    direct = [
        _run_one(
            s, "golden_smoke", 0, row_timeout_s(LadderLevel.L1), False, LadderLevel.L1
        )
        for s in smoke_specs
    ]
    for a, b in zip(served, direct):
        assert (a["name"], a["verdict"], a["terminal"]) == (
            b["name"],
            b["verdict"],
            b["terminal"],
        )
        assert a["ladder_level"] == "L1"
        assert (a.get("enum") or {}).get("status") == (b.get("enum") or {}).get(
            "status"
        )
    # the worker restored nothing it did not have to: its log says what leaked
    log = (tmp_path / "worker.log").read_text()
    assert "Traceback" not in log


def test_crashed_worker_is_a_crash_row_and_is_respawned(tmp_path, smoke_specs):
    worker = _Worker(tmp_path / "worker.log")
    try:
        spec = smoke_specs[0]
        row = _run_one_reused(
            worker, spec, "golden_smoke", 0, 60, False, LadderLevel.L0, probe="crash"
        )
        assert row["terminal"] == "crash" and row["verdict"] == "error"
        assert not worker.alive()
        # the next row transparently gets a fresh worker
        row2 = _run_one_reused(
            worker, spec, "golden_smoke", 0, 60, False, LadderLevel.L0
        )
        assert row2["terminal"] not in ("crash", "timeout")
        assert worker.alive() and worker.rows == 1
    finally:
        worker.stop()


def test_silent_worker_is_killed_at_the_budget(tmp_path, smoke_specs):
    worker = _Worker(tmp_path / "worker.log")
    try:
        t0 = time.perf_counter()
        row = _run_one_reused(
            worker,
            smoke_specs[0],
            "golden_smoke",
            0,
            3,
            False,
            LadderLevel.L0,
            probe="hang",
        )
        assert row["terminal"] == "timeout" and row["harness_error"] == "exceeded 3s"
        assert 3.0 <= time.perf_counter() - t0 < 30.0
        assert not worker.alive()
    finally:
        worker.stop()


def test_recycling_and_header_stamp(tmp_path, smoke_specs, monkeypatch):
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    out = runner_mod.run_corpus(
        "golden_smoke",
        None,
        0,
        None,
        ladder_level=LadderLevel.L1,
        out_suffix="_reuse_test",
        reuse_workers=True,
        worker_rows=3,
    )
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    header, rows = lines[0], lines[1:]
    assert header["worker_reuse"] == {
        "rows_per_worker": 3,
        "rss_limit_mb": runner_mod.WORKER_RSS_MB,
    }
    assert header["ladder_level"] == "L1" and header["row_timeout_s"] == 200
    assert len(rows) == len(smoke_specs)
    assert all(r["terminal"] not in ("crash", "timeout") for r in rows)
    # 7 rows at 3 per worker: three workers were started (the log records each)
    log = (tmp_path / "golden_smoke_L1_reuse_test_worker.log").read_text()
    assert "Traceback" not in log
    assert results_header("golden_smoke", 0, {})["worker_reuse"] is False
