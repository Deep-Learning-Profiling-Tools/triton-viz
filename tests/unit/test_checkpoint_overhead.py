"""Measurement bookkeeping checks; these execute no detector kernels."""

import json
import sqlite3

import pytest

from evaluation.checkpoint_overhead import Samples, micro, paired_summary


def test_paired_summary_uses_balanced_adjacent_pairs_and_keeps_semantic_changes():
    records = []
    for block, (mode, wall) in enumerate(zip("ABBA", (10, 12, 14, 11))):
        row = {
            "wall_s": wall,
            "verdict": "race-free",
            "terminal": "proved@T1",
            "static": {"reason": "baseline"},
        }
        if block == 2:
            row["static"]["reason"] = "changed"
        records.append(
            {
                "group": "representative",
                "corpus": "example",
                "name": "same",
                "block": block,
                "mode": mode,
                "ledger_external_s": 0.01 if mode == "B" else 0,
                "row": row,
            }
        )
    summary = paired_summary(records)
    measured = summary["rows"][0]
    assert measured["A_wall_s"] == [10, 11]
    assert measured["B_wall_s"] == [12, 14]
    assert measured["B_minus_A_mean_s"] == 2.5
    assert measured["B_over_A_mean"] == pytest.approx(13 / 10.5)
    assert measured["adjacent_pairs"][0]["B_minus_A_s"] == 2
    assert measured["adjacent_pairs"][1]["B_minus_A_s"] == 3
    assert not measured["semantic_outcomes_identical"]
    assert summary["direct_checkpoint_fraction_of_B_wall"] == pytest.approx(0.02 / 26)


def test_micro_persists_every_input_in_every_round_without_detector_execution(tmp_path):
    pytest.importorskip("evaluation.pinned_state")
    source = tmp_path / "source.jsonl"
    header = {"header": True, "ladder_level": "L2", "row_timeout_s": 200}
    rows = [
        {
            "corpus": "example",
            "name": f"row-{index}",
            "ladder_level": "L2",
            "fence_order": True,
            "wall_s": 1.25,
            "verdict": "race-free",
            "terminal": "proved@T1",
        }
        for index in range(2)
    ]
    source.write_text("\n".join(json.dumps(row) for row in [header, *rows]) + "\n")
    output = tmp_path / "measurement"
    output.mkdir()
    samples = Samples(output / "samples.jsonl")
    try:
        summary = micro(output, [source], 3, samples)
    finally:
        samples.close()
    saved = [
        json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()
    ]
    assert len(saved) == 6
    assert len({sample["row"]["name"] for sample in saved}) == 6
    assert summary["source_row_bytes"]["n"] == 2
    assert summary["result_transaction_s"]["n"] == 6
    for round_index in range(3):
        with sqlite3.connect(
            output / f"micro-{round_index}" / "checkpoint.sqlite"
        ) as con:
            accepted = con.execute("SELECT row_json FROM results").fetchall()
        assert len(accepted) == 2
        assert {json.loads(row[0])["wall_s"] for row in accepted} == {1.25}
