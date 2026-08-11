import csv
import json

import pytest

from triton_viz.tools.nki_workload_cases import load_cases, write_csv

pytestmark = pytest.mark.nki


def test_load_cases_expands_deduplicates_and_sorts(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [
        {"kind": "tensor_add", "matrix": {"p": [8, 1], "f": [128, 32]}},
        {"kind": "dma_transpose", "p": 4, "f": 64},
        {"kind": "sbuf_transpose", "p": 8, "x": 4, "y": 8},
    ]}))
    cases = load_cases(path)
    ids = [case["id"] for case in cases]
    assert ids == sorted(ids)
    assert len(ids) == 6
    assert {case["kind"] for case in cases} == {
        "tensor_add", "dma_transpose", "sbuf_transpose"
    }


def test_load_cases_rejects_empty_invalid_and_duplicate_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": []}))
    with pytest.raises(ValueError, match="non-empty"):
        load_cases(path)

    path.write_text(json.dumps({"cases": [{"kind": "unknown", "p": 1, "f": 1}]}))
    with pytest.raises(ValueError, match="unsupported kind"):
        load_cases(path)

    path.write_text(json.dumps({"cases": [{"kind": "tensor_add", "p": 129, "f": 1}]}))
    with pytest.raises(ValueError, match="must not exceed 128"):
        load_cases(path)

    path.write_text(json.dumps({"cases": [
        {"kind": "tensor_add", "p": 1, "f": 32},
        {"kind": "tensor_add", "p": 1, "f": 32},
    ]}))
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)

    path.write_text(json.dumps({"cases": [
        {"kind": "sbuf_transpose", "p": 1, "x": 32, "y": 32},
    ]}))
    with pytest.raises(ValueError, match="must not exceed 512"):
        load_cases(path)


def test_write_csv_preserves_error_rows_and_column_order(tmp_path):
    path = tmp_path / "results.csv"
    rows = [
        {"case_id": "a", "kind": "tensor_add", "status": "error", "error": "boom"},
        {"case_id": "b", "kind": "dma_transpose", "status": "ok", "p": 8},
    ]
    write_csv(rows, path)
    with path.open(newline="") as file:
        exported = list(csv.DictReader(file))
    assert [row["case_id"] for row in exported] == ["a", "b"]
    assert exported[0]["status"] == "error"
    assert exported[0]["error"] == "boom"
    assert exported[1]["p"] == "8"
