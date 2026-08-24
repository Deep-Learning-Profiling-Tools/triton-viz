import csv
import json

import pytest

from triton_viz.tools.nki_cost_model import ComputeCalibration
from triton_viz.tools.nki_fit_structured_controls import (
    _load_completion_by_case,
    collect_source_only,
    main,
)


def test_structured_fit_rejects_target_before_opening_artifacts(tmp_path):
    with pytest.raises(SystemExit, match="Refusing target post-compile"):
        main(
            [
                str(tmp_path / "target"),
                "--compute-calibration-csv",
                str(tmp_path / "missing.csv"),
                "--artifact-role",
                "target",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )


def test_load_completion_accepts_operator_results_schema(tmp_path):
    path = tmp_path / "operator_results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("op", "rows", "cols", "dtype", "hardware_nc_p50_us"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "op": "softmax",
                "rows": "128",
                "cols": "3584",
                "dtype": "float32",
                "hardware_nc_p50_us": "17.5",
            }
        )

    assert _load_completion_by_case([tmp_path]) == {
        "softmax__r128__c3584__float32": 17_500.0
    }


def test_source_only_completion_uses_declared_semantics_without_profile(tmp_path):
    case_name = "control_reduce__p16__f512__n1__bfloat16"
    case = tmp_path / case_name
    case.mkdir()

    declared_events = [
        {
            "seq": 0,
            "op": "load",
            "src_dtype": "bfloat16",
            "active_lanes": 16 * 512,
            "partition_count": 16,
            "offsets_shape": [16, 512],
            "mem_src": "HBM",
            "mem_dst": "SBUF",
        },
        {
            "seq": 1,
            "op": "reduce_sum",
            "api_op": "reduce_sum",
            "input_ptrs": [10],
            "output_ptr": 11,
            "input_shape": [16, 512],
            "output_shape": [16, 1],
            "input_dtypes": ["bfloat16"],
            "output_dtype": "float32",
        },
    ]
    runtime_events = [
        declared_events[0],
        {
            **declared_events[1],
            "input_dtypes": ["float32"],
            "output_dtype": "float32",
        },
    ]
    for name, events in (
        ("trace.jsonl", declared_events),
        ("dependency_trace.jsonl", runtime_events),
    ):
        (case / name).write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    with (tmp_path / "control_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=("case", "hardware_nc_p50_us")
        )
        writer.writeheader()
        writer.writerow(
            {"case": case_name, "hardware_nc_p50_us": "52.0"}
        )

    rows = collect_source_only([tmp_path], ComputeCalibration({}))

    assert len(rows) == 1
    assert rows[0]["engine"] == "completion"
    assert rows[0]["dtype"] == "bfloat16"
    assert rows[0]["nc_completion_ns"] == 52_000.0
    assert rows[0]["mapping_status"] == "accepted_source_only_completion"
