import csv
import json

import pytest

from triton_viz.tools.nki_cost_model import ComputeCalibration
from triton_viz.tools.nki_fit_structured_controls import (
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


def test_source_only_export_no_longer_emits_completion_rows(tmp_path):
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

    # Per-structure NC completion is no longer a calibration product: a single
    # global completion term replaced every structural completion table.
    assert not [row for row in rows if row.get("engine") == "completion"]
    assert all("nc_completion_ns" not in row for row in rows)
