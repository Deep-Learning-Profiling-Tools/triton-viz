"""Freeze attention TensorE/NC surfaces from independent pipeline controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> list[tuple[int, float, float]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok" or row.get("spec.kind") != "tensor_attention_pipeline":
                continue
            rows.append(
                (
                    int(row["spec.dv"]),
                    float(row["profile.tensor_engine_active_time"]) * 1e9,
                    float(row["latency.nc_latency.p50_us"]) * 1e3,
                )
            )
    return sorted(rows)


def _predict(rows: list[tuple[int, float, float]], width: int, column: int) -> float:
    if len(rows) < 2:
        raise ValueError("An independent attention control suite needs >=2 points")
    upper_index = next((i for i, row in enumerate(rows) if row[0] >= width), len(rows))
    if upper_index == 0:
        lower, upper = rows[0], rows[1]
    elif upper_index == len(rows):
        lower, upper = rows[-2], rows[-1]
    else:
        lower, upper = rows[upper_index - 1], rows[upper_index]
    weight = (width - lower[0]) / (upper[0] - lower[0])
    return lower[column] + weight * (upper[column] - lower[column])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    parser.add_argument("--max-tensor-wape", type=float, default=20.0)
    parser.add_argument("--max-nc-mape", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in attention pipeline fit")

    suites = {path.name: _load(path) for path in args.inputs}
    if any(not rows for rows in suites.values()) or len(suites) < 2:
        raise ValueError("Strict CV requires at least two nonempty independent suites")
    folds = []
    for held_name, test in suites.items():
        train = sorted(row for name, rows in suites.items() if name != held_name for row in rows)
        tensor_pairs = [(_predict(train, width, 1), tensor) for width, tensor, _ in test]
        nc_pairs = [(_predict(train, width, 2), nc) for width, _, nc in test]
        folds.append(
            {
                "held_suite": held_name,
                "samples": len(test),
                "tensor_wape_pct": 100 * sum(abs(p - y) for p, y in tensor_pairs) / sum(y for _, y in tensor_pairs),
                "nc_mape_pct": 100 * sum(abs(p - y) / y for p, y in nc_pairs) / len(nc_pairs),
            }
        )
    mean_tensor = sum(row["tensor_wape_pct"] for row in folds) / len(folds)
    mean_nc = sum(row["nc_mape_pct"] for row in folds) / len(folds)
    passed = mean_tensor < args.max_tensor_wape and mean_nc < args.max_nc_mape
    report = {
        "schema": "triton-viz.attention-pipeline-control-cv-v1",
        "protocol": "leave-one-independent-width-grid-out linear interpolation",
        "engine_metric": "TensorE WAPE",
        "headline_metric": "NC-p50 MAPE",
        "folds": folds,
        "mean_tensor_wape_pct": mean_tensor,
        "mean_nc_mape_pct": mean_nc,
        "tensor_gate_pct": args.max_tensor_wape,
        "nc_gate_pct": args.max_nc_mape,
        "passed": passed,
        "target_postcompile_prediction_reads": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("dtype", "value_width", "tensor_active_ns", "nc_completion_ns")
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for width, tensor, nc in sorted(
            row for rows in suites.values() for row in rows
        ):
            writer.writerow(
                {
                    "dtype": "float32",
                    "value_width": width,
                    "tensor_active_ns": tensor,
                    "nc_completion_ns": nc,
                }
            )
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
