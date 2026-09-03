"""Freeze the attention TensorE busy surface from independent pipeline controls.

This fitter no longer emits an NC completion column: per-structure
completion floors were removed from the cost model in favour of a single
global completion term, so only the TensorE occupancy surface survives.

Each control width is compiled several times and the **median** TensorE active
time is taken.  Compilation is not deterministic: for an identical instruction
stream (43 TensorE instructions in every case) about 21% of compilations land
in an allocation that runs ~2.4x slower end to end -- measured as 5 slow
results in 24 recompilations of one suite, with the same width flipping between
modes across trials.  A single compilation is therefore not a reliable estimate
of a width's cost, and with only 8 widths per suite and two CV folds this
calibration has no averaging to absorb such an outlier: the probability that
all 16 control points land in the fast mode is 0.79**16 = 2.3%.  The median
over trials estimates the typical compilation, which is the quantity the cost
model predicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _load(path: Path) -> list[tuple[int, float]]:
    """Median TensorE active time per control width, across compilations."""
    trials: dict[int, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok" or row.get("spec.kind") != "tensor_attention_pipeline":
                continue
            trials.setdefault(int(row["spec.dv"]), []).append(
                float(row["profile.tensor_engine_active_time"]) * 1e9
            )
    return sorted(
        (width, statistics.median(values)) for width, values in trials.items()
    )


def trial_spread(path: Path) -> dict[int, dict[str, float]]:
    """Per-width compilation spread, so bimodality stays visible in the report."""
    trials: dict[int, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok" or row.get("spec.kind") != "tensor_attention_pipeline":
                continue
            trials.setdefault(int(row["spec.dv"]), []).append(
                float(row["profile.tensor_engine_active_time"]) * 1e9
            )
    return {
        width: {
            "trials": len(values),
            "median_ns": statistics.median(values),
            "min_ns": min(values),
            "max_ns": max(values),
            "spread_ratio": (max(values) / min(values)) if min(values) else 0.0,
        }
        for width, values in sorted(trials.items())
    }


def _predict(rows: list[tuple[int, float]], width: int, column: int) -> float:
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
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in attention pipeline fit")

    suites = {path.name: _load(path) for path in args.inputs}
    if any(not rows for rows in suites.values()) or len(suites) < 2:
        raise ValueError("Strict CV requires at least two nonempty independent suites")
    folds = []
    for held_name, test in suites.items():
        train = sorted(row for name, rows in suites.items() if name != held_name for row in rows)
        tensor_pairs = [(_predict(train, width, 1), tensor) for width, tensor in test]
        folds.append(
            {
                "held_suite": held_name,
                "samples": len(test),
                "tensor_wape_pct": 100 * sum(abs(p - y) for p, y in tensor_pairs) / sum(y for _, y in tensor_pairs),
            }
        )
    mean_tensor = sum(row["tensor_wape_pct"] for row in folds) / len(folds)
    passed = mean_tensor < args.max_tensor_wape
    report = {
        "schema": "triton-viz.attention-pipeline-control-cv-v1",
        "protocol": "leave-one-independent-width-grid-out linear interpolation; "
        "median TensorE active time over independent compilations per width",
        "engine_metric": "TensorE WAPE",
        "folds": folds,
        "mean_tensor_wape_pct": mean_tensor,
        "tensor_gate_pct": args.max_tensor_wape,
        "passed": passed,
        # Kept in the report so the bimodal compilation behaviour stays visible
        # instead of being hidden by the median.
        "compilation_spread": {
            path.name: trial_spread(path) for path in args.inputs
        },
        "target_postcompile_prediction_reads": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("dtype", "value_width", "tensor_active_ns")
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for width, tensor in sorted(
            row for rows in suites.values() for row in rows
        ):
            writer.writerow(
                {
                    "dtype": "float32",
                    "value_width": width,
                    "tensor_active_ns": tensor,
                }
            )
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
