"""Leave-one-size-out CV for mapped control structured calibration."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from triton_viz.tools.nki_cost_model import (
    ComputeCalibration,
    CostModel,
    StructuredControlCalibration,
    simulate_jsonl,
)
from triton_viz.tools.nki_fit_structured_controls import (
    FIELDS,
    aggregate_rows,
    collect,
)


def _column(case: str) -> int | None:
    match = re.search(r"__c(\d+)__", case)
    return int(match.group(1)) if match else None


def _wape(pairs: list[tuple[float, float]]) -> float:
    denominator = sum(actual for _, actual in pairs)
    return (
        sum(abs(predicted - actual) for predicted, actual in pairs)
        / denominator
        * 100.0
        if pairs and denominator > 0
        else float("nan")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-control-root", nargs="+", type=Path, required=True)
    parser.add_argument("--sequence-control-root", nargs="+", type=Path, required=True)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--runtime-overhead-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-role", required=True, choices=("control", "target")
    )
    args = parser.parse_args(argv)

    if args.artifact_role != "control":
        raise SystemExit("Refusing target post-compile artifacts in control CV")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    level_b = ComputeCalibration.from_csv(args.compute_calibration_csv)
    base_rows = collect(args.base_control_root, level_b)
    sequence_rows = collect(args.sequence_control_root, level_b)
    folds = sorted({_column(str(row["case"])) for row in sequence_rows} - {None})
    results = []
    for held_column in folds:
        train = base_rows + [
            row for row in sequence_rows if _column(str(row["case"])) != held_column
        ]
        calibration_rows = aggregate_rows(train)
        calibration_path = args.output_dir / f"structured_without_c{held_column}.csv"
        with calibration_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(calibration_rows)
        model = CostModel(
            compute_calibration=level_b,
            structured_control_lowering=StructuredControlCalibration.from_csv(
                calibration_path
            ),
            strict_calibration=True,
            enable_structured_completion_floor=False,
        )
        pairs = {"vector": [], "scalar": []}
        cases = 0
        for root in args.sequence_control_root:
            result_path = root / "operator_results.csv"
            if not result_path.is_file():
                continue
            with result_path.open(encoding="utf-8", newline="") as file:
                source_rows = list(csv.DictReader(file))
            for row in source_rows:
                if row.get("status") != "ok" or int(row["cols"]) != held_column:
                    continue
                case = f"{row['op']}__r{row['rows']}__c{row['cols']}__{row['dtype']}"
                simulation = simulate_jsonl(root / case / "trace.jsonl", model)
                for engine in pairs:
                    actual = float(row.get(f"hardware_{engine}_active_us") or 0.0)
                    predicted = simulation.engine_busy_ns.get(engine, 0.0) / 1000.0
                    pairs[engine].append((predicted, actual))
                cases += 1
        results.append(
            {
                "held_column": held_column,
                "cases": cases,
                "vector_wape_pct": _wape(pairs["vector"]),
                "scalar_wape_pct": _wape(pairs["scalar"]),
                "vector_samples": len(pairs["vector"]),
                "scalar_samples": len(pairs["scalar"]),
            }
        )
    report = {
        "schema": "triton-viz.mapped-control-cv-v2",
        "split": "leave-one-control-column-out",
        "folds": results,
        "vector_mean_fold_wape_pct": statistics.mean(
            row["vector_wape_pct"] for row in results
        ),
        "scalar_mean_fold_wape_pct": statistics.mean(
            row["scalar_wape_pct"] for row in results
        ),
        "vector_worst_fold_wape_pct": max(
            row["vector_wape_pct"] for row in results
        ),
        "scalar_worst_fold_wape_pct": max(
            row["scalar_wape_pct"] for row in results
        ),
        "target_postcompile_prediction_reads": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
