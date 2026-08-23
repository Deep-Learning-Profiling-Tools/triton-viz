"""Leave-one-size-out CV for source-only structured control calibration."""

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
    collect_source_only,
    load_runtime_engine_baselines,
    runtime_engine_baseline_ns,
)


def _column(case: str) -> int | None:
    match = re.search(r"__c(\d+)__", case)
    return int(match.group(1)) if match else None


def _mape(errors: list[float]) -> float:
    return statistics.mean(errors) if errors else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-control-root", nargs="+", type=Path, required=True)
    parser.add_argument("--sequence-control-root", nargs="+", type=Path, required=True)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--runtime-overhead-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    level_b = ComputeCalibration.from_csv(args.compute_calibration_csv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    base_rows = collect_source_only(args.base_control_root, level_b, runtime_baselines=baselines)
    sequence_rows = collect_source_only(
        args.sequence_control_root, level_b, runtime_baselines=baselines
    )
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
        errors = {"vector": [], "scalar": []}
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
                for engine in errors:
                    actual = float(row.get(f"hardware_{engine}_active_us") or 0.0)
                    if actual <= 0:
                        continue
                    predicted_ns = simulation.engine_busy_ns.get(engine, 0.0)
                    predicted_ns += runtime_engine_baseline_ns(
                        baselines,
                        str(row["dtype"]),
                        int(row["rows"]),
                        engine,
                    )
                    predicted = predicted_ns / 1000.0
                    errors[engine].append(abs(predicted - actual) / actual * 100.0)
                cases += 1
        results.append(
            {
                "held_column": held_column,
                "cases": cases,
                "vector_mape_pct": _mape(errors["vector"]),
                "scalar_mape_pct": _mape(errors["scalar"]),
                "vector_samples": len(errors["vector"]),
                "scalar_samples": len(errors["scalar"]),
            }
        )
    report = {
        "schema": "triton-viz.source-control-cv-v1",
        "split": "leave-one-control-column-out",
        "folds": results,
        "vector_mape_pct": _mape([row["vector_mape_pct"] for row in results]),
        "scalar_mape_pct": _mape([row["scalar_mape_pct"] for row in results]),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
