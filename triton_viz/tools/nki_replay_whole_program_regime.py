"""Frozen whole-program engine replay using target source and aggregate labels."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_evaluate_whole_program_regime import (
    _source_sample,
    source_descriptor,
)


def _wape(rows: list[dict], predicted: str, actual: str) -> float:
    denominator = sum(float(row[actual]) for row in rows)
    return sum(abs(float(row[predicted]) - float(row[actual])) for row in rows) / denominator * 100.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-evaluation-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with (args.control_root / "operator_results.csv").open(encoding="utf-8", newline="") as file:
        controls = [_source_sample(args.control_root, row) for row in csv.DictReader(file) if row.get("status") == "ok"]
    output = []
    with args.target_evaluation_csv.open(encoding="utf-8", newline="") as file:
        target_rows = list(csv.DictReader(file))
    for row in target_rows:
        descriptor = source_descriptor(args.target_root, row["case"], row["dtype"])
        candidates = [item for item in controls if item["key"] == descriptor["key"]]
        if not candidates:
            output.append({**row, "whole_program_regime_status": "ood"})
            continue
        distance = min(abs(item["distance_feature"] - descriptor["distance_feature"]) for item in candidates)
        nearest = [item for item in candidates if abs(item["distance_feature"] - descriptor["distance_feature"]) == distance]
        predicted = {
            engine: statistics.mean(item["actual"][engine] for item in nearest)
            for engine in ("vector", "scalar", "gpsimd")
        }
        output.append(
            {
                **row,
                "whole_program_regime_status": "covered",
                **{f"regime_predicted_{engine}_us": value for engine, value in predicted.items()},
            }
        )
    covered = [row for row in output if row["whole_program_regime_status"] == "covered"]
    metrics = {
        "cases": len(output),
        "covered_cases": len(covered),
        "coverage_pct": len(covered) / len(output) * 100.0,
        "vector_wape_pct": _wape(covered, "regime_predicted_vector_us", "hardware_vector_active_us"),
        "scalar_wape_pct": _wape(covered, "regime_predicted_scalar_us", "hardware_scalar_active_us"),
        "gpsimd_wape_pct": _wape(covered, "regime_predicted_gpsimd_us", "hardware_gpsimd_active_us"),
        "nc_p50_mape_pct": statistics.mean(abs(float(row["nc_error_pct"])) for row in output),
        "target_postcompile_prediction_reads": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        fields = list(output[0])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (args.output.with_suffix(".metrics.json")).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
