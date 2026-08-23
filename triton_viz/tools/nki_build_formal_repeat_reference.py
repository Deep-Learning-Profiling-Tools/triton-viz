"""Build a median-of-three formal aggregate-engine evaluation reference."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_fit_source_sequence_lowering import PAYLOAD_RESOLUTION_NS
from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines, runtime_engine_baseline_ns,
)


_AGGREGATE_KEYS = {
    "vector": "vector_engine_active_time",
    "scalar": "scalar_engine_active_time",
    "gpsimd": "gpsimd_engine_active_time",
    "tensor": "tensor_engine_active_time",
}


def _aggregate_active_us(case: Path) -> dict[str, float]:
    # Evaluation-label phase only.  Read a strict aggregate-key allowlist;
    # never inspect instruction tables/counts/opcodes or mapping artifacts.
    path = case / "hardware" / "explorer_summary.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if len(document) != 1:
        raise ValueError(f"expected one aggregate summary record in {path}")
    summary = next(iter(document.values()))
    return {
        engine: float(summary[key]) * 1e6
        for engine, key in _AGGREGATE_KEYS.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--prediction-csv", action="append", required=True, type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    prediction_rows = []
    for path in args.prediction_csv:
        with path.open(encoding="utf-8", newline="") as file:
            prediction_rows.extend(csv.DictReader(file))
    predictions = {row["case"]: row for row in prediction_rows}

    cases, errors = [], {engine: [] for engine in ("vector", "scalar", "gpsimd")}
    for suffix in ("0", "1"):
        original = args.experiment_root / "holdouts" / f"formal_fp32_v1_{suffix}"
        repeat_roots = [
            args.experiment_root / "target_label_repeats" / f"formal_fp32_rep{rep}_{suffix}"
            for rep in (1, 2)
        ]
        for original_case in sorted(path for path in original.iterdir() if path.is_dir()):
            name = original_case.name
            if name not in predictions:
                raise ValueError(f"missing frozen prediction for {name}")
            measurements = [
                _aggregate_active_us(root / name)
                for root in [original, *repeat_roots]
            ]
            record = {"case": name, "engines": {}}
            for engine in _AGGREGATE_KEYS:
                values = [item[engine] for item in measurements]
                median_active = statistics.median(values)
                baseline_us = runtime_engine_baseline_ns(
                    baselines, "float32", 128, engine
                ) / 1000.0
                payload = max(0.0, median_active - baseline_us)
                item = {
                    "active_us": values, "median_active_us": median_active,
                    "runtime_baseline_us": baseline_us,
                    "median_payload_us": payload,
                    "active_range_us": max(values) - min(values),
                }
                if engine in errors and payload * 1000.0 > PAYLOAD_RESOLUTION_NS:
                    predicted = float(predictions[name][f"predicted_{engine}_payload_us"])
                    ape = abs(predicted - payload) / payload * 100.0
                    item.update({"predicted_payload_us": predicted, "ape_pct": ape})
                    errors[engine].append(ape)
                record["engines"][engine] = item
            cases.append(record)
    by_engine = {
        engine: {
            "evaluable_cases": len(values),
            "mean_mape_pct": statistics.mean(values) if values else None,
            "max_case_mape_pct": max(values, default=None),
        }
        for engine, values in errors.items()
    }
    report = {
        "schema": "triton-viz.formal-aggregate-median-reference-v1",
        "protocol": "original plus two full-set repeats; per-case median; aggregate-key allowlist",
        "aggregate_key_allowlist": _AGGREGATE_KEYS,
        "by_engine": by_engine, "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_engine, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
