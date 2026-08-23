"""Evaluate frozen source-only payload coefficients on an untouched control audit."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_fit_source_sequence_lowering import PAYLOAD_RESOLUTION_NS, _cases
from triton_viz.tools.nki_fit_structured_controls import load_runtime_engine_baselines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--seed-min", required=True, type=int)
    parser.add_argument("--seed-max", required=True, type=int)
    parser.add_argument("--engines", nargs="+", choices=("vector", "scalar", "gpsimd"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    with args.candidate.open(encoding="utf-8", newline="") as file:
        candidate_rows = list(csv.DictReader(file))
    coefficients = {
        engine: {
            row["feature"]: float(row["coefficient"])
            for row in candidate_rows
            if row["engine"] == engine and row["target"] == "fixed_ns"
        }
        for engine in args.engines
    }
    if any(not coefficients[engine] for engine in args.engines):
        raise ValueError("candidate is missing fixed payload coefficients")

    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    rows = [
        row for row in _cases(args.roots, baselines)
        if row["engine"] in args.engines
        and args.seed_min <= row["chain"] <= args.seed_max
    ]
    expected = (args.seed_max - args.seed_min + 1) * len(args.engines)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} untouched audit rows, found {len(rows)}")

    cases, by_engine = [], {}
    for engine in args.engines:
        errors, violations = [], 0
        for row in (item for item in rows if item["engine"] == engine):
            prediction = max(0.0, sum(
                value * float(row["features"].get(name, 0.0))
                for name, value in coefficients[engine].items()
            ))
            actual = float(row["payload_ns"])
            error = None
            if actual > PAYLOAD_RESOLUTION_NS:
                error = abs(prediction - actual) / actual * 100.0
                errors.append(error)
            elif prediction > PAYLOAD_RESOLUTION_NS:
                violations += 1
            cases.append({
                "engine": engine, "seed": row["chain"],
                "actual_payload_ns": actual, "predicted_payload_ns": prediction,
                "ape_pct": error,
            })
        by_engine[engine] = {
            "samples": sum(item["engine"] == engine for item in rows),
            "mean_mape_pct": statistics.mean(errors) if errors else 0.0,
            "max_case_mape_pct": max(errors, default=0.0),
            "zero_payload_violations": violations,
        }
    report = {
        "schema": "triton-viz.frozen-payload-control-audit-v1",
        "protocol": "predict untouched controls with frozen coefficients; no refit",
        "candidate": str(args.candidate.resolve()),
        "seed_range": [args.seed_min, args.seed_max],
        "payload_resolution_ns": PAYLOAD_RESOLUTION_NS,
        "by_engine": by_engine, "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_engine, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
