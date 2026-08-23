"""Fit two-phase factorial DAG controls with whole-cell cross-phase holdout."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from triton_viz.tools.nki_fit_randommixed2k_vector import _features, _matrix
from triton_viz.tools.nki_fit_source_sequence_lowering import PAYLOAD_RESOLUTION_NS, _cases
from triton_viz.tools.nki_fit_structured_controls import load_runtime_engine_baselines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--engines", nargs="+", choices=("vector", "scalar"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    args = parser.parse_args(argv)

    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    rows = [
        row for row in _cases(args.roots, baselines)
        if row["engine"] in args.engines
        and (3000 <= row["chain"] <= 3053 or 4000 <= row["chain"] <= 4053)
    ]
    expected = 108 * len(args.engines)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} two-phase rows, found {len(rows)}")
    features = _features(rows)
    folds = []
    for engine in args.engines:
        engine_rows = [row for row in rows if row["engine"] == engine]
        for cell in range(9):
            in_cell = lambda row: ((row["chain"] % 1000) // 6) == cell
            train = [row for row in engine_rows if not in_cell(row)]
            test = [row for row in engine_rows if in_cell(row)]
            coefficients = nnls(
                _matrix(train, features),
                np.asarray([row["payload_ns"] for row in train]),
            )[0]
            errors, violations = [], 0
            for prediction, row in zip(_matrix(test, features) @ coefficients, test):
                if row["payload_ns"] > PAYLOAD_RESOLUTION_NS:
                    errors.append(abs(prediction - row["payload_ns"])
                                  / row["payload_ns"] * 100.0)
                elif prediction > PAYLOAD_RESOLUTION_NS:
                    violations += 1
            folds.append({
                "engine": engine, "cell": cell, "samples": len(test),
                "mape_pct": statistics.mean(errors) if errors else 0.0,
                "max_case_mape_pct": max(errors, default=0.0),
                "zero_payload_violations": violations,
            })
    by_engine = {
        engine: {
            "mean_mape_pct": statistics.mean(
                fold["mape_pct"] for fold in folds if fold["engine"] == engine
            ),
            "max_fold_mape_pct": max(
                fold["mape_pct"] for fold in folds if fold["engine"] == engine
            ),
            "zero_payload_violations": sum(
                fold["zero_payload_violations"] for fold in folds
                if fold["engine"] == engine
            ),
        }
        for engine in args.engines
    }
    report = {
        "schema": "triton-viz.factorial-dag-two-phase-cv-v1",
        "protocol": "hold out whole depth/gap cell across both order phases",
        "feature_names": features, "folds": folds, "by_engine": by_engine,
    }
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    output_rows = []
    for engine in args.engines:
        engine_rows = [row for row in rows if row["engine"] == engine]
        coefficients = nnls(
            _matrix(engine_rows, features),
            np.asarray([row["payload_ns"] for row in engine_rows]),
        )[0]
        output_rows.extend({
            "engine": engine, "dtype": "float32", "target": "fixed_ns",
            "feature": feature, "coefficient": float(value),
        } for feature, value in zip(features, coefficients))
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "engine", "dtype", "target", "feature", "coefficient"
        ))
        writer.writeheader(); writer.writerows(output_rows)
    print(json.dumps(by_engine, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
