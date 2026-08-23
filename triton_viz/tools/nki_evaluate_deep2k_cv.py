"""Audit fixed-2K long arithmetic controls with nonnegative source-op fits."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from triton_viz.tools.nki_fit_source_sequence_lowering import _cases
from triton_viz.tools.nki_fit_structured_controls import load_runtime_engine_baselines


FEATURES = ("intercept", "op_add", "op_multiply")


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([
        [1.0, float(row["features"].get("op_add", 0.0)),
         float(row["features"].get("op_multiply", 0.0))]
        for row in rows
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = _cases(
        [args.root], load_runtime_engine_baselines(args.runtime_overhead_results)
    )
    folds = []
    for engine in ("vector", "scalar", "gpsimd"):
        selected = [row for row in rows if row["engine"] == engine]
        for field in ("chain", "logical_free_dim", "family"):
            for held in sorted({row[field] for row in selected}, key=str):
                train = [row for row in selected if row[field] != held]
                test = [row for row in selected if row[field] == held]
                coefficients = nnls(
                    _matrix(train), np.asarray([row["payload_ns"] for row in train])
                )[0]
                errors = [
                    abs(prediction + row["baseline_ns"] - row["active_ns"])
                    / row["active_ns"] * 100.0
                    for prediction, row in zip(_matrix(test) @ coefficients, test)
                ]
                folds.append({
                    "engine": engine, "held_field": field, "held_value": held,
                    "samples": len(test), "mape_pct": statistics.mean(errors),
                    "coefficients": list(map(float, coefficients)),
                })
    report = {
        "schema": "triton-viz.deep2k-control-cv-v1",
        "protocols": ["leave-chain-out", "leave-logical-free-dim-out", "leave-family-out"],
        "feature_order": FEATURES,
        "folds": folds,
        "summary": {
            engine: {
                "mean_mape_pct": statistics.mean(
                    fold["mape_pct"] for fold in folds if fold["engine"] == engine
                ),
                "max_fold_mape_pct": max(
                    fold["mape_pct"] for fold in folds if fold["engine"] == engine
                ),
            }
            for engine in ("vector", "scalar", "gpsimd")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
