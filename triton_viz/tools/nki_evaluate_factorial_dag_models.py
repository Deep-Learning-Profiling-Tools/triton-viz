"""Control-only linear model search for two-phase factorial DAG payloads."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triton_viz.tools.nki_evaluate_randommixed2k_models import _feature_sets, _matrix
from triton_viz.tools.nki_fit_source_sequence_lowering import PAYLOAD_RESOLUTION_NS, _cases
from triton_viz.tools.nki_fit_structured_controls import load_runtime_engine_baselines


def _predict(kind, train_x, train_y, test_x):
    if kind == "nnls":
        return test_x @ nnls(train_x, train_y)[0]
    alpha = float(kind.split("_", 1)[1])
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y)
    return np.maximum(0.0, model.predict(test_x))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    rows = [row for row in _cases(args.roots, baselines)
            if 3000 <= row["chain"] <= 3053 or 4000 <= row["chain"] <= 4053]
    if len(rows) != 324:
        raise ValueError(f"expected 324 two-phase engine rows, found {len(rows)}")
    feature_sets = _feature_sets(rows)
    reports = []
    for engine in ("vector", "scalar", "gpsimd"):
        engine_rows = [row for row in rows if row["engine"] == engine]
        for feature_mode, names in feature_sets.items():
            # The 50--1000 range is an explicitly adaptive, control-only
            # refinement after ridge_100 missed the Scalar max-fold gate by
            # 0.22 percentage points.  It is never selected on target labels.
            for kind in (
                "nnls", "ridge_0.1", "ridge_1", "ridge_10", "ridge_50",
                "ridge_100", "ridge_200", "ridge_500", "ridge_1000",
            ):
                folds = []
                for cell in range(9):
                    in_cell = lambda row: ((row["chain"] % 1000) // 6) == cell
                    train = [row for row in engine_rows if not in_cell(row)]
                    test = [row for row in engine_rows if in_cell(row)]
                    predictions = _predict(
                        kind, _matrix(train, names),
                        np.asarray([row["payload_ns"] for row in train]),
                        _matrix(test, names),
                    )
                    errors, violations = [], 0
                    for prediction, row in zip(predictions, test):
                        if row["payload_ns"] > PAYLOAD_RESOLUTION_NS:
                            errors.append(abs(prediction - row["payload_ns"])
                                          / row["payload_ns"] * 100.0)
                        elif prediction > PAYLOAD_RESOLUTION_NS:
                            violations += 1
                    folds.append({
                        "cell": cell, "samples": len(test),
                        "mape_pct": statistics.mean(errors) if errors else 0.0,
                        "zero_payload_violations": violations,
                    })
                reports.append({
                    "engine": engine, "features": feature_mode, "model": kind,
                    "feature_count": len(names),
                    "mean_mape_pct": statistics.mean(x["mape_pct"] for x in folds),
                    "max_fold_mape_pct": max(x["mape_pct"] for x in folds),
                    "zero_payload_violations": sum(
                        x["zero_payload_violations"] for x in folds
                    ), "folds": folds,
                })
    best = {
        engine: min(
            (report for report in reports if report["engine"] == engine),
            key=lambda report: (
                report["zero_payload_violations"],
                report["max_fold_mape_pct"], report["mean_mape_pct"],
            ),
        )
        for engine in ("vector", "scalar", "gpsimd")
    }
    result = {
        "schema": "triton-viz.factorial-dag-two-phase-model-search-v1",
        "protocol": "control-only whole-cell holdout across both order phases",
        "reports": reports, "best_by_engine": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({engine: {
        key: value for key, value in report.items() if key != "folds"
    } for engine, report in best.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
