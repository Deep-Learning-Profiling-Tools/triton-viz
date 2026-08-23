"""Freeze the control-CV-selected two-phase factorial DAG linear candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triton_viz.tools.nki_evaluate_randommixed2k_models import _feature_sets, _matrix
from triton_viz.tools.nki_fit_source_sequence_lowering import _cases
from triton_viz.tools.nki_fit_structured_controls import load_runtime_engine_baselines


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
    names = _feature_sets(rows)["contextual"]
    output = []

    vector = [row for row in rows if row["engine"] == "vector"]
    vector_coef = nnls(
        _matrix(vector, names), np.asarray([row["payload_ns"] for row in vector])
    )[0]
    output.extend({
        "engine": "vector", "dtype": "float32", "target": "fixed_ns",
        "feature": name, "coefficient": float(value),
    } for name, value in zip(names, vector_coef))

    scalar = [row for row in rows if row["engine"] == "scalar"]
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=200.0))
    pipeline.fit(
        _matrix(scalar, names), np.asarray([row["payload_ns"] for row in scalar])
    )
    scaler, ridge = pipeline.steps[0][1], pipeline.steps[1][1]
    coefficients = ridge.coef_ / scaler.scale_
    intercept = float(ridge.intercept_ - np.sum(coefficients * scaler.mean_))
    output.append({
        "engine": "scalar", "dtype": "float32", "target": "fixed_ns",
        "feature": "intercept", "coefficient": intercept,
    })
    output.extend({
        "engine": "scalar", "dtype": "float32", "target": "fixed_ns",
        "feature": name, "coefficient": float(value),
    } for name, value in zip(names, coefficients) if name != "intercept")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "engine", "dtype", "target", "feature", "coefficient"
        ))
        writer.writeheader(); writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
