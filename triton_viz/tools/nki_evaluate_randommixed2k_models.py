"""Control-only model search for frozen random mixed-2K payload schedules."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triton_viz.tools.nki_fit_source_sequence_lowering import (
    PAYLOAD_RESOLUTION_NS,
    _cases,
)
from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
)


def _feature_sets(rows: list[dict]) -> dict[str, list[str]]:
    available = sorted({
        key for row in rows for key, value in row["features"].items()
        if isinstance(value, (int, float))
    })
    op_counts = [
        name for name in available
        if name.startswith("op_") and "_x_" not in name
    ]
    basic = op_counts + [name for name in (
        "intercept", "reduction_count",
        "transcendental_count", "token_run_count", "token_change_count",
        "first_special_position", "last_special_position", "special_span",
        "affine_segment_count", "affine_segment_total_unique_ops",
        "affine_segment_max_length", "affine_segment_internal_changes",
        "dag_branch_value_count", "dag_join_node_count", "dag_max_fanout",
        "dag_max_fanin", "dag_max_live_values", "dag_critical_path_length",
    ) if name in available]
    local = basic + [
        name for name in available
        if name.startswith(("first_op_", "bigram_", "trigram_"))
        and "_x_" not in name
    ]
    positional = local + [
        name for name in available
        if name.startswith(("prefix_pos", "suffix_pos", "absolute_pos"))
    ]
    contextual = positional + [
        name for name in available
        if name.startswith((
            "run_count_", "max_run_length_", "before_special_count_",
            "after_special_count_", "affine_segment",
        ))
    ]
    return {name: list(dict.fromkeys(values)) for name, values in {
        "basic": basic, "local": local, "positional": positional,
        "contextual": contextual,
    }.items()}


def _matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray([
        [float(row["features"].get(name, 0.0)) for name in names]
        for row in rows
    ], dtype=float)


def _predict(kind: str, train_x, train_y, test_x) -> np.ndarray:
    if kind == "nnls":
        return test_x @ nnls(train_x, train_y)[0]
    if kind.startswith("ridge_"):
        alpha = float(kind.split("_", 1)[1])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    elif kind.startswith("extra_"):
        leaf = int(kind.split("_", 1)[1])
        model = ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=leaf, max_features=0.8,
            random_state=20260823, n_jobs=-1,
        )
    elif kind.startswith("forest_"):
        leaf = int(kind.split("_", 1)[1])
        model = RandomForestRegressor(
            n_estimators=80, min_samples_leaf=leaf, max_features=0.8,
            random_state=20260823, n_jobs=-1,
        )
    else:  # pragma: no cover
        raise ValueError(kind)
    model.fit(train_x, train_y)
    return np.maximum(0.0, model.predict(test_x))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--seed-min", type=int, required=True)
    parser.add_argument("--seed-max", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    rows = [row for row in _cases(args.roots, baselines)
            if args.seed_min <= row["chain"] <= args.seed_max]
    expected = (args.seed_max - args.seed_min + 1) * 3
    if len(rows) != expected:
        raise ValueError(f"expected {expected} engine rows, found {len(rows)}")
    feature_sets = _feature_sets(rows)
    model_kinds = (
        "nnls", "ridge_1", "ridge_10",
        "extra_1", "extra_2", "extra_4", "forest_2",
    )
    reports = []
    for engine in ("vector", "scalar", "gpsimd"):
        engine_rows = [row for row in rows if row["engine"] == engine]
        for feature_mode, names in feature_sets.items():
            for kind in model_kinds:
                folds = []
                for lower in range(args.seed_min, args.seed_max + 1, args.block_size):
                    upper = min(args.seed_max + 1, lower + args.block_size)
                    train = [row for row in engine_rows
                             if not lower <= row["chain"] < upper]
                    test = [row for row in engine_rows
                            if lower <= row["chain"] < upper]
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
                        "seed_block": [lower, upper - 1],
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
                    ),
                    "folds": folds,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "triton-viz.randommixed2k-model-search-v1",
        "protocol": "control-only contiguous seed-block CV",
        "seed_range": [args.seed_min, args.seed_max],
        "payload_resolution_ns": PAYLOAD_RESOLUTION_NS,
        "reports": reports,
        "best_by_engine": {
            engine: min(
                (row for row in reports if row["engine"] == engine),
                key=lambda row: (
                    row["zero_payload_violations"],
                    row["max_fold_mape_pct"], row["mean_mape_pct"],
                ),
            )
            for engine in ("vector", "scalar", "gpsimd")
        },
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        engine: min(
            (row for row in reports if row["engine"] == engine),
            key=lambda row: (
                row["zero_payload_violations"],
                row["max_fold_mape_pct"], row["mean_mape_pct"],
            ),
        ) | {"folds": "omitted"}
        for engine in ("vector", "scalar", "gpsimd")
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
