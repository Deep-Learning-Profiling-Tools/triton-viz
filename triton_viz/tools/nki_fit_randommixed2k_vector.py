"""Fit a Vector-only source model from frozen random mixed-2K controls."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from triton_viz.tools.nki_fit_source_sequence_lowering import (
    PAYLOAD_RESOLUTION_NS,
    _cases,
)
from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
    runtime_engine_baseline_ns,
)


def _features(rows: list[dict]) -> list[str]:
    available = {
        key for row in rows for key, value in row["features"].items()
        if isinstance(value, (int, float))
    }
    topology = [name for name in (
        "dag_branch_value_count", "dag_join_node_count", "dag_max_fanout",
        "dag_max_fanin", "dag_max_live_values", "dag_critical_path_length",
    ) if name in available]
    counts = ["intercept"] + sorted(
        name for name in available
        if name.startswith("op_") and "_x_" not in name
    ) + ["reduction_count", "transcendental_count"] + topology
    order = sorted(
        key for key in available
        if key.startswith("bigram_") or key.startswith("first_op_")
    )
    return counts + order


def _matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.asarray([
        [float(row["features"].get(name, 0.0)) for name in features]
        for row in rows
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    parser.add_argument("--seed-min", type=int, default=100)
    parser.add_argument("--seed-max", type=int, default=163)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument(
        "--engines", nargs="+", choices=("vector", "scalar", "gpsimd"),
        default=("vector",),
    )
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    rows = [
        row for row in _cases(args.roots, baselines)
        if row["engine"] in args.engines
        and args.seed_min <= row["chain"] <= args.seed_max
    ]
    expected = (args.seed_max - args.seed_min + 1) * len(args.engines)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} frozen controls, found {len(rows)}")
    features = _features(rows)
    folds = []
    for engine in args.engines:
        engine_rows = [row for row in rows if row["engine"] == engine]
        for lower in range(args.seed_min, args.seed_max + 1, args.block_size):
            upper = min(args.seed_max + 1, lower + args.block_size)
            train = [
                row for row in engine_rows
                if not lower <= row["chain"] < upper
            ]
            test = [
                row for row in engine_rows if lower <= row["chain"] < upper
            ]
            coefficients = nnls(
                _matrix(train, features),
                np.asarray([row["payload_ns"] for row in train]),
            )[0]
            errors = []
            zero_payload_violations = 0
            for prediction, row in zip(
                _matrix(test, features) @ coefficients, test
            ):
                if row["payload_ns"] > PAYLOAD_RESOLUTION_NS:
                    errors.append(
                        abs(prediction - row["payload_ns"])
                        / row["payload_ns"] * 100.0
                    )
                elif prediction > PAYLOAD_RESOLUTION_NS:
                    zero_payload_violations += 1
            folds.append({
                "engine": engine,
                "seed_block": [lower, upper - 1], "samples": len(test),
                "positive_payload_samples": len(errors),
                "zero_payload_violations": zero_payload_violations,
                "mape_pct": (
                    statistics.mean(errors)
                    if errors else (100.0 if zero_payload_violations else 0.0)
                ),
                "max_case_mape_pct": max(errors, default=0.0),
            })
    report = {
        "schema": "triton-viz.randommixed2k-payload-cv-v2",
        "protocol": "contiguous_seed_blocks; payload-to-payload MAPE",
        "payload_resolution_ns": PAYLOAD_RESOLUTION_NS,
        "seed_range": [args.seed_min, args.seed_max],
        "feature_names": features, "folds": folds,
        "mean_mape_pct": statistics.mean(fold["mape_pct"] for fold in folds),
        "max_fold_mape_pct": max(fold["mape_pct"] for fold in folds),
        "by_engine": {
            engine: {
                "mean_mape_pct": statistics.mean(
                    fold["mape_pct"] for fold in folds
                    if fold["engine"] == engine
                ),
                "max_fold_mape_pct": max(
                    fold["mape_pct"] for fold in folds
                    if fold["engine"] == engine
                ),
                "zero_payload_violations": sum(
                    fold["zero_payload_violations"] for fold in folds
                    if fold["engine"] == engine
                ),
            }
            for engine in args.engines
        },
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
        output_rows.extend(
            {"engine": engine, "dtype": "float32", "target": "fixed_ns",
             "feature": feature, "coefficient": float(value)}
            for feature, value in zip(features, coefficients)
        )
        for partition in (1, 16, 128):
            output_rows.append({
                "engine": engine, "dtype": "float32",
                "target": "runtime_baseline_ns", "feature": f"partition_p{partition}",
                "coefficient": runtime_engine_baseline_ns(
                    baselines, "float32", partition, engine
                ),
            })
        output_rows.append({
            "engine": engine, "dtype": "float32", "target": "effective_count",
            "feature": "long_mixed_tile2k_applicable", "coefficient": 1e-9,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "engine", "dtype", "target", "feature", "coefficient"
        ))
        writer.writeheader(); writer.writerows(output_rows)
    print(json.dumps({
        "mean_mape_pct": report["mean_mape_pct"],
        "max_fold_mape_pct": report["max_fold_mape_pct"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
