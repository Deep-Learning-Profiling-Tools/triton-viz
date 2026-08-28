"""Fit TensorE active time from independent source-visible tiled-Dot controls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

FEATURES = ("startup_ns", "dot_ns", "lhs_tile_ns", "rhs_tile_ns", "output_tile_ns")


def _samples(paths: list[Path]) -> dict[str, list[tuple[list[float], float, str]]]:
    samples = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("status") != "ok" or row.get("kind") != "tensor_matmul_tiled":
                    continue
                m, n, k = (int(row[f"spec.{name}"]) for name in ("m", "n", "k"))
                mt, nt, kt = m // 128, n // 512, k // 128
                features = [1.0, mt * nt * kt, mt * kt, kt * nt, mt * nt]
                active_ns = float(row["profile.tensor_engine_active_time"]) * 1e9
                samples[row["spec.dtype"]].append((features, active_ns, path.name))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    parser.add_argument("--max-mean-wape", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in TensorE geometry fit")
    samples = _samples(args.inputs)
    output_rows, folds = [], []
    for dtype, values in sorted(samples.items()):
        matrix = np.asarray([features for features, _, _ in values])
        labels = np.asarray([label for _, label, _ in values])
        coefficients = nnls(matrix, labels)[0]
        output_rows.append({"dtype": dtype, "samples": len(values), **dict(zip(FEATURES, map(float, coefficients)))})
        for held_suite in sorted({suite for _, _, suite in values}):
            train = [(features, label) for features, label, suite in values if suite != held_suite]
            test = [(features, label) for features, label, suite in values if suite == held_suite]
            fold_coefficients = nnls(np.asarray([x for x, _ in train]), np.asarray([y for _, y in train]))[0]
            predictions = [(float(np.dot(x, fold_coefficients)), y) for x, y in test]
            wape = sum(abs(predicted - actual) for predicted, actual in predictions) / sum(actual for _, actual in predictions) * 100.0
            folds.append({"dtype": dtype, "held_suite": held_suite, "samples": len(test), "wape_pct": wape})
    means = {
        dtype: float(np.mean([fold["wape_pct"] for fold in folds if fold["dtype"] == dtype]))
        for dtype in sorted(samples)
    }
    report = {
        "schema": "triton-viz.tensor-source-geometry-control-cv-v2",
        "protocol": "leave-one-independent-control-suite-out NNLS",
        "metric": "per-engine WAPE",
        "features": list(FEATURES),
        "folds": folds,
        "mean_wape_pct": means,
        "gate_pct": args.max_mean_wape,
        "passed": all(value < args.max_mean_wape for value in means.values()),
        "target_postcompile_prediction_reads": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dtype", *FEATURES, "samples"))
        writer.writeheader(); writer.writerows(output_rows)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
