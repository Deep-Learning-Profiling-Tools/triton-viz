"""Fit TensorE active time from source-visible tiled-Dot geometry controls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cv-output", type=Path,
        help="Optional leave-one-input-suite-out NNLS audit JSON.",
    )
    args = parser.parse_args(argv)
    samples = defaultdict(list)
    for path in args.inputs:
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            if row.get("status") != "ok" or row.get("kind") != "tensor_matmul_tiled":
                continue
            m, n, k = (int(row[f"spec.{name}"]) for name in ("m", "n", "k"))
            mt, nt, kt = m // 128, n // 512, k // 128
            features = [1.0, mt * nt * kt, mt * kt, kt * nt, mt * nt]
            active_ns = float(row["profile.tensor_engine_active_time"]) * 1e9
            samples[row["spec.dtype"]].append((features, active_ns, path.name))
    rows = []
    for dtype, values in sorted(samples.items()):
        coefficients = nnls(
            np.asarray([x for x, _, _ in values]),
            np.asarray([y for _, y, _ in values]),
        )[0]
        rows.append(dict(zip(
            ("startup_ns", "dot_ns", "lhs_tile_ns", "rhs_tile_ns", "output_tile_ns"),
            map(float, coefficients),
        )) | {"dtype": dtype, "samples": len(values)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dtype", "startup_ns", "dot_ns", "lhs_tile_ns", "rhs_tile_ns", "output_tile_ns", "samples"))
        writer.writeheader(); writer.writerows(rows)
    if args.cv_output:
        folds = []
        for dtype, values in sorted(samples.items()):
            for held_suite in sorted({suite for _, _, suite in values}):
                train = [(x, y) for x, y, suite in values if suite != held_suite]
                test = [(x, y) for x, y, suite in values if suite == held_suite]
                coefficients = nnls(
                    np.asarray([x for x, _ in train]),
                    np.asarray([y for _, y in train]),
                )[0]
                errors = [
                    abs(float(np.dot(x, coefficients)) - y) / y * 100.0
                    for x, y in test
                ]
                folds.append({
                    "dtype": dtype,
                    "held_suite": held_suite,
                    "samples": len(test),
                    "mape_pct": float(np.mean(errors)),
                    "coefficients": list(map(float, coefficients)),
                })
        payload = {
            "protocol": "leave_one_input_suite_out_nnls",
            "feature_order": [
                "startup", "dot_count", "lhs_tile_count",
                "rhs_tile_count", "output_tile_count",
            ],
            "folds": folds,
            "suite_mean_mape_pct": float(np.mean([x["mape_pct"] for x in folds])),
            "max_fold_mape_pct": float(max(x["mape_pct"] for x in folds)),
        }
        args.cv_output.parent.mkdir(parents=True, exist_ok=True)
        args.cv_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
