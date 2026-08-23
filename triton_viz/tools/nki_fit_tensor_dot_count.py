"""Fit TensorE active time from source Dot count on independent controls."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FIELDS = ("dtype", "startup_ns", "dot_ns", "dot_count_min", "dot_count_max", "samples")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    samples = defaultdict(list)
    for path in args.inputs:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("status") != "ok" or row.get("kind") != "tensor_matmul_tiled":
                    continue
                dots = int(float(row["work.dot_count"]))
                active_ns = float(row["profile.tensor_engine_active_time"]) * 1e9
                if dots > 0 and active_ns > 0:
                    samples[row["spec.dtype"]].append((dots, active_ns))
    rows = []
    for dtype, values in sorted(samples.items()):
        xs, ys = zip(*values)
        slope, intercept = statistics.linear_regression(xs, ys)
        if slope <= 0:
            raise ValueError(f"Non-positive Tensor Dot slope for {dtype}")
        rows.append({"dtype": dtype, "startup_ns": max(0.0, intercept), "dot_ns": slope,
                     "dot_count_min": min(xs), "dot_count_max": max(xs), "samples": len(xs)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} source-Dot Tensor calibration rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
