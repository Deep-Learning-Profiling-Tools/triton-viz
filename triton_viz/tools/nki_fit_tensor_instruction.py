"""Fit TensorE active time from static MATMUL lowering counts."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from triton_viz.tools.nki_cost_model import TensorCalibrationSurface


FIELDS = [
    "dtype",
    "instructions_per_dot",
    "intercept_ns",
    "instruction_ns",
    "instruction_count_min",
    "instruction_count_max",
    "sample_count",
]


def fit(paths: list[Path], output: Path) -> list[dict[str, object]]:
    samples: dict[tuple[str, float], list[tuple[int, float]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("status") != "ok" or row.get("kind") != "tensor_matmul_tiled":
                    continue
                try:
                    instructions = int(float(row["profile.matmul_instruction_count"]))
                    dots = int(float(row["work.dot_count"]))
                    active_ns = float(row["profile.tensor_engine_active_time"]) * 1e9
                    dtype = TensorCalibrationSurface._normalize_dtype(row["spec.dtype"])
                except (KeyError, TypeError, ValueError):
                    continue
                if instructions > 0 and dots > 0 and active_ns > 0:
                    samples[(dtype, instructions / dots)].append(
                        (instructions, active_ns)
                    )
    rows: list[dict[str, object]] = []
    for (dtype, ratio), values in sorted(samples.items()):
        counts = [value[0] for value in values]
        times = [value[1] for value in values]
        if len(set(counts)) < 2:
            continue
        slope, intercept = statistics.linear_regression(counts, times)
        if slope <= 0:
            continue
        rows.append(
            {
                "dtype": dtype,
                "instructions_per_dot": ratio,
                "intercept_ns": intercept,
                "instruction_ns": slope,
                "instruction_count_min": min(counts),
                "instruction_count_max": max(counts),
                "sample_count": len(values),
            }
        )
    if not rows:
        raise ValueError("No multi-point Tensor instruction lowering buckets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = fit(args.inputs, args.output)
    print(f"Wrote {len(rows)} Tensor instruction calibration rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
