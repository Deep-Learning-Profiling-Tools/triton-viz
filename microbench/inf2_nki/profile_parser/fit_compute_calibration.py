"""Fit Level-B per-instruction engine cost from an exported microbench CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


FIELDS = [
    "engine", "dtype", "input_stream_count", "startup_ns",
    "ns_per_free_elem", "instruction_count_min", "instruction_count_max",
    "points", "source",
]


def fit_rows(path: Path) -> list[dict]:
    groups: dict[tuple[str, str, int], list[tuple[int, float, int]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            kind = row.get("kind")
            engine = {"vector_add": "vector", "scalar_exp": "scalar"}.get(kind)
            if not engine or row.get("status") != "ok":
                continue
            try:
                free = int(float(row["work.free_dimension_elements"] or row["spec.f"]))
            except (KeyError, TypeError, ValueError):
                try:
                    free = int(float(row["spec.f"]))
                except (KeyError, TypeError, ValueError):
                    continue
            try:
                streams = int(float(row.get("work.input_stream_count") or 1))
                active_ns = float(row[f"profile.{engine}_engine_active_time"]) * 1e9
                count = int(float(row[f"profile.{engine}_engine_instruction_count"]))
            except (KeyError, TypeError, ValueError):
                continue
            if free > 0 and active_ns > 0 and count > 0:
                groups[(engine, row.get("spec.dtype") or "float32", streams)].append(
                    (free, active_ns / count, count)
                )
    output: list[dict] = []
    for (engine, dtype, streams), points in sorted(groups.items()):
        if len(points) < 2:
            continue
        counts = [point[2] for point in points]
        # A shape-dependent instruction count means the source pattern changed;
        # it must be represented by Level A instead of folded into Level B.
        if min(counts) != max(counts):
            continue
        slope, intercept = np.polyfit(
            np.asarray([point[0] for point in points], dtype=float),
            np.asarray([point[1] for point in points], dtype=float), 1,
        )
        if not all(math.isfinite(value) and value >= 0 for value in (intercept, slope)):
            continue
        output.append({
            "engine": engine, "dtype": dtype, "input_stream_count": streams,
            "startup_ns": intercept, "ns_per_free_elem": slope,
            "instruction_count_min": min(counts), "instruction_count_max": max(counts),
            "points": len(points), "source": str(path),
        })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = fit_rows(args.csv)
    if not rows:
        raise SystemExit("No stable instruction-count compute groups found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} compute calibration rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
