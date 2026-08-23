"""Merge only independently CV-qualified source-only payload subdomains."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-csv", type=Path, required=True)
    parser.add_argument("--long-vector-csv", type=Path)
    parser.add_argument("--no-long-vector", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    atomic = _rows(args.atomic_csv)
    if not args.no_long_vector and args.long_vector_csv is None:
        parser.error("--long-vector-csv is required unless --no-long-vector is set")
    long_vector = [] if args.no_long_vector else _rows(args.long_vector_csv)
    output: list[dict[str, str]] = []

    # Atomic coefficients are accepted only inside the independently audited
    # p128 + wide-allocation + explicit-compute-mask + one-token domain.
    for row in atomic:
        if row["target"] == "fixed_ns" and "compute_mask" in row["feature"]:
            output.append(row)

    # The random long-grammar model is independently accepted for Vector only.
    for row in long_vector:
        if row["engine"] == "vector" and row["target"] == "fixed_ns":
            output.append(row)

    # Runtime baselines are independent controls, not target-derived residuals.
    seen = set()
    for row in atomic + long_vector:
        key = (row["engine"], row["dtype"], row["target"], row["feature"])
        if row["target"] == "runtime_baseline_ns" and key not in seen:
            output.append(row)
            seen.add(key)

    for engine in ("vector", "scalar"):
        output.append({
            "engine": engine,
            "dtype": "float32",
            "target": "effective_count",
            "feature": "atomic_wide_masked_applicable",
            "coefficient": "1e-9",
        })
    if not args.no_long_vector:
        output.append({
            "engine": "vector",
            "dtype": "float32",
            "target": "effective_count",
            "feature": "long_mixed_tile2k_applicable",
            "coefficient": "1e-9",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("engine", "dtype", "target", "feature", "coefficient"),
        )
        writer.writeheader()
        writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
