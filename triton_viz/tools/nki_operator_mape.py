"""Compute aggregate MAPE from one or more ``operator_results.csv`` files.

This is the small reproducibility helper used after a hardware holdout run:

    python -m triton_viz.tools.nki_operator_mape \
        /tmp/tensor_rows512_fp32_stable/operator_results.csv \
        /tmp/tensor_rows512_bf16_stable/operator_results.csv

It reports the same mean-absolute-percentage-error metrics printed by
``nki_operator_experiments`` for individual dtype runs, and combines all input
files so every holdout point can be reported with one command.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

FIELDS = (
    "error_vs_nc_pct",
    "dma_busy_error_pct",
    "vector_busy_error_pct",
    "scalar_busy_error_pct",
    "tensor_busy_error_pct",
)


def _load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(
                row
                for row in csv.DictReader(file)
                if row.get("status") == "ok"
            )
    if not rows:
        raise ValueError("No successful operator rows found in supplied CSV files")
    return rows


def _mape(rows: list[dict[str, str]], field: str) -> float:
    values = [
        abs(float(row[field]))
        for row in rows
        if row.get(field) not in (None, "")
    ]
    return statistics.mean(values) if values else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", type=Path, help="operator_results.csv files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.csvs)
    print(f"points={len(rows)}")
    for field in FIELDS:
        print(f"{field}={_mape(rows, field):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
