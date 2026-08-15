"""Compute aggregate MAPE from operator or evaluation CSV files.

This is the small reproducibility helper used after a hardware holdout run.
It accepts either debug-stage ``operator_results.csv`` files from
``nki_operator_experiments`` or the frozen ``evaluation/*.csv`` replays written
by ``nki_cost_model_pipeline evaluate``:

    python -m triton_viz.tools.nki_operator_mape \
        /tmp/nki_cost_model_run/evaluation/*.csv

It reports the same mean-absolute-percentage-error metrics printed by the
pipeline for individual dtype runs, and combines all input files so every
holdout point (existing elementwise/norm/reduction/matmul operators plus the
new tiled attention operator) can be reported with one command.
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
EVALUATION_FIELDS = (
    "nc_error_pct",
    "tensor_error_pct",
    "dma_error_pct",
)


def _load_rows(paths: list[Path]) -> list[dict[str, str]]:
    if not paths:
        raise ValueError("No CSV files supplied")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(
                row
                for row in csv.DictReader(file)
                if row.get("status") in (None, "", "ok")
            )
    if not rows:
        raise ValueError("No successful operator rows found in supplied CSV files")
    return rows


def _is_evaluation(rows: list[dict[str, str]]) -> bool:
    return any("nc_error_pct" in row for row in rows)


def _mape(rows: list[dict[str, str]], field: str) -> float:
    values = [
        abs(float(row[field]))
        for row in rows
        if row.get(field) not in (None, "")
    ]
    return statistics.mean(values) if values else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", type=Path, help="operator/evaluation CSV files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.csvs)
    print(f"points={len(rows)}")
    fields = EVALUATION_FIELDS if _is_evaluation(rows) else FIELDS
    for field in fields:
        print(f"{field}={_mape(rows, field):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
