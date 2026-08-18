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
import json
import math
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


def _case_key(row: dict[str, str]) -> tuple[str, ...] | None:
    """Return a stable workload identity for duplicate-safe aggregation."""
    case = row.get("case")
    dtype = row.get("dtype", "")
    hardware = (
        row.get("hardware_fingerprint")
        or row.get("compiler_fingerprint")
        or row.get("instance_type")
        or ""
    )
    if case:
        return case, dtype, hardware
    if not any(row.get(field) for field in ("op", "rows", "cols")):
        return None
    return (
        row.get("op", ""),
        row.get("rows", ""),
        row.get("cols", ""),
        dtype,
        hardware,
    )


def _deduplicate_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int, int]:
    """Deduplicate workloads using order-independent median measurements.

    Formal/full splits may repeat the same workload but contain independently
    sampled hardware measurements, so their errors need not be bit-identical.
    Treating one arbitrarily as authoritative would make the result depend on
    CSV argument order.  Instead aggregate every numeric error field by median
    and report how many duplicate groups had non-identical values.
    """
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    duplicate_rows = 0
    conflicting_groups = 0
    fields = EVALUATION_FIELDS if _is_evaluation(rows) else FIELDS
    for row_index, row in enumerate(rows):
        key = _case_key(row)
        if key is None:
            grouped[("__anonymous_row__", str(row_index))] = [row]
            continue
        grouped.setdefault(key, []).append(row)
    unique_rows = []
    for group in grouped.values():
        duplicate_rows += len(group) - 1
        merged = dict(group[0])
        conflict = False
        for field in fields:
            values = [
                float(row[field])
                for row in group
                if row.get(field) not in (None, "")
            ]
            if not values:
                continue
            if any(
                not math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-9)
                for value in values[1:]
            ):
                conflict = True
            merged[field] = str(statistics.median(values))
        conflicting_groups += int(conflict)
        unique_rows.append(merged)
    return unique_rows, duplicate_rows, conflicting_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", type=Path, help="operator/evaluation CSV files")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the aggregate audit as one JSON object.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.csvs)
    unique_rows, duplicate_rows, conflicting_groups = _deduplicate_rows(rows)
    fields = EVALUATION_FIELDS if _is_evaluation(rows) else FIELDS
    report = {
        "rows": len(rows),
        "unique_cases": len(unique_rows),
        "duplicate_rows": duplicate_rows,
        "conflicting_duplicate_groups": conflicting_groups,
        "split_weighted_rows_mape": {
            field: _mape(rows, field) for field in fields
        },
        "unique_case_mape": {
            field: _mape(unique_rows, field) for field in fields
        },
    }
    if args.json:
        print(json.dumps(report, sort_keys=True))
        return 0
    # Compatibility alias for older scripts. It is explicitly row-weighted,
    # not a claim that all rows are unique workloads.
    print(f"points={len(rows)}")
    print(f"rows={len(rows)}")
    print(f"unique_cases={len(unique_rows)}")
    print(f"duplicate_rows={duplicate_rows}")
    print(f"conflicting_duplicate_groups={conflicting_groups}")
    for field in fields:
        print(f"{field}={report['split_weighted_rows_mape'][field]:.4f}")
        print(
            f"split_weighted_rows_{field}="
            f"{report['split_weighted_rows_mape'][field]:.4f}"
        )
        print(f"unique_case_{field}={report['unique_case_mape'][field]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
