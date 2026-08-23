"""Fit source-signature engine/NC surfaces from disjoint geometry controls."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ENGINES = ("vector", "scalar", "gpsimd")
FIELDS = (
    "signature_json",
    "dtype",
    "rows",
    "cols",
    "nc_us",
    "vector_us",
    "scalar_us",
    "gpsimd_us",
    "case",
)


def source_signature(events: list[dict], dtype: str, rows: int) -> str:
    compute = [
        event
        for event in events
        if event.get("op") in {"compute", "binary", "reduce_sum"}
    ]
    payload = {
        "dtype": str(dtype),
        "rows": int(rows),
        "ops": sorted(
            Counter(str(event.get("api_op") or event.get("op") or "") for event in compute).items()
        ),
        "arities": sorted(
            Counter(len(event.get("input_ptrs") or ()) for event in compute).items()
        ),
        "masks": sorted(
            Counter(
                bool(event.get("mask_provided"))
                for event in events
                if event.get("op") in {"load", "store"}
            ).items()
        ),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class SourceSignatureSurface:
    def __init__(self, rows: list[dict[str, str]]):
        self.points: dict[str, list[tuple[int, dict[str, float]]]] = {}
        for row in rows:
            values = {
                name: float(row[f"{name}_us"])
                for name in ("nc", *ENGINES)
            }
            self.points.setdefault(row["signature_json"], []).append(
                (int(row["cols"]), values)
            )

    @classmethod
    def from_csv(cls, path: Path) -> "SourceSignatureSurface":
        with path.open(encoding="utf-8", newline="") as file:
            return cls(list(csv.DictReader(file)))

    @staticmethod
    def _linear(points: list[tuple[int, float]], target: int) -> float:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            / denominator
            if denominator
            else 0.0
        )
        return max(0.0, mean_y + slope * (target - mean_x))

    def predict(
        self, events: list[dict], dtype: str, rows: int, cols: int
    ) -> tuple[dict[str, float] | None, str]:
        points = self.points.get(source_signature(events, dtype, rows), [])
        unique_cols = sorted({point[0] for point in points})
        if len(unique_cols) < 2:
            return None, "ood_insufficient_signature_points"
        medians = [
            (
                column,
                {
                    name: statistics.median(
                        value[name] for measured, value in points if measured == column
                    )
                    for name in ("nc", *ENGINES)
                },
            )
            for column in unique_cols
        ]
        result = {
            name: self._linear(
                [(column, values[name]) for column, values in medians], cols
            )
            for name in ("nc", *ENGINES)
        }
        if cols in unique_cols:
            match = "exact_control_geometry"
        elif min(unique_cols) < cols < max(unique_cols):
            match = "interpolated_control_geometry"
        else:
            match = "extrapolated_control_geometry"
        return result, match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output_rows = []
    for root in args.roots:
        result_path = root / "operator_results.csv"
        if not result_path.is_file():
            continue
        with result_path.open(encoding="utf-8", newline="") as file:
            source_rows = list(csv.DictReader(file))
        for row in source_rows:
            if row.get("status") != "ok":
                continue
            case = f"{row['op']}__r{row['rows']}__c{row['cols']}__{row['dtype']}"
            trace = root / case / "trace.jsonl"
            summary = root / case / "hardware/explorer_summary.json"
            if not trace.is_file() or not summary.is_file():
                continue
            events = [json.loads(line) for line in trace.read_text().splitlines() if line]
            profile = next(iter(json.loads(summary.read_text()).values()))
            output_rows.append(
                {
                    "signature_json": source_signature(events, row["dtype"], int(row["rows"])),
                    "dtype": row["dtype"],
                    "rows": row["rows"],
                    "cols": row["cols"],
                    "nc_us": row["hardware_nc_p50_us"],
                    "vector_us": float(profile.get("vector_engine_active_time", 0.0)) * 1e6,
                    "scalar_us": float(profile.get("scalar_engine_active_time", 0.0)) * 1e6,
                    "gpsimd_us": float(profile.get("gpsimd_engine_active_time", 0.0)) * 1e6,
                    "case": case,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} source-signature control rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
