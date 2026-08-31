"""Fit the on-chip PSUM/SBUF copy latency surface from repeat-differenced controls.

The control suite (``microbench/inf2_nki/configs/onchip_copy_disjoint_v2.json``)
sweeps ``repeat`` for each ``(dtype, free width)``.  Every kernel pays the same
fixed initialization/store/runtime instructions once, so regressing measured
VectorE active time on ``repeat`` cancels that constant and leaves the
*incremental* on-chip transfer cost.  Those per-width slopes are then regressed
on the free width to give ``startup_ns + ns_per_free_elem * free_elements``.

The gate is leave-one-width-out: hold one width out, refit on the remaining
widths, and predict the held-out slope.  Nothing enters production without
passing, and the fitter mechanically refuses target artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCHEMA = "triton-viz.onchip-transfer-surface-v1"
FIELDNAMES = [
    "engine",
    "dtype",
    "input_stream_count",
    "startup_ns",
    "ns_per_free_elem",
    "domain_min_free",
    "domain_max_free",
    "instruction_count_min",
    "instruction_count_max",
    "points",
    "excluded_branch_points",
    "run_ids",
    "source",
]


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return ``(intercept, slope)`` of the least-squares line through the points."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two points to fit a line")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("degenerate fit: all abscissae are equal")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("spec.kind") == "onchip_copy"
        ]


def width_slopes(rows: list[dict[str, str]]) -> dict[str, dict[int, float]]:
    """Return ``{dtype: {free_elements: incremental copy latency in ns}}``."""
    buckets: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for row in rows:
        key = (row["spec.dtype"], int(row["spec.f"]))
        repeat = float(row["spec.repeat"])
        active_ns = float(row["profile.vector_engine_active_time"]) * 1e9
        buckets.setdefault(key, []).append((repeat, active_ns))
    slopes: dict[str, dict[int, float]] = {}
    for (dtype, free), points in buckets.items():
        if len(points) < 2:
            raise SystemExit(
                f"{dtype} f={free}: need >=2 repeat points to difference out the fixed cost"
            )
        _intercept, slope = _ols([p for p, _ in points], [a for _, a in points])
        slopes.setdefault(dtype, {})[free] = slope
    return slopes


def cross_validate(slopes: dict[str, dict[int, float]], gate_pct: float) -> dict:
    folds = []
    for dtype in sorted(slopes):
        widths = sorted(slopes[dtype])
        if len(widths) < 3:
            raise SystemExit(
                f"{dtype}: leave-one-width-out needs >=3 widths, found {len(widths)}"
            )
        for held in widths:
            train = [w for w in widths if w != held]
            intercept, per_elem = _ols(
                [float(w) for w in train], [slopes[dtype][w] for w in train]
            )
            actual = slopes[dtype][held]
            predicted = intercept + per_elem * held
            folds.append(
                {
                    "dtype": dtype,
                    "held_width": held,
                    "actual_ns": actual,
                    "predicted_ns": predicted,
                    "ape_pct": abs(predicted - actual) / abs(actual) * 100.0,
                }
            )
    mean_wape = sum(f["ape_pct"] for f in folds) / len(folds)
    return {
        "protocol": "repeat-difference then leave-one-width-out",
        "mean_wape_pct": mean_wape,
        "gate_pct": gate_pct,
        "pass": mean_wape < gate_pct,
        "folds": folds,
        "target_postcompile_prediction_reads": False,
    }


def fit(rows: list[dict[str, str]], source: Path) -> list[dict[str, object]]:
    slopes = width_slopes(rows)
    run_ids = sorted({row["run_id"] for row in rows})
    instr = [int(row["work.logical_instructions"]) for row in rows]
    records = []
    for dtype in sorted(slopes):
        widths = sorted(slopes[dtype])
        startup, per_elem = _ols(
            [float(w) for w in widths], [slopes[dtype][w] for w in widths]
        )
        records.append(
            {
                "engine": "vector",
                "dtype": dtype,
                "input_stream_count": 1,
                "startup_ns": startup,
                "ns_per_free_elem": per_elem,
                "domain_min_free": min(widths),
                "domain_max_free": max(widths),
                "instruction_count_min": 1,
                "instruction_count_max": 1,
                "points": len(widths),
                "excluded_branch_points": 0,
                "run_ids": ";".join(run_ids),
                "source": str(source),
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path,
                        help="aggregated microbench CSV for the onchip_copy suite")
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument("--max-mean-wape", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in on-chip copy calibration")

    rows = load_rows(args.results_csv)
    if not rows:
        raise SystemExit(f"no ok onchip_copy rows in {args.results_csv}")
    slopes = width_slopes(rows)
    report = cross_validate(slopes, args.max_mean_wape)
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["pass"]:
        raise SystemExit(
            f"leave-one-width-out mean WAPE {report['mean_wape_pct']:.6f}% "
            f">= gate {args.max_mean_wape}%; refusing to emit a production surface"
        )

    records = fit(rows, args.results_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
