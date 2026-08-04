"""Fit latency microbenchmark slopes from ``results.jsonl``.

Use this after a repeat sweep (e.g. pointer chasing with repeat=1,2,4,8...).
The intercept absorbs launch/fixed overhead; the slope estimates per-dependent
operation latency in microseconds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latency_us(row: dict[str, Any], source: str, percentile: str) -> float | None:
    data = row.get("latency_percentiles") or {}
    table = data.get(source) or {}
    value = table.get(percentile)
    return float(value) if value is not None else None


def _x_value(row: dict[str, Any]) -> float | None:
    work = row.get("work") or {}
    if "dependent_hbm_loads" in work:
        return float(work["dependent_hbm_loads"])
    if "serialized_roundtrips" in work:
        return float(work["serialized_roundtrips"])
    repeat = row.get("spec", {}).get("repeat")
    return float(repeat) if repeat is not None else None


def _fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two points for slope fit")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("all x values are identical")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {"points": float(n), "slope_us": slope, "intercept_us": intercept, "r2": r2}


def fit_results(rows: list[dict[str, Any]], source: str = "nc_latency", percentile: str = "p50_us") -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        spec = row.get("spec") or {}
        if spec.get("kind") not in {"pointer_chase", "dma_roundtrip_latency"}:
            continue
        x = _x_value(row)
        y = _latency_us(row, source, percentile) or _latency_us(row, "latency", percentile)
        if x is None or y is None:
            continue
        # Group by every parameter except repeat, because repeat is the x-axis.
        key_items = tuple(sorted((k, v) for k, v in spec.items() if k not in {"repeat", "modes"}))
        groups[key_items].append((x, y))
    out: dict[str, Any] = {"source": source, "percentile": percentile, "fits": []}
    for key, pairs in groups.items():
        pairs = sorted(pairs)
        if len(pairs) < 2:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        try:
            fit = _fit(xs, ys)
        except ValueError as exc:
            fit = {"error": str(exc)}
        out["fits"].append({"group": dict(key), "x": xs, "y_us": ys, **fit})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--source", choices=["nc_latency", "latency"], default="nc_latency")
    parser.add_argument("--percentile", default="p50_us")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    data = fit_results(_load_rows(args.results_jsonl), source=args.source, percentile=args.percentile)
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
