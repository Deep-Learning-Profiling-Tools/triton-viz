"""Control-only audit of additive source primitives on unseen compositions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
    runtime_engine_baseline_ns,
)


ATOMIC_KIND_TOKEN = {
    "primitive_add": "add",
    "primitive_subtract": "subtract",
    "primitive_multiply": "multiply",
    "primitive_divide": "divide",
    "primitive_exp": "exp",
    "primitive_log": "log",
    "primitive_rsqrt": "rsqrt",
    "primitive_reduce_sum": "reduce_sum",
    "elementwise_maximum": "maximum",
    "elementwise_maximum_masked": "maximum",
    "elementwise_sigmoid": "sigmoid",
    "elementwise_sigmoid_masked": "sigmoid",
    "elementwise_multiply": "multiply",
    "elementwise_multiply_masked": "multiply",
}


def _profile(case: Path) -> dict:
    path = case / "hardware/explorer_summary.json"
    return next(iter(json.loads(path.read_text(encoding="utf-8")).values()), {})


def _one_region(case: Path) -> dict | None:
    path = case / "dependency_trace.jsonl"
    if not path.is_file():
        return None
    regions = {
        json.dumps(event["region_ir"], sort_keys=True): event["region_ir"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for event in [json.loads(line)]
        if event.get("region_ir") is not None
    }
    return next(iter(regions.values())) if len(regions) == 1 else None


def _primitive_specs(case: Path) -> list[tuple[str, int]]:
    specs = []
    for line in (case / "dependency_trace.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("region_ir") is None:
            continue
        token = str(event.get("api_op") or event.get("op") or "")
        arity = len(event.get("input_ptrs") or event.get("input_dtypes") or [])
        specs.append((token, min(2, arity)))
    return specs


def _interpolate(points: list[tuple[int, float]], free_dim: int) -> float:
    points = sorted(points)
    exact = [value for size, value in points if size == free_dim]
    if exact:
        return statistics.median(exact)
    lower = max((point for point in points if point[0] <= free_dim), default=points[0])
    upper = min((point for point in points if point[0] >= free_dim), default=points[-1])
    if lower[0] == upper[0]:
        return lower[1]
    weight = (math.log2(free_dim) - math.log2(lower[0])) / (
        math.log2(upper[0]) - math.log2(lower[0])
    )
    return lower[1] + weight * (upper[1] - lower[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-root", nargs="+", type=Path, required=True)
    parser.add_argument("--composition-root", nargs="+", type=Path, required=True)
    parser.add_argument("--runtime-overhead-results", type=Path, required=True)
    parser.add_argument("--aggregation", choices=["sum", "max"], default="sum")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    surfaces: dict[tuple[str, int, bool, str, int, str], list[tuple[int, float]]] = {}
    for root in args.atomic_root:
        for case in root.glob("control_*"):
            match = re.match(r"control_(.*?)__p", case.name)
            kind = match.group(1) if match else ""
            token = ATOMIC_KIND_TOKEN.get(kind)
            region = _one_region(case)
            if token is None or region is None or not (case / "hardware/explorer_summary.json").is_file():
                continue
            dtype = str(region["dtype"])
            partition = int(region.get("partition_count") or 1)
            free = int(region.get("logical_free_dim") or region["free_dim"])
            matching_specs = [spec for spec in _primitive_specs(case) if spec[0] == token]
            if len(matching_specs) != 1:
                continue
            arity = matching_specs[0][1]
            masked = bool(region.get("has_mask_or_tail"))
            profile = _profile(case)
            for engine in ("vector", "scalar"):
                payload = max(
                    0.0,
                    float(profile.get(f"{engine}_engine_active_time", 0.0)) * 1e9
                    - runtime_engine_baseline_ns(baselines, dtype, partition, engine),
                )
                surfaces.setdefault(
                    (token, arity, masked, dtype, partition, engine), []
                ).append((free, payload))

    rows = []
    for root in args.composition_root:
        for case in root.glob("control_*"):
            match = re.match(r"control_(.*?)__p", case.name)
            kind = match.group(1) if match else ""
            if kind in ATOMIC_KIND_TOKEN:
                continue
            region = _one_region(case)
            summary = case / "hardware/explorer_summary.json"
            if region is None or not summary.is_file():
                continue
            specs = _primitive_specs(case)
            dtype = str(region["dtype"])
            partition = int(region.get("partition_count") or 1)
            free = int(region.get("logical_free_dim") or region["free_dim"])
            masked = bool(region.get("has_mask_or_tail"))
            profile = _profile(case)
            missing = sorted(
                f"{token}/{arity}"
                for token, arity in specs
                if not any(
                    key[0] == token and key[1] == arity and key[3] == dtype
                    for key in surfaces
                )
            )
            if missing:
                continue
            for engine in ("vector", "scalar"):
                primitive_payloads = []
                for token, arity in specs:
                    candidates = [
                        (abs(control_p - partition), points)
                        for (
                            control_token,
                            control_arity,
                            control_masked,
                            control_dtype,
                            control_p,
                            control_engine,
                        ), points in surfaces.items()
                        if control_token == token
                        and control_arity == arity
                        and control_dtype == dtype
                        and control_engine == engine
                        and control_masked == masked
                    ]
                    if not candidates:
                        candidates = [
                            (abs(control_p - partition), points)
                            for (
                                control_token,
                                control_arity,
                                _control_masked,
                                control_dtype,
                                control_p,
                                control_engine,
                            ), points in surfaces.items()
                            if control_token == token
                            and control_arity == arity
                            and control_dtype == dtype
                            and control_engine == engine
                        ]
                    primitive_payloads.append(_interpolate(min(candidates)[1], free))
                predicted = runtime_engine_baseline_ns(
                    baselines, dtype, partition, engine
                ) + (
                    max(primitive_payloads, default=0.0)
                    if args.aggregation == "max"
                    else sum(primitive_payloads)
                )
                actual = float(profile.get(f"{engine}_engine_active_time", 0.0)) * 1e9
                rows.append({
                    "case": case.name,
                    "kind": kind,
                    "dtype": dtype,
                    "partition_dim": partition,
                    "free_dim": free,
                    "engine": engine,
                    "primitive_count": len(primitive_payloads),
                    "reduction_count": int(region.get("reduction_count") or 0),
                    "primitive_payload_sum_ns": sum(primitive_payloads),
                    "primitive_payload_max_ns": max(primitive_payloads, default=0.0),
                    "runtime_baseline_ns": runtime_engine_baseline_ns(
                        baselines, dtype, partition, engine
                    ),
                    "actual_active_ns": actual,
                    "predicted_active_ns": predicted,
                    "error_pct": abs(predicted - actual) / max(actual, 1.0) * 100.0,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader(); writer.writerows(rows)
    report = {
        "schema": "triton-viz.atomic-composition-cv-v1",
        "protocol": "atomic-primitives-to-unseen-control-compositions",
        "aggregation": args.aggregation,
        "samples": len(rows),
        "vector_mape_pct": statistics.mean(r["error_pct"] for r in rows if r["engine"] == "vector") if rows else None,
        "scalar_mape_pct": statistics.mean(r["error_pct"] for r in rows if r["engine"] == "scalar") if rows else None,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
