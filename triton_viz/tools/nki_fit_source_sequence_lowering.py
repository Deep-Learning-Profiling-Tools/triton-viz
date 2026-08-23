"""Fit and cross-validate operator-free source-sequence lowering controls."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np

from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
    runtime_engine_baseline_ns,
)
from triton_viz.tools.nki_region_ir import build_region_ir, compositional_features

SEQUENCE_FEATURES = [
    "two_input_elementwise_count",
    "two_input_elementwise_count_x_free",
    "transcendental_count",
    "transcendental_count_x_free",
    "reduction_count",
    "reduction_count_x_free",
    "has_transcendental",
    "has_reduction",
    "arithmetic_only",
    "free_dim_linear",
    "two_reduction_interaction",
    "two_reduction_interaction_x_free",
]
SEQUENCE_FEATURES += [
    f"{name}_x_partition_p128" for name in tuple(SEQUENCE_FEATURES)
]
_ORDER_TOKENS = ("add", "subtract", "multiply", "exp", "reduce_sum")
SEQUENCE_FEATURES += [f"first_op_{token}" for token in _ORDER_TOKENS]
SEQUENCE_FEATURES += [
    f"bigram_{lhs}__{rhs}"
    for lhs in _ORDER_TOKENS
    for rhs in _ORDER_TOKENS
    if lhs != rhs or lhs == "multiply"
]
SEQUENCE_FEATURES += [
    f"{name}_x_tile2k_masked"
    for name in tuple(SEQUENCE_FEATURES)
    if name.startswith("first_op_") or name.startswith("bigram_")
]
SEQUENCE_FEATURES += [
    f"op_{token}_x_wide_allocation"
    for token in (
        "add", "subtract", "multiply", "divide", "maximum", "sigmoid",
        "exp", "log", "rsqrt", "reduce_sum", "max", "where",
        "broadcast_to", "greater",
    )
]
SEQUENCE_FEATURES += [
    f"{name}_x_wide_allocation"
    for name in (
        "two_input_elementwise_count",
        "one_input_elementwise_count",
        "transcendental_count",
        "reduction_count",
        "has_transcendental",
        "has_reduction",
        "arithmetic_only",
        "two_reduction_interaction",
    )
]

MASKED_ATOMIC_TOKENS = ("maximum", "multiply", "sigmoid")
ATOMIC_FEATURES = [
    feature
    for token in MASKED_ATOMIC_TOKENS
    for feature in (
        f"op_{token}_x_compute_mask",
        f"op_{token}_x_free_x_compute_mask",
        f"op_{token}_x_sqrt_free_x_compute_mask",
        f"op_{token}_x_log2_free_x_compute_mask",
    )
]

# Retained as an explicit inventory for diagnostics; these are deliberately not
# fitted into the p128 compute-masked atomic applicability domain.
ATOMIC_SOURCE_FEATURE_INVENTORY = [
    feature
    for token in (
        "add", "subtract", "multiply", "divide", "maximum", "sigmoid",
        "exp", "log", "rsqrt", "reduce_sum", "max", "where",
        "broadcast_to", "greater",
    )
    for feature in (
        f"op_{token}", f"op_{token}_x_free", f"op_{token}_x_mask",
        f"op_{token}_x_free_x_mask", f"op_{token}_x_allocation_free",
        f"op_{token}_x_compute_mask", f"op_{token}_x_free_x_compute_mask",
        f"op_{token}_x_one_input_arity",
        f"op_{token}_x_free_x_one_input_arity",
        f"op_{token}_x_two_input_arity",
        f"op_{token}_x_free_x_two_input_arity",
    )
]

# Aggregate Explorer counters minus an independently measured runtime baseline
# cannot resolve sub-10 ns residuals: the p128 FP32 baseline range across the
# control sizes is about 2.9 ns.  Treat <=10 ns as inactive (and audit false
# activations separately) instead of reporting arbitrarily large MAPE on noise.
PAYLOAD_RESOLUTION_NS = 10.0


def _cases(roots: list[Path], baselines: dict) -> list[dict]:
    rows = []
    for case in sorted(case for root in roots for case in root.glob("control_*")):
        summary = case / "hardware/explorer_summary.json"
        trace = case / "dependency_trace.jsonl"
        if not summary.is_file() or not trace.is_file():
            continue
        events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        regions = {
            json.dumps(event["region_ir"], sort_keys=True): event["region_ir"]
            for event in events if event.get("region_ir") is not None
        }
        if not regions:
            built = build_region_ir(events)
            regions = {json.dumps(built, sort_keys=True): built}
        if len(regions) != 1:
            continue
        region = next(iter(regions.values()))
        region_events = [
            event for event in events
            if event.get("op") in {"compute", "reduce_sum"}
        ]
        features = compositional_features(region)
        dtype = str(region["dtype"])
        partition = int(region.get("partition_count") or 1)
        profile = next(iter(json.loads(summary.read_text()).values()), {})
        match = re.match(r"control_(.*?)__p", case.name)
        family = match.group(1) if match else case.name
        geometry = re.search(r"__p(\d+)__f(\d+)__n(\d+)__", case.name)
        is_sequence = family.startswith("sequence_") or len(region_events) > 1
        for engine in ("vector", "scalar", "gpsimd"):
            baseline = runtime_engine_baseline_ns(baselines, dtype, partition, engine)
            active = float(profile.get(f"{engine}_engine_active_time", 0.0)) * 1e9
            rows.append({
                "case": case.name,
                "family": family,
                "is_sequence": is_sequence,
                "domain": case.parent.name,
                "engine": engine,
                "dtype": dtype,
                "logical_free_dim": int(region.get("logical_free_dim") or 1),
                "chain": int(geometry.group(3)) if geometry else 1,
                "features": features,
                "payload_ns": max(0.0, active - baseline),
                "baseline_ns": baseline,
                "active_ns": active,
            })
    return rows


def _fit(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.linalg.lstsq(
        np.asarray(
            [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
            dtype=float,
        ),
        np.asarray([row["payload_ns"] for row in rows], dtype=float),
        rcond=None,
    )[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-output", type=Path, required=True)
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    rows = _cases(args.roots, baselines)
    folds = []
    for engine in ("vector", "scalar", "gpsimd"):
        for dtype in sorted({row["dtype"] for row in rows if row["engine"] == engine}):
            selected = [
                row for row in rows
                if row["engine"] == engine and row["dtype"] == dtype
            ]
            sequence_families = sorted(
                {row["family"] for row in selected if row["is_sequence"]}
            )
            for family in sequence_families:
                train = [
                    row for row in selected
                    if row["is_sequence"] and row["family"] != family
                ]
                test = [row for row in selected if row["is_sequence"] and row["family"] == family]
                coefficients = _fit(train, SEQUENCE_FEATURES)
                errors = []
                zero_payload_violations = 0
                for row in test:
                    vector = [float(row["features"].get(name, 0.0)) for name in SEQUENCE_FEATURES]
                    payload = max(0.0, float(np.dot(coefficients, vector)))
                    if row["payload_ns"] > PAYLOAD_RESOLUTION_NS:
                        errors.append(
                            abs(payload - row["payload_ns"])
                            / row["payload_ns"] * 100
                        )
                    elif payload > PAYLOAD_RESOLUTION_NS:
                        zero_payload_violations += 1
                folds.append({
                    "engine": engine,
                    "dtype": dtype,
                    "held_family": family,
                    "samples": len(test),
                    "positive_payload_samples": len(errors),
                    "zero_payload_violations": zero_payload_violations,
                    "mape_pct": (
                        statistics.mean(errors)
                        if errors else (100.0 if zero_payload_violations else 0.0)
                    ),
                })
    atomic_folds = []
    for engine in ("vector", "scalar"):
        selected = [
            row for row in rows
            if row["engine"] == engine and not row["is_sequence"]
            and row["features"].get("has_compute_mask", 0.0) > 0
        ]
        for free_dim in sorted({row["logical_free_dim"] for row in selected}):
            train = [row for row in selected if row["logical_free_dim"] != free_dim]
            test = [row for row in selected if row["logical_free_dim"] == free_dim]
            if not train or not test:
                continue
            coefficients = _fit(train, ATOMIC_FEATURES)
            errors = []
            zero_payload_violations = 0
            for row in test:
                vector = [float(row["features"].get(name, 0.0)) for name in ATOMIC_FEATURES]
                payload = max(0.0, float(np.dot(coefficients, vector)))
                if row["payload_ns"] > PAYLOAD_RESOLUTION_NS:
                    errors.append(
                        abs(payload - row["payload_ns"])
                        / row["payload_ns"] * 100
                    )
                elif payload > PAYLOAD_RESOLUTION_NS:
                    zero_payload_violations += 1
            atomic_folds.append({
                "engine": engine,
                "held_logical_free_dim": free_dim,
                "samples": len(test),
                "positive_payload_samples": len(errors),
                "zero_payload_violations": zero_payload_violations,
                "mape_pct": (
                    statistics.mean(errors)
                    if errors else (100.0 if zero_payload_violations else 0.0)
                ),
            })
    report = {
        "schema": "triton-viz.source-sequence-cv-v1",
        "protocol": "leave-one-synthetic-sequence-family-out; payload-to-payload MAPE",
        "folds": folds,
        "vector_mape_pct": statistics.mean(row["mape_pct"] for row in folds if row["engine"] == "vector"),
        "scalar_mape_pct": statistics.mean(row["mape_pct"] for row in folds if row["engine"] == "scalar"),
        "gpsimd_mape_pct": statistics.mean(row["mape_pct"] for row in folds if row["engine"] == "gpsimd"),
        "max_fold_mape_pct": max(row["mape_pct"] for row in folds),
        "atomic_protocol": (
            "leave-one-logical-free-dimension-out within the independent "
            "p128 compute-masked atomic applicability domain; payload-to-payload MAPE"
        ),
        "atomic_folds": atomic_folds,
        "atomic_vector_mape_pct": statistics.mean(
            row["mape_pct"] for row in atomic_folds if row["engine"] == "vector"
        ),
        "atomic_scalar_mape_pct": statistics.mean(
            row["mape_pct"] for row in atomic_folds if row["engine"] == "scalar"
        ),
        "atomic_max_fold_mape_pct": max(row["mape_pct"] for row in atomic_folds),
    }
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    output_rows = []
    for engine in ("vector", "scalar", "gpsimd"):
        for dtype in sorted({row["dtype"] for row in rows if row["engine"] == engine}):
            selected = [row for row in rows if row["engine"] == engine and row["dtype"] == dtype]
            sequence_rows = [row for row in selected if row["is_sequence"]]
            atomic_rows = [
                row for row in selected
                if not row["is_sequence"]
                and row["features"].get("has_compute_mask", 0.0) > 0
            ]
            if not sequence_rows or (engine != "gpsimd" and not atomic_rows):
                continue
            sequence_coefficients = _fit(sequence_rows, SEQUENCE_FEATURES)
            combined: dict[str, float] = {}
            if atomic_rows:
                atomic_coefficients = _fit(atomic_rows, ATOMIC_FEATURES)
                for feature, value in zip(ATOMIC_FEATURES, atomic_coefficients):
                    combined[feature] = combined.get(feature, 0.0) + float(value)
                    multi_feature = f"{feature}_x_multi"
                    combined[multi_feature] = combined.get(multi_feature, 0.0) - float(value)
            for feature, value in zip(SEQUENCE_FEATURES, sequence_coefficients):
                multi_feature = f"{feature}_x_multi"
                combined[multi_feature] = combined.get(multi_feature, 0.0) + float(value)
            output_rows.extend(
                {"engine": engine, "dtype": dtype, "target": "fixed_ns", "feature": feature, "coefficient": value}
                for feature, value in sorted(combined.items())
            )
            for partition in (1, 16, 128):
                output_rows.append({
                    "engine": engine,
                    "dtype": dtype,
                    "target": "runtime_baseline_ns",
                    "feature": f"partition_p{partition}",
                    "coefficient": runtime_engine_baseline_ns(baselines, dtype, partition, engine),
                })
            output_rows.append({
                "engine": engine, "dtype": dtype, "target": "effective_count",
                "feature": "intercept", "coefficient": 1e-9,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["engine", "dtype", "target", "feature", "coefficient"])
        writer.writeheader(); writer.writerows(output_rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
