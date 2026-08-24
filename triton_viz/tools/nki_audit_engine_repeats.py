"""Audit repeatability of aggregate engine payload labels on full controls."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
    runtime_engine_baseline_ns,
)

ENGINES = ("vector", "scalar", "gpsimd")


def _rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "control_results.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return {
            str(row["case"]): row
            for row in csv.DictReader(file)
            if row.get("case") and not row.get("error")
        }


def audit_repeats(
    roots: list[Path],
    baselines: dict,
    *,
    payload_resolution_ns: float,
    stable_coverage_min: float,
    relative_mad_max: float,
) -> dict:
    if len(roots) < 3:
        raise ValueError("repeat audit requires at least three independent roots")
    documents = [_rows(root) for root in roots]
    case_sets = [set(document) for document in documents]
    if any(cases != case_sets[0] for cases in case_sets[1:]):
        missing = {
            root.name: sorted(case_sets[0] - cases)
            for root, cases in zip(roots[1:], case_sets[1:])
            if cases != case_sets[0]
        }
        raise ValueError(f"repeat roots do not contain identical case sets: {missing}")

    cases = []
    engine_stable = {engine: 0 for engine in ENGINES}
    engine_positive_relative_mads = {engine: [] for engine in ENGINES}
    for case in sorted(case_sets[0]):
        reference = documents[0][case]
        dtype = str(reference.get("dtype") or "float32")
        partition = int(reference.get("p") or 1)
        record = {"case": case, "engines": {}}
        for engine in ENGINES:
            active = [
                float(document[case][f"{engine}_active_ns"])
                for document in documents
            ]
            baseline = runtime_engine_baseline_ns(
                baselines, dtype, partition, engine
            )
            payloads = [max(0.0, value - baseline) for value in active]
            classifications = [
                value > payload_resolution_ns for value in payloads
            ]
            stable = len(set(classifications)) == 1
            engine_stable[engine] += int(stable)
            median_payload = statistics.median(payloads)
            mad = statistics.median(
                abs(value - median_payload) for value in payloads
            )
            relative_mad = (
                mad / median_payload
                if median_payload > payload_resolution_ns
                else None
            )
            if relative_mad is not None:
                engine_positive_relative_mads[engine].append(relative_mad)
            record["engines"][engine] = {
                "active_ns": active,
                "runtime_baseline_ns": baseline,
                "payload_ns": payloads,
                "active_classifications": classifications,
                "active_frequency": sum(classifications) / len(classifications),
                "stable_activation": stable,
                "median_payload_ns": median_payload,
                "payload_mad_ns": mad,
                "relative_mad": relative_mad,
            }
        cases.append(record)

    total = len(cases)
    summary = {}
    for engine in ENGINES:
        relative_mads = engine_positive_relative_mads[engine]
        stable_coverage = engine_stable[engine] / max(1, total)
        summary[engine] = {
            "cases": total,
            "stable_activation_cases": engine_stable[engine],
            "stable_activation_coverage": stable_coverage,
            "positive_median_cases": len(relative_mads),
            "mean_positive_relative_mad": (
                statistics.mean(relative_mads) if relative_mads else None
            ),
            "worst_positive_relative_mad": (
                max(relative_mads) if relative_mads else None
            ),
            "pass": (
                stable_coverage >= stable_coverage_min
                and bool(relative_mads)
                and max(relative_mads) <= relative_mad_max
            ),
        }
    return {
        "schema": "triton-viz.aggregate-engine-repeat-audit-v1",
        "protocol": (
            "full-domain independent aggregate profiles; median/MAD; "
            "control-derived runtime baseline"
        ),
        "roots": [str(root) for root in roots],
        "payload_resolution_ns": payload_resolution_ns,
        "gates": {
            "stable_activation_coverage_min": stable_coverage_min,
            "worst_positive_relative_mad_max": relative_mad_max,
        },
        "summary": summary,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--payload-resolution-ns", type=float, default=10.0)
    parser.add_argument("--stable-coverage-min", type=float, default=0.95)
    parser.add_argument("--relative-mad-max", type=float, default=0.10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit_repeats(
        args.roots,
        load_runtime_engine_baselines(args.runtime_overhead_results),
        payload_resolution_ns=args.payload_resolution_ns,
        stable_coverage_min=args.stable_coverage_min,
        relative_mad_max=args.relative_mad_max,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if all(item["pass"] for item in report["summary"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
