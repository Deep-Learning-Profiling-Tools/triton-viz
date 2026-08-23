"""Audit repeatability of aggregate-only GpSimd payload control labels."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_fit_source_sequence_lowering import PAYLOAD_RESOLUTION_NS, _cases
from triton_viz.tools.nki_fit_structured_controls import load_runtime_engine_baselines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    selected = (4034, 4035, 4044, 4045, 4052, 4053,
                5020, 5021, 5028, 5029, 5038, 5039, 5052, 5053)
    by_seed = {seed: [] for seed in selected}
    for repeat in (1, 2, 3):
        roots = [
            args.root / f"rep{repeat}_phase2",
            args.root / f"rep{repeat}_phase3",
        ]
        rows = [row for row in _cases(roots, baselines) if row["engine"] == "gpsimd"]
        for seed in selected:
            matches = [row for row in rows if row["chain"] == seed]
            if len(matches) != 1:
                raise ValueError(f"repeat {repeat} seed {seed}: found {len(matches)} rows")
            by_seed[seed].append(float(matches[0]["payload_ns"]))
    cases = []
    for seed, values in by_seed.items():
        active = [value > PAYLOAD_RESOLUTION_NS for value in values]
        cases.append({
            "seed": seed, "payload_ns": values,
            "median_payload_ns": statistics.median(values),
            "min_payload_ns": min(values), "max_payload_ns": max(values),
            "span_ns": max(values) - min(values),
            "active_classifications": active,
            "active_classification_stable": len(set(active)) == 1,
        })
    report = {
        "schema": "triton-viz.gpsimd-aggregate-repeat-audit-v1",
        "protocol": "three new independent aggregate-only measurements; no refit",
        "payload_resolution_ns": PAYLOAD_RESOLUTION_NS,
        "cases": cases,
        "stable_active_classification_cases": sum(
            case["active_classification_stable"] for case in cases
        ),
        "total_cases": len(cases),
        "max_repeat_span_ns": max(case["span_ns"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
