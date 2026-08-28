"""Control CV for source-visible whole-program lowering regimes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from triton_viz.tools.nki_program_context import program_context_features
from triton_viz.tools.nki_region_ir import structural_calibration_key
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature


def source_descriptor_from_events(events: list[dict], dtype: str, case: str = "") -> dict:
    """Build the reusable program key using source-trace fields only."""
    _annotate_fusion_signature(events)
    regions = []
    for event in events:
        if event.get("fusion_group_index") == 0 and event.get("region_ir"):
            regions.append((int(event["fusion_group"]), event["region_ir"]))
    regions.sort(key=lambda item: item[0])
    context = program_context_features(events)
    logical_partition = max(
        (
            int(region.get("logical_active_partition_count") or 1)
            for _, region in regions
        ),
        default=1,
    )
    key = (
        dtype,
        logical_partition,
        tuple(structural_calibration_key(region) for _, region in regions),
        int(context["program_region_count"]),
        int(context["program_dag_join_count"]),
        int(context["program_masked_event_count"] > 0),
    )
    return {
        "case": case,
        "key": key,
        "distance_feature": float(context["program_total_transfer_bytes"]),
    }


def source_descriptor(root: Path, case: str, dtype: str) -> dict:
    trace = root / case / "trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines() if line]
    return source_descriptor_from_events(events, dtype, case)


def _source_sample(root: Path, row: dict[str, str]) -> dict:
    case = f"{row['op']}__r{row['rows']}__c{row['cols']}__{row['dtype']}"
    descriptor = source_descriptor(root, case, str(row["dtype"]))
    summary = json.loads(
        (root / case / "hardware/explorer_summary.json").read_text(encoding="utf-8")
    )
    profile = next(iter(summary.values()), {})
    return {
        **descriptor,
        "column": int(row["cols"]),
        "completion_ns": float(row["hardware_nc_p50_us"]) * 1000.0,
        "actual": {
            engine: float(profile.get(f"{engine}_engine_active_time") or 0.0)
            * 1e6
            for engine in ("vector", "scalar", "gpsimd")
        },
    }


def _wape(pairs: list[tuple[float, float]]) -> float:
    denominator = sum(actual for _, actual in pairs)
    return sum(abs(predicted - actual) for predicted, actual in pairs) / denominator * 100.0


def _mape(pairs: list[tuple[float, float]]) -> float:
    return sum(abs(predicted - actual) / actual for predicted, actual in pairs) / len(pairs) * 100.0


def evaluate(root: Path) -> dict:
    with (root / "operator_results.csv").open(encoding="utf-8", newline="") as file:
        samples = [
            _source_sample(root, row)
            for row in csv.DictReader(file)
            if row.get("status") == "ok"
        ]
    folds = []
    for held_column in sorted({sample["column"] for sample in samples}):
        train = [sample for sample in samples if sample["column"] != held_column]
        test = [sample for sample in samples if sample["column"] == held_column]
        pairs = {engine: [] for engine in ("vector", "scalar", "gpsimd")}
        completion_pairs = []
        ood = []
        for sample in test:
            candidates = [item for item in train if item["key"] == sample["key"]]
            if not candidates:
                ood.append(sample["case"])
                continue
            nearest_distance = min(
                abs(item["distance_feature"] - sample["distance_feature"])
                for item in candidates
            )
            nearest = [
                item
                for item in candidates
                if abs(item["distance_feature"] - sample["distance_feature"])
                == nearest_distance
            ]
            for engine in pairs:
                prediction = sum(item["actual"][engine] for item in nearest) / len(nearest)
                pairs[engine].append((prediction, sample["actual"][engine]))
            completion_pairs.append(
                (
                    sum(item["completion_ns"] for item in nearest) / len(nearest),
                    sample["completion_ns"],
                )
            )
        folds.append(
            {
                "held_column": held_column,
                "cases": len(test),
                "covered_cases": len(test) - len(ood),
                "ood_cases": ood,
                "completion_mape_pct": _mape(completion_pairs) if completion_pairs else None,
                **{
                    f"{engine}_wape_pct": _wape(values) if values else None
                    for engine, values in pairs.items()
                },
            }
        )
    mean_completion_mape = sum(
        fold["completion_mape_pct"] for fold in folds
    ) / len(folds)
    coverage_pct = sum(fold["covered_cases"] for fold in folds) / sum(
        fold["cases"] for fold in folds
    ) * 100.0
    return {
        "schema": "triton-viz.whole-program-regime-control-cv-v1",
        "protocol": "leave-one-free-dimension-out; nearest source transfer geometry within exact reusable program grammar",
        "folds": folds,
        "mean_wape_pct": {
            engine: sum(fold[f"{engine}_wape_pct"] for fold in folds) / len(folds)
            for engine in ("vector", "scalar", "gpsimd")
        },
        "mean_completion_mape_pct": mean_completion_mape,
        "completion_gate_pct": 20.0,
        "completion_gate_pass": coverage_pct == 100.0 and mean_completion_mape < 20.0,
        "coverage_pct": coverage_pct,
        "target_postcompile_prediction_reads": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in whole-program regime CV")
    report = evaluate(args.control_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
