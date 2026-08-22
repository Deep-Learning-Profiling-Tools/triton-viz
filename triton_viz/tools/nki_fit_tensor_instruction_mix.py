"""Fit mixed REGULAR/TRANSPOSE Tensor busy time from independent controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq

from triton_viz.tools.nki_tensor_instruction_mix import (
    INSTRUCTION_COLUMNS, tensor_mix_features,
)


def _samples(roots: list[Path]) -> list[dict]:
    samples = []
    for root in roots:
        for path in root.glob("*/explorer_parquet/Instruction.parquet"):
            summary_path = path.parent.parent / "explorer_summary.json"
            if not summary_path.is_file():
                continue
            profile = next(iter(json.loads(summary_path.read_text()).values()))
            target = float(profile.get("tensor_engine_active_time") or 0) * 1e9
            rows = pq.read_table(path, columns=INSTRUCTION_COLUMNS).to_pylist()
            features = tensor_mix_features(rows)
            if target > 0 and features[0] > 0 and features[1] > 0:
                samples.append({
                    "case": path.parent.parent.name,
                    "group": root.parent.name,
                    "features": features,
                    "target_ns": target,
                })
    return samples


def _model(samples: list[dict], neighbors: int) -> dict:
    vectors = [sample["features"] for sample in samples]
    scales = [statistics.pstdev(column) or 1.0 for column in zip(*vectors)]
    return {"vectors": vectors, "scales": scales, "neighbors": min(neighbors, len(samples))}


def _predict(model: dict, sample: dict, train: list[dict]) -> float:
    distances = sorted(
        (
            sum(abs((a - b) / scale) for a, b, scale in zip(sample["features"], row["features"], model["scales"])),
            row["target_ns"],
        )
        for row in train
    )[: model["neighbors"]]
    if distances[0][0] == 0:
        return statistics.fmean(target for distance, target in distances if distance == 0)
    weights = [1.0 / distance for distance, _target in distances]
    return sum(weight * target for weight, (_distance, target) in zip(weights, distances)) / sum(weights)


def fit(roots: list[Path], output: Path, audit_output: Path, neighbors: int = 2) -> int:
    samples = _samples(roots)
    if not samples:
        raise ValueError("No mixed Tensor instruction controls")
    model = _model(samples, neighbors)
    artifact = {
        "schema_version": 1, "feature_count": 8, "neighbors": neighbors,
        "vectors": model["vectors"],
        "targets_ns": [sample["target_ns"] for sample in samples],
        "feature_scales": model["scales"],
    }
    predictions = []
    for group in sorted({sample["group"] for sample in samples}):
        train = [sample for sample in samples if sample["group"] != group]
        for sample in (item for item in samples if item["group"] == group):
            predicted = _predict(_model(train, neighbors), sample, train)
            predictions.append({
                "case": sample["case"], "group": group,
                "absolute_percentage_error": abs(predicted - sample["target_ns"]) / sample["target_ns"] * 100,
            })
    audit = {
        "schema_version": 1, "method": "leave_one_control_suite_out_refit",
        "control_count": len(samples), "neighbors": neighbors,
        "aggregate_mape_pct": statistics.fmean(row["absolute_percentage_error"] for row in predictions),
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(samples)} mixed Tensor controls; suite CV MAPE {audit['aggregate_mape_pct']:.6f}%")
    return len(samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--neighbors", type=int, default=2)
    args = parser.parse_args(argv)
    fit(args.roots, args.output, args.audit_output, args.neighbors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
