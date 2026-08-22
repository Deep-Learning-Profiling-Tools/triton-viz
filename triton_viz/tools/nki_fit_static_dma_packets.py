"""Fit Static DMA from independent, timing-free compiler packet descriptors."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from triton_viz.tools.nki_static_dma_packets import (
    PACKET_COLUMNS, packet_features, packet_fingerprint,
)


def _hardware(packet: Path) -> Path:
    return packet.parent.parent


def _group(case: Path) -> str:
    name = case.name
    group = name.split("__", 1)[0]
    if name.startswith("dma_strided_store_surface"):
        dtype = re.search(r"dtype(\w+?)__", name)
        partition = re.search(r"__p(\d+)__", name)
        if dtype and partition:
            group = f"strided_{dtype.group(1)}_p{partition.group(1)}"
    return group


def _model(samples: list[dict], threshold: float) -> dict:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["fingerprint"]].append(sample["target_ns"])
    stable = {}
    for fingerprint, targets in grouped.items():
        if len(targets) < 2:
            continue
        errors = []
        for index, target in enumerate(targets):
            others = targets[:index] + targets[index + 1:]
            errors.append(abs(statistics.median(others) - target) / target * 100)
        if statistics.fmean(errors) <= threshold:
            stable[fingerprint] = statistics.median(targets)
    vectors = [sample["vector"] for sample in samples]
    means = [statistics.fmean(column) for column in zip(*vectors)]
    scales = [statistics.pstdev(column) or 1.0 for column in zip(*vectors)]
    return {"stable": stable, "vectors": vectors, "means": means, "scales": scales}


def _predict(model: dict, sample: dict, train: list[dict]) -> tuple[float, str]:
    exact = model["stable"].get(sample["fingerprint"])
    if exact is not None:
        return exact, "stable_exact"
    index = min(
        range(len(train)),
        key=lambda candidate: sum(
            abs((value - other) / scale)
            for value, other, scale in zip(
                sample["vector"], model["vectors"][candidate], model["scales"]
            )
        ),
    )
    return train[index]["target_ns"], "knn1_fallback"


def fit(
    roots: list[Path], output: Path, stability_mape_pct: float = 10.0,
    audit_output: Path | None = None,
) -> int:
    samples = []
    for root in roots:
        paths = list(root.glob("*/hardware/explorer_parquet/DmaPacket.parquet"))
        paths += list(root.glob("*/explorer_parquet/DmaPacket.parquet"))
        for path in paths:
            hardware = _hardware(path)
            summary_path = hardware / "explorer_summary.json"
            if not summary_path.is_file():
                continue
            profile = next(iter(json.loads(summary_path.read_text()).values()))
            target = float(profile.get("static_dma_active_time") or 0) * 1e9
            rows = pq.read_table(path, columns=PACKET_COLUMNS).to_pylist()
            if target > 0 and any(r.get("queue_type") != "software_dynamic" for r in rows):
                case = hardware.parent if hardware.name == "hardware" else hardware
                samples.append({
                    "case": case.name,
                    "group": _group(case),
                    "fingerprint": packet_fingerprint(rows),
                    "vector": packet_features(rows),
                    "target_ns": target,
                })
    if not samples:
        raise ValueError("No Static DMA packet controls")
    model = _model(samples, stability_mape_pct)
    stable = model["stable"]
    vectors = model["vectors"]
    means = model["means"]
    scales = model["scales"]
    artifact = {
        "schema_version": 1,
        "feature_count": 87,
        "stability_mape_pct": stability_mape_pct,
        "stable_exact_ns": stable,
        "vectors": vectors,
        "targets_ns": [sample["target_ns"] for sample in samples],
        "feature_means": means,
        "feature_scales": scales,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    if audit_output is not None:
        predictions = []
        all_errors = []
        folds = {}
        for group in sorted({sample["group"] for sample in samples}):
            train = [sample for sample in samples if sample["group"] != group]
            test = [sample for sample in samples if sample["group"] == group]
            fold_model = _model(train, stability_mape_pct)
            errors = []
            matches = defaultdict(int)
            for sample in test:
                predicted, match = _predict(fold_model, sample, train)
                error = abs(predicted - sample["target_ns"]) / sample["target_ns"] * 100
                errors.append(error)
                all_errors.append(error)
                matches[match] += 1
                predictions.append({
                    "case": sample["case"], "group": group,
                    "absolute_percentage_error": error,
                    "match": match,
                })
            folds[group] = {
                "cases": len(test), "mape_pct": statistics.fmean(errors),
                "matches": dict(matches),
            }
        aggregate = statistics.fmean(all_errors)
        audit = {
            "schema_version": 1,
            "method": "complete_leave_one_grammar_out_refit",
            "control_count": len(samples),
            "grammar_count": len(folds),
            "stability_mape_pct": stability_mape_pct,
            "aggregate_mape_pct": aggregate,
            "folds": folds,
            "predictions": predictions,
        }
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Control leave-one-grammar-out MAPE: {aggregate:.6f}%")
    print(f"Wrote {len(samples)} controls and {len(stable)} stable fingerprints")
    return len(samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stability-mape-pct", type=float, default=10.0)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)
    fit(args.roots, args.output, args.stability_mape_pct, args.audit_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
