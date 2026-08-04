"""Fit an additive structured Level-A model from mapped region controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from triton_viz.tools.nki_cost_model import ComputeCalibration
from triton_viz.tools.nki_region_ir import compositional_features
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature


def _samples(roots: list[Path], level_b: ComputeCalibration):
    samples = []
    traces = sorted(trace for root in roots for trace in root.glob("*/trace.jsonl"))
    for trace in traces:
        case = trace.parent
        audit_path = case / "hardware/source_mapping/audit.json"
        mapping_path = case / "hardware/source_mapping/instruction_mapping.csv"
        if not audit_path.is_file() or not mapping_path.is_file():
            continue
        events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        _annotate_fusion_signature(events)
        groups = {}
        for event in events:
            if event.get("fusion_group") is not None:
                groups.setdefault(int(event["fusion_group"]), []).append(event)
        audit = json.loads(audit_path.read_text())
        mappings = list(csv.DictReader(mapping_path.open()))
        for group, members in groups.items():
            ir = members[0]["region_ir"]
            features = compositional_features(ir)
            for engine, streams in (("vector", 2), ("scalar", 1)):
                active = float(audit["engines"][engine]["regions"].get(str(group), 0))
                one = level_b.instruction_ns(engine, ir["dtype"], streams, ir["logical_free_dim"])
                selected = [row for row in mappings if row["engine"] == engine and row["fusion_group"] == str(group)
                            and row["opcode"] not in {"DRAIN", "NOTIFY", "EVENT_SEMAPHORE",
                                                     "EVENT_SEMAPHORE_RANGE_CLEAR", "SET_ORDERING_MODE"}]
                engine_audit = audit["engines"][engine]
                fixed = max(0.0, float(engine_audit["explorer_active_ns"]) - float(engine_audit["mapped_active_ns"]))
                samples.append({"case": case.name, "engine": engine, "dtype": ir["dtype"], "features": features,
                                "effective_count": active / one if one else 0.0, "instruction_count": len(selected),
                                "fixed_ns": fixed})
    return samples


def _legacy_samples(paths: list[Path]):
    result = []
    for path in paths:
        for row in csv.DictReader(path.open()):
            tokens = str(row["fusion_signature"]).replace("reduce_sum", "reducesum").split("_")
            tokens = ["reduce_sum" if token == "reducesum" else token for token in tokens]
            if tokens[0] == "pattern:reduce":
                continue
            reductions = sum(token in {"reduce", "sum", "reduce_sum", "max", "min", "mean"} for token in tokens)
            trans = sum(token in {"exp", "rsqrt", "sqrt", "log", "tanh", "sigmoid"} for token in tokens)
            ir = {"reduction_count": reductions, "broadcast_edge_count": reductions,
                  "one_input_elementwise_count": trans,
                  "two_input_elementwise_count": max(0, len(tokens) - reductions - trans),
                  "transcendental_count": trans, "free_block_count": 1,
                  "logical_free_dim": int(float(row["free_dim"])), "has_mask_or_tail": False,
                  "op_histogram": {token: tokens.count(token) for token in set(tokens)}}
            result.append({"case": f"legacy:{path.name}", "engine": row["engine"], "dtype": row["dtype"],
                           "features": compositional_features(ir),
                           "effective_count": float(row["effective_instruction_count"]),
                           "instruction_count": int(float(row["hardware_instruction_count"])), "fixed_ns": 0.0})
    return result


def fit(roots: list[Path], level_b: ComputeCalibration, legacy_paths: list[Path] | None = None):
    samples = _samples(roots, level_b)
    samples.extend(_legacy_samples(legacy_paths or []))
    names = sorted({name for sample in samples for name in sample["features"]})
    rows = []
    for engine, dtype in sorted({(s["engine"], s["dtype"]) for s in samples}):
        subset = [s for s in samples if s["engine"] == engine and s["dtype"] == dtype]
        x = np.asarray([[s["features"].get(name, 0.0) for name in names] for s in subset], dtype=float)
        for target in ("effective_count", "instruction_count", "fixed_ns"):
            y = np.asarray([s[target] for s in subset], dtype=float)
            ridge = 1e-6 * np.eye(x.shape[1]); ridge[names.index("intercept"), names.index("intercept")] = 0
            beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
            pred = np.maximum(0, x @ beta)
            mape = float(np.mean(np.abs(pred - y) / np.maximum(1.0, np.abs(y))) * 100)
            for name, value in zip(names, beta):
                rows.append({"engine": engine, "dtype": dtype, "target": target, "feature": name,
                             "coefficient": float(value), "training_points": len(subset), "training_mape_pct": mape})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+"); parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-level-a-csv", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    rows = fit(args.roots, ComputeCalibration.from_csv(args.compute_calibration_csv), args.legacy_level_a_csv)
    if not rows: raise SystemExit("No mapped controls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} coefficients")
    return 0


if __name__ == "__main__": raise SystemExit(main())
