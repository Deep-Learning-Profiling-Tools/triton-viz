"""Fit Level-A fusion expansion rows from Tilebench Explorer artifacts.

For every operator case directory, the source trace supplies the fusion
signature/shape/dtype and Explorer supplies actual per-engine instruction count
and active time.  Dividing active time by the Level-B single-instruction cost
produces an effective (possibly fractional) expansion count used by the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from triton_viz.tools.nki_cost_model import ComputeCalibration
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature


FIELDS = [
    "operator", "fusion_signature", "dtype", "free_dim", "engine",
    "input_stream_count", "hardware_instruction_count", "hardware_active_ns",
    "level_b_instruction_ns", "effective_instruction_count", "source_region_id",
    "opcode_fingerprint", "mapping_payload_coverage_percent", "compiler_version", "case_dir",
    "kernel_control_active_ns",
]


def _model(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    models = [value for value in data.values() if isinstance(value, dict)]
    if len(models) != 1:
        raise ValueError(f"Expected one Explorer model in {summary_path}, got {len(models)}")
    return models[0]


def fit(root: Path, level_b: ComputeCalibration, case_glob: str = "*") -> list[dict]:
    rows: list[dict] = []
    for trace_path in sorted(root.glob(f"{case_glob}/trace.jsonl")):
        case_dir = trace_path.parent
        summary_path = case_dir / "hardware" / "explorer_summary.json"
        if not summary_path.is_file():
            continue
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        _annotate_fusion_signature(events)
        groups: dict[int, list[dict]] = {}
        for event in events:
            if event.get("fusion_group") is not None:
                groups.setdefault(int(event["fusion_group"]), []).append(event)
        profile = _model(summary_path)
        mapping_dir = case_dir / "hardware" / "source_mapping"
        audit_path, mapping_path = mapping_dir / "audit.json", mapping_dir / "instruction_mapping.csv"
        mapping_rows = list(csv.DictReader(mapping_path.open(encoding="utf-8", newline=""))) if mapping_path.is_file() else []
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
        compiler_version = str(profile.get("compiler_version") or "")
        for group_id, members in sorted(groups.items()):
            # Multi-region cases require auditable instruction mapping. The
            # legacy whole-kernel path remains only for one-region artifacts.
            if len(groups) > 1 and not audit:
                continue
            signature = str(members[0]["fusion_signature"])
            signatures = [signature, f"pattern:{members[0]['fusion_pattern']}"]
            free_dim = max(int((event.get("input_shape") or event.get("output_shape") or [0])[-1])
                           for event in members)
            dtype = str(next((event.get("output_dtype") for event in members
                              if event.get("output_dtype")), "float32"))
            region_id = str(members[0].get("source_region_id") or "")
            for engine, streams in (("vector", 2), ("scalar", 1)):
                selected = [row for row in mapping_rows
                            if row.get("engine") == engine and row.get("fusion_group") == str(group_id)]
                if audit:
                    engine_audit = audit["engines"].get(engine, {})
                    active_ns = float(engine_audit.get("regions", {}).get(str(group_id), 0))
                    count = len(selected)
                    coverage = float(engine_audit.get("mapped_payload_coverage_percent", 0))
                    opcodes: dict[str, int] = {}
                    for item in selected:
                        opcode = str(item.get("opcode") or "unknown")
                        opcodes[opcode] = opcodes.get(opcode, 0) + 1
                    fingerprint = ";".join(f"{key}:{opcodes[key]}" for key in sorted(opcodes))
                    fixed_control_ns = (
                        max(0.0, float(engine_audit.get("explorer_active_ns", 0))
                            - float(engine_audit.get("mapped_active_ns", 0)))
                        if group_id == min(groups) else 0.0
                    )
                else:
                    active_ns = float(profile.get(f"{engine}_engine_active_time") or 0) * 1e9
                    count = int(profile.get(f"{engine}_engine_instruction_count") or 0)
                    coverage, fingerprint, fixed_control_ns = 100.0, "summary-only", 0.0
                one_ns = level_b.instruction_ns(engine, dtype, streams, free_dim)
                if not active_ns or not one_ns:
                    continue
                base_row = {
                    "operator": case_dir.name.split("__", 1)[0],
                    "dtype": dtype,
                    "free_dim": free_dim,
                    "engine": engine,
                    "input_stream_count": streams,
                    "hardware_instruction_count": int(count),
                    "hardware_active_ns": active_ns,
                    "level_b_instruction_ns": one_ns,
                    "effective_instruction_count": active_ns / one_ns,
                    "source_region_id": region_id,
                    "opcode_fingerprint": fingerprint,
                    "mapping_payload_coverage_percent": coverage,
                    "compiler_version": compiler_version,
                    "case_dir": str(case_dir),
                    "kernel_control_active_ns": fixed_control_ns,
                }
                for calibration_signature in signatures:
                    rows.append({**base_row, "fusion_signature": calibration_signature})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-glob", default="*", help="Case-directory glob used for calibration/holdout splits")
    args = parser.parse_args(argv)
    rows = fit(args.root, ComputeCalibration.from_csv(args.compute_calibration_csv), args.case_glob)
    if not rows:
        raise SystemExit("No unambiguous lowering calibration cases found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} lowering rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
