"""Export structural-family Level-A points from mapped control artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_cost_model import ComputeCalibration
from triton_viz.tools.nki_region_ir import (
    match_structural_family,
    structural_calibration_key,
)
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature

FIELDS = [
    "family",
    "calibration_key",
    "rule_id",
    "rule_evidence",
    "ood_reasons",
    "engine",
    "dtype",
    "free_dim",
    "effective_count",
    "instruction_count",
    "fixed_ns",
    "case",
    "compiler_version",
    "opcode_fingerprint",
    "replicate_count",
    "effective_count_variance",
    "fixed_ns_variance",
]
EXCLUDED_RUNTIME_OPCODES = {
    "DRAIN",
    "NOTIFY",
    "EVENT_SEMAPHORE",
    "EVENT_SEMAPHORE_RANGE_CLEAR",
    "SET_ORDERING_MODE",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compiler_version(case: Path) -> str:
    summary_path = case / "hardware/explorer_summary.json"
    if not summary_path.is_file():
        return ""
    summary = _load_json(summary_path)
    profile = next(iter(summary.values()), {})
    return str(profile.get("compiler_version", ""))


def _opcode_fingerprint(rows: list[dict]) -> str:
    payload = ";".join(
        f"{row.get('engine','')}:{row.get('opcode','')}:{row.get('source_opcode','')}"
        for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def collect(
    roots: list[Path],
    level_b: ComputeCalibration,
    include_prefixes: tuple[str, ...] = (),
) -> list[dict]:
    """Collect audited Level-A points; incomplete cases remain excluded explicitly."""
    rows: list[dict] = []
    traces = sorted(trace for root in roots for trace in root.glob("*/trace.jsonl"))
    for trace in traces:
        case = trace.parent
        audit_path = case / "hardware/source_mapping/audit.json"
        mapping_path = case / "hardware/source_mapping/instruction_mapping.csv"
        if include_prefixes and not case.name.startswith(include_prefixes):
            continue
        if not audit_path.is_file() or not mapping_path.is_file():
            continue

        events = [
            json.loads(line) for line in trace.read_text().splitlines() if line.strip()
        ]
        _annotate_fusion_signature(events)
        groups: dict[int, list[dict]] = {}
        for event in events:
            if event.get("fusion_group") is not None:
                groups.setdefault(int(event["fusion_group"]), []).append(event)

        audit = _load_json(audit_path)
        with mapping_path.open(encoding="utf-8", newline="") as file:
            mappings = list(csv.DictReader(file))
        compiler_version = _compiler_version(case)

        for group, members in groups.items():
            region = members[0]["region_ir"]
            match = match_structural_family(region)
            free_dim = int(region.get("logical_free_dim") or region["free_dim"])
            for engine, streams in (("vector", 2), ("scalar", 1)):
                engine_audit = audit["engines"][engine]
                active_ns = float(engine_audit["regions"].get(str(group), 0))
                instruction_ns = level_b.instruction_ns(
                    engine, region["dtype"], streams, free_dim
                )
                selected = [
                    row
                    for row in mappings
                    if row["engine"] == engine
                    and row["fusion_group"] == str(group)
                    and row["opcode"] not in EXCLUDED_RUNTIME_OPCODES
                ]
                fixed_ns = 0.0
                if group == 0:
                    fixed_ns = max(
                        0.0,
                        float(engine_audit["explorer_active_ns"])
                        - float(engine_audit["mapped_active_ns"]),
                    )
                rows.append(
                    {
                        "family": match.family,
                        "calibration_key": structural_calibration_key(region),
                        "rule_id": match.rule_id,
                        "rule_evidence": ";".join(match.evidence),
                        "ood_reasons": ";".join(match.ood_reasons),
                        "engine": engine,
                        "dtype": region["dtype"],
                        "free_dim": free_dim,
                        "effective_count": active_ns / instruction_ns
                        if instruction_ns
                        else 0,
                        "instruction_count": len(selected),
                        "fixed_ns": fixed_ns,
                        "case": case.name,
                        "compiler_version": compiler_version,
                        "opcode_fingerprint": _opcode_fingerprint(selected),
                        "replicate_count": 1,
                        "effective_count_variance": 0.0,
                        "fixed_ns_variance": 0.0,
                    }
                )
    return rows


def collect_legacy(paths: list[Path]) -> list[dict]:
    """Import mapped softmax points without making its full signature a key."""
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            source_rows = list(csv.DictReader(file))
        for row in source_rows:
            rows.append(
                {
                    "family": "reduction_transcendental",
                    "calibration_key": "",
                    "rule_id": "legacy.softmax",
                    "rule_evidence": "softmax",
                    "ood_reasons": "legacy_region_ir_unavailable",
                    "engine": row["engine"],
                    "dtype": row["dtype"],
                    "free_dim": int(float(row["free_dim"])),
                    "effective_count": float(row["effective_instruction_count"]),
                    "instruction_count": int(float(row["hardware_instruction_count"])),
                    "fixed_ns": 0.0,
                    "case": Path(row.get("case_dir", path.stem)).name,
                    "compiler_version": "legacy-mapped",
                    "opcode_fingerprint": "legacy",
                    "replicate_count": 1,
                    "effective_count_variance": 0.0,
                    "fixed_ns_variance": 0.0,
                }
            )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Median repeated controls and reject incompatible compiler lowerings."""
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["calibration_key"] or row["family"],
            row["engine"],
            row["dtype"],
            int(row["free_dim"]),
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in grouped.items():
        compiler_versions = {row["compiler_version"] for row in group}
        opcode_fingerprints = {row["opcode_fingerprint"] for row in group}
        if len(compiler_versions) > 1 or len(opcode_fingerprints) > 1:
            raise ValueError(
                "Incompatible structured controls for "
                f"{key}: compiler_versions={sorted(compiler_versions)}, "
                f"opcode_fingerprints={sorted(opcode_fingerprints)}"
            )
        effective = [float(row["effective_count"]) for row in group]
        fixed = [float(row["fixed_ns"]) for row in group]
        instructions = [int(row["instruction_count"]) for row in group]
        row = dict(group[0])
        row.update(
            {
                "effective_count": statistics.median(effective),
                "instruction_count": round(statistics.median(instructions)),
                "fixed_ns": statistics.median(fixed),
                "case": ";".join(sorted(str(item["case"]) for item in group)),
                "replicate_count": len(group),
                "effective_count_variance": (
                    statistics.pvariance(effective) if len(effective) > 1 else 0.0
                ),
                "fixed_ns_variance": (
                    statistics.pvariance(fixed) if len(fixed) > 1 else 0.0
                ),
            }
        )
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            row["calibration_key"] or row["family"],
            row["engine"],
            row["dtype"],
            int(row["free_dim"]),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--legacy-level-a-csv", nargs="*", type=Path, default=[])
    parser.add_argument("--include-case-prefix", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    level_b = ComputeCalibration.from_csv(args.compute_calibration_csv)
    rows = collect(args.roots, level_b, tuple(args.include_case_prefix))
    rows.extend(collect_legacy(args.legacy_level_a_csv))
    rows = aggregate_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} structured control points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
