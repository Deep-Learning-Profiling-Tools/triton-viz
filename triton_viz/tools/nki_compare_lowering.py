"""Compare mapped source-region lowering across two compiler artifact roots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from triton_viz.tools.nki_fit_structured_controls import EXCLUDED_RUNTIME_OPCODES
from triton_viz.tools.nki_provenance import compare_fingerprints
from triton_viz.tools.nki_region_ir import match_structural_family
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature

DIFF_FIELDS = [
    "case",
    "fusion_group",
    "engine",
    "rule_id_reference",
    "rule_id_candidate",
    "structural_key_reference",
    "structural_key_candidate",
    "instruction_count_reference",
    "instruction_count_candidate",
    "opcode_fingerprint_reference",
    "opcode_fingerprint_candidate",
    "status",
]


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "experiment_manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing experiment manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "compiler_fingerprint" not in manifest:
        raise ValueError(f"Manifest has no compiler_fingerprint: {path}")
    return manifest


def _region_metadata(trace: Path) -> dict[int, dict[str, str]]:
    events = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    _annotate_fusion_signature(events)
    regions: dict[int, dict[str, str]] = {}
    for event in events:
        if event.get("fusion_group_index") != 0 or not event.get("region_ir"):
            continue
        group = int(event["fusion_group"])
        region = event["region_ir"]
        match = match_structural_family(region)
        regions[group] = {
            "rule_id": match.rule_id,
            "structural_key": str(region["structural_key"]),
        }
    return regions


def load_lowering(root: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Load payload opcode fingerprints keyed by case, source region and engine."""
    lowering: dict[tuple[str, int, str], dict[str, Any]] = {}
    for trace in sorted(root.glob("*/trace.jsonl")):
        mapping_path = trace.parent / "hardware/source_mapping/instruction_mapping.csv"
        if not mapping_path.is_file():
            continue
        regions = _region_metadata(trace)
        counters: dict[tuple[int, str], Counter[str]] = {}
        with mapping_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                group_text = row.get("fusion_group", "")
                opcode = row.get("opcode", "")
                engine = row.get("engine", "")
                if not group_text or opcode in EXCLUDED_RUNTIME_OPCODES:
                    continue
                group = int(group_text)
                if group not in regions:
                    continue
                counters.setdefault((group, engine), Counter())[opcode] += 1
        for (group, engine), opcodes in counters.items():
            lowering[(trace.parent.name, group, engine)] = {
                **regions[group],
                "instruction_count": sum(opcodes.values()),
                "opcode_fingerprint": dict(sorted(opcodes.items())),
            }
    return lowering


def compare_lowering(
    reference: Mapping[tuple[str, int, str], Mapping[str, Any]],
    candidate: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, group, engine in sorted(set(reference) | set(candidate)):
        ref = reference.get((case, group, engine))
        cand = candidate.get((case, group, engine))
        if ref is None:
            status = "missing_reference"
        elif cand is None:
            status = "missing_candidate"
        elif (
            ref["rule_id"] == cand["rule_id"]
            and ref["instruction_count"] == cand["instruction_count"]
            and ref["opcode_fingerprint"] == cand["opcode_fingerprint"]
        ):
            status = "same_lowering"
        else:
            status = "structural_drift"
        rows.append(
            {
                "case": case,
                "fusion_group": group,
                "engine": engine,
                "rule_id_reference": ref["rule_id"] if ref else "",
                "rule_id_candidate": cand["rule_id"] if cand else "",
                "structural_key_reference": ref["structural_key"] if ref else "",
                "structural_key_candidate": cand["structural_key"] if cand else "",
                "instruction_count_reference": ref["instruction_count"] if ref else "",
                "instruction_count_candidate": cand["instruction_count"]
                if cand
                else "",
                "opcode_fingerprint_reference": json.dumps(
                    ref["opcode_fingerprint"] if ref else {}, sort_keys=True
                ),
                "opcode_fingerprint_candidate": json.dumps(
                    cand["opcode_fingerprint"] if cand else {}, sort_keys=True
                ),
                "status": status,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    reference_manifest = _load_manifest(args.reference)
    candidate_manifest = _load_manifest(args.candidate)
    provenance = compare_fingerprints(
        reference_manifest["compiler_fingerprint"],
        candidate_manifest["compiler_fingerprint"],
    )
    rows = compare_lowering(
        load_lowering(args.reference), load_lowering(args.candidate)
    )
    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "comparison_schema_version": 1,
        "provenance": provenance,
        "region_engine_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lowering_diff.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=DIFF_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "lowering_diff_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Compared {len(rows)} region-engine fingerprints: {dict(status_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
