"""Fit compiler-generated Static DMA busy time from structural control artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from triton_viz.tools.nki_region_ir import (
    match_structural_family,
    structural_calibration_key,
)

FIELDS = [
    "case",
    "structural_calibration_sequence",
    "structural_rule_sequence",
    "element_bytes",
    "partition_count",
    "logical_free_dim",
    "static_dma_ns",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def collect_case(case: Path) -> dict[str, object] | None:
    trace = case / "trace.jsonl"
    summary = case / "hardware/explorer_summary.json"
    if not trace.is_file() or not summary.is_file():
        return None
    events = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    regions = {
        int(event["fusion_group"]): event["region_ir"]
        for event in events
        if event.get("region_ir") is not None
    }
    transfer = next(
        (
            event
            for event in events
            if event.get("op") in {"load", "store"}
            and int(event.get("active_lanes") or 0) > 0
        ),
        None,
    )
    if not regions or transfer is None:
        return None
    element_bytes = int(transfer["bytes"]) // int(transfer["active_lanes"])
    model = next(iter(json.loads(summary.read_text()).values()))
    return {
        "case": case.name,
        "structural_calibration_sequence": ";".join(
            structural_calibration_key(regions[group]) for group in sorted(regions)
        ),
        "structural_rule_sequence": ";".join(
            match_structural_family(regions[group]).rule_id for group in sorted(regions)
        ),
        "element_bytes": element_bytes,
        "partition_count": max(
            int(region.get("partition_count") or 1) for region in regions.values()
        ),
        "logical_free_dim": max(
            int(region.get("logical_free_dim") or 0) for region in regions.values()
        ),
        "static_dma_ns": float(model.get("static_dma_active_time", 0.0)) * 1e9,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [
        row
        for root in args.roots
        for trace in sorted(root.glob("*/trace.jsonl"))
        if (row := collect_case(trace.parent)) is not None
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} structural Static DMA points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
