"""Export the lowering-rule catalog and Region IR coverage from trace artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from triton_viz.tools.nki_region_ir import grammar_catalog, match_structural_family
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature

REGION_FIELDS = [
    "case",
    "trace",
    "fusion_group",
    "source_region_id",
    "structural_key",
    "schema_version",
    "family",
    "rule_id",
    "rule_evidence",
    "ood_reasons",
    "in_scope",
]


def _trace_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_file():
            paths.add(item)
        elif item.is_dir():
            paths.update(item.rglob("trace.jsonl"))
    return sorted(paths)


def collect_region_coverage(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    """Return one auditable row per source region in the supplied traces."""
    rows: list[dict[str, Any]] = []
    for trace in _trace_paths(inputs):
        events = [
            json.loads(line) for line in trace.read_text().splitlines() if line.strip()
        ]
        if not all(
            "region_ir" in event
            for event in events
            if event.get("engine") in {"vector", "scalar"}
        ):
            _annotate_fusion_signature(events)

        leaders = [
            event
            for event in events
            if event.get("region_ir") and event.get("fusion_group_index") == 0
        ]
        for event in leaders:
            region = event["region_ir"]
            match = match_structural_family(region)
            rows.append(
                {
                    "case": trace.parent.name,
                    "trace": str(trace),
                    "fusion_group": event.get("fusion_group"),
                    "source_region_id": event.get("source_region_id", ""),
                    "structural_key": region.get("structural_key", ""),
                    "schema_version": region.get("schema_version", 1),
                    "family": match.family,
                    "rule_id": match.rule_id,
                    "rule_evidence": ";".join(match.evidence),
                    "ood_reasons": ";".join(match.ood_reasons),
                    "in_scope": not match.ood_reasons,
                }
            )
    return rows


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize coverage without dropping OOD regions from the denominator."""
    rule_counts = Counter(str(row["rule_id"]) for row in rows)
    ood_counts = Counter(
        reason
        for row in rows
        for reason in str(row["ood_reasons"]).split(";")
        if reason
    )
    in_scope = sum(bool(row["in_scope"]) for row in rows)
    catalog = grammar_catalog()
    for rule in catalog["rules"]:
        observed_cases = sorted(
            {
                str(row["case"])
                for row in rows
                if row["rule_id"] == rule["rule_id"] and bool(row["in_scope"])
            }
        )
        rule["observed_cases"] = observed_cases
        rule["observed_region_count"] = sum(
            row["rule_id"] == rule["rule_id"] and bool(row["in_scope"]) for row in rows
        )
        rule["has_observed_evidence"] = bool(observed_cases)
    return {
        **catalog,
        "coverage": {
            "region_count": len(rows),
            "in_scope_region_count": in_scope,
            "ood_region_count": len(rows) - in_scope,
            "in_scope_percent": 100.0 * in_scope / len(rows) if rows else 0.0,
            "rule_counts": dict(sorted(rule_counts.items())),
            "ood_reason_counts": dict(sorted(ood_counts.items())),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Trace files or roots containing trace.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = collect_region_coverage(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "region_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=REGION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "grammar_report.json").write_text(
        json.dumps(build_report(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote grammar catalog and coverage for {len(rows)} regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
