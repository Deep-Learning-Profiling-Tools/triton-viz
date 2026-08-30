"""Fit stride-aware DMA busy time from independent microbench controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "dtype",
    "stride_items",
    "partition_count",
    "free_dim",
    "dma_active_ns",
    "dynamic_dma_active_ns",
    "static_dma_active_ns",
    "case",
    "compiler_version",
]


def collect(results_jsonl: Path) -> list[dict]:
    rows = []
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        if result.get("status") != "ok" or result.get("spec", {}).get("kind") != "dma_strided_store":
            continue
        case = Path(result["dir"])
        profile = next(iter(json.loads((case / "explorer_summary.json").read_text()).values()))
        dma_ns = float(profile["dma_active_time"]) * 1e9
        static_dma_ns = float(profile.get("static_dma_active_time") or 0.0) * 1e9
        dynamic_dma_ns = max(
            (
                float(profile.get("software_dynamic_dma_active_time") or 0.0)
                + float(profile.get("hardware_dynamic_dma_active_time") or 0.0)
            )
            * 1e9,
            dma_ns - static_dma_ns,
        )
        spec = result["spec"]
        rows.append(
            {
                "dtype": spec["dtype"],
                "stride_items": spec["stride"],
                "partition_count": spec["p"],
                "free_dim": spec["f"],
                "dma_active_ns": dma_ns,
                "dynamic_dma_active_ns": dynamic_dma_ns,
                "static_dma_active_ns": static_dma_ns,
                "case": case.name,
                "compiler_version": profile.get("compiler_version", ""),
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = collect(args.results_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} strided DMA points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
