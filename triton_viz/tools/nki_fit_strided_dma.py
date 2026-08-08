"""Fit stride-aware DMA busy and completion residual from microbench controls."""

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
    "completion_residual_ns",
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
        nc_us = float(result["latency_percentiles"]["nc_latency"]["p50_us"])
        dma_ns = float(profile["dma_active_time"]) * 1e9
        compute_ns = max(
            float(profile.get("vector_engine_active_time", 0)),
            float(profile.get("scalar_engine_active_time", 0)),
            float(profile.get("tensor_engine_active_time", 0)),
        ) * 1e9
        spec = result["spec"]
        rows.append(
            {
                "dtype": spec["dtype"],
                "stride_items": spec["stride"],
                "partition_count": spec["p"],
                "free_dim": spec["f"],
                "dma_active_ns": dma_ns,
                "completion_residual_ns": max(0.0, nc_us * 1000 - dma_ns - compute_ns),
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
