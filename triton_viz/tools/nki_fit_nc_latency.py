"""Fit kernel dispatch residuals from independent structural controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from triton_viz.tools.nki_region_ir import structural_calibration_key

FIELDS = ["calibration_key", "dtype", "free_dim", "residual_ns", "case", "compiler_version"]


def collect(root: Path) -> list[dict]:
    results = list(csv.DictReader((root / "control_results.csv").open()))
    rows = []
    for result in results:
        if not result.get("hardware_nc_p50_us"):
            continue
        case = root / result["case"]
        events = [json.loads(line) for line in (case / "trace.jsonl").read_text().splitlines() if line]
        region = next(event["region_ir"] for event in events if event.get("region_ir"))
        summary = json.loads((case / "hardware/explorer_summary.json").read_text())
        profile = next(iter(summary.values()))
        dma_ns = float(profile["dma_active_time"]) * 1e9
        compute_ns = max(float(result["vector_active_ns"]), float(result["scalar_active_ns"]))
        residual = float(result["hardware_nc_p50_us"]) * 1000 - dma_ns - compute_ns
        rows.append({
            "calibration_key": structural_calibration_key(region),
            "dtype": region["dtype"],
            "free_dim": region.get("logical_free_dim") or region["free_dim"],
            "residual_ns": max(0.0, residual),
            "case": result["case"],
            "compiler_version": profile.get("compiler_version", ""),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = collect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} NC residual points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
