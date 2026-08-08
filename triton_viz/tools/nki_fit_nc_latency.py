"""Fit kernel dispatch residuals from independent structural controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from triton_viz.tools.nki_cost_model import (
    ComputeCalibration,
    CostModel,
    DmaAffineCalibration,
    StructuralStaticDmaCalibration,
    StructuredControlCalibration,
    simulate,
)
from triton_viz.tools.nki_region_ir import structural_calibration_key

FIELDS = [
    "calibration_key",
    "dtype",
    "free_dim",
    "residual_ns",
    "scheduler_makespan_ns",
    "case",
    "compiler_version",
]


def collect(root: Path, models: dict[str, CostModel]) -> list[dict]:
    results = list(csv.DictReader((root / "control_results.csv").open()))
    rows = []
    for result in results:
        if not result.get("hardware_nc_p50_us"):
            continue
        case = root / result["case"]
        dependency_trace = case / "dependency_trace.jsonl"
        if not dependency_trace.is_file():
            raise FileNotFoundError(
                f"missing runtime dependency trace for {result['case']}: "
                f"{dependency_trace}"
            )
        events = [
            json.loads(line)
            for line in dependency_trace.read_text().splitlines()
            if line
        ]
        region = next(event["region_ir"] for event in events if event.get("region_ir"))
        summary = json.loads((case / "hardware/explorer_summary.json").read_text())
        profile = next(iter(summary.values()))
        dtype = ComputeCalibration._norm_dtype(result.get("dtype"))
        scheduler_ns = simulate(events, models[dtype]).components_ns[
            "resource_overlap_makespan"
        ]
        residual = float(result["hardware_nc_p50_us"]) * 1000 - scheduler_ns
        rows.append({
            "calibration_key": structural_calibration_key(region),
            "dtype": dtype,
            "free_dim": region.get("logical_free_dim") or region["free_dim"],
            "residual_ns": max(0.0, residual),
            "scheduler_makespan_ns": scheduler_ns,
            "case": result["case"],
            "compiler_version": profile.get("compiler_version", ""),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dma-affine-read-csv", type=Path, required=True)
    parser.add_argument("--dma-affine-write-csv", type=Path, required=True)
    parser.add_argument("--dma-affine-write-bf16-csv", type=Path)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--structured-control-csv", type=Path, required=True)
    parser.add_argument("--structural-static-dma-csv", type=Path)
    args = parser.parse_args(argv)
    compute = ComputeCalibration.from_csv(args.compute_calibration_csv)
    structured = StructuredControlCalibration.from_csv(
        args.structured_control_csv
    )
    static_dma = (
        StructuralStaticDmaCalibration.from_csv(args.structural_static_dma_csv)
        if args.structural_static_dma_csv
        else None
    )
    models = {}
    for dtype in ("float32", "bfloat16"):
        write_csv = (
            args.dma_affine_write_bf16_csv
            if dtype == "bfloat16" and args.dma_affine_write_bf16_csv
            else args.dma_affine_write_csv
        )
        models[dtype] = CostModel(
            dma_affine_calibration=DmaAffineCalibration.from_csvs(
                args.dma_affine_read_csv, write_csv, dtype
            ),
            compute_calibration=compute,
            structured_control_lowering=structured,
            structural_static_dma=static_dma,
        )
    rows = [row for root in args.roots for row in collect(root, models)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} NC residual points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
