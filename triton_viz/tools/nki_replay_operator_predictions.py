"""Replay cost-model predictions on saved operator traces without recompiling hardware."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_cost_model import (
    ComputeCalibration,
    CostModel,
    DmaAffineCalibration,
    NcLatencyCalibration,
    StructuralStaticDmaCalibration,
    StructuredControlCalibration,
    eliminate_redundant_hbm_loads,
    simulate,
)

FIELDS = [
    "case",
    "op",
    "rows",
    "cols",
    "dtype",
    "compiler_elided_load_count",
    "compiler_elided_load_bytes",
    "predicted_dynamic_dma_us",
    "predicted_static_dma_us",
    "predicted_total_dma_us",
    "predicted_vector_us",
    "predicted_scalar_us",
    "predicted_total_us",
    "hardware_nc_p50_us",
    "nc_error_pct",
    "hardware_total_dma_us",
    "dma_error_pct",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dma-affine-read-csv", type=Path, required=True)
    parser.add_argument("--dma-affine-write-csv", type=Path, required=True)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--structured-control-csv", type=Path, required=True)
    parser.add_argument("--structural-static-dma-csv", type=Path, required=True)
    parser.add_argument("--nc-latency-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_rows = list(csv.DictReader((args.root / "operator_results.csv").open()))
    out = []
    models: dict[str, CostModel] = {}
    for source in source_rows:
        if source.get("status") != "ok" or not source.get("hardware_dma_active_us"):
            continue
        dtype = source["dtype"]
        if dtype not in models:
            models[dtype] = CostModel(
                dma_affine_calibration=DmaAffineCalibration.from_csvs(
                    args.dma_affine_read_csv, args.dma_affine_write_csv, dtype
                ),
                compute_calibration=ComputeCalibration.from_csv(
                    args.compute_calibration_csv
                ),
                structured_control_lowering=StructuredControlCalibration.from_csv(
                    args.structured_control_csv
                ),
                structural_static_dma=StructuralStaticDmaCalibration.from_csv(
                    args.structural_static_dma_csv
                ),
                nc_latency_calibration=(
                    NcLatencyCalibration.from_csv(args.nc_latency_csv)
                    if args.nc_latency_csv
                    else None
                ),
            )
        case_name = (
            f"{source['op']}__r{source['rows']}__c{source['cols']}__{source['dtype']}"
        )
        trace = args.root / case_name / "trace.jsonl"
        events = [json.loads(line) for line in trace.read_text().splitlines() if line]
        model_events, cse = eliminate_redundant_hbm_loads(events)
        result = simulate(model_events, models[dtype])
        dynamic_us = result.engine_busy_ns.get("dma", 0.0) / 1000.0
        static_us = result.engine_busy_ns.get("static_dma", 0.0) / 1000.0
        predicted_us = dynamic_us + static_us
        hardware_us = float(source["hardware_dma_active_us"])
        predicted_total_us = result.predicted_latency_ns / 1000.0
        hardware_nc_us = float(source["hardware_nc_p50_us"])
        out.append(
            {
                "case": case_name,
                "op": source["op"],
                "rows": source["rows"],
                "cols": source["cols"],
                "dtype": dtype,
                "compiler_elided_load_count": cse["eliminated_load_count"],
                "compiler_elided_load_bytes": cse["eliminated_load_bytes"],
                "predicted_dynamic_dma_us": dynamic_us,
                "predicted_static_dma_us": static_us,
                "predicted_total_dma_us": predicted_us,
                "predicted_vector_us": result.engine_busy_ns.get("vector", 0.0)
                / 1000.0,
                "predicted_scalar_us": result.engine_busy_ns.get("scalar", 0.0)
                / 1000.0,
                "predicted_total_us": predicted_total_us,
                "hardware_nc_p50_us": hardware_nc_us,
                "nc_error_pct": (predicted_total_us - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "hardware_total_dma_us": hardware_us,
                "dma_error_pct": (predicted_us - hardware_us) / hardware_us * 100,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    mape = statistics.mean(abs(float(row["dma_error_pct"])) for row in out)
    print(f"Replayed {len(out)} cases; combined DMA MAPE={mape:.6f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
