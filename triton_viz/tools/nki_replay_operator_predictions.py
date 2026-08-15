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
    DmaCalibrationSurface,
    RuntimeOverheadCalibration,
    StructuralStaticDmaCalibration,
    StridedDmaCalibration,
    StructuredControlCalibration,
    TensorCalibrationSurface,
    _canonical_engine,
    _compute_value_dtype,
    _expand_lowering_groups,
    _free_dim,
    _input_stream_count,
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
    "predicted_tensor_us",
    "predicted_total_us",
    "predicted_compute_only_us",
    "predicted_compute_dma_us",
    "predicted_resource_overlap_us",
    "compute_only_error_pct",
    "compute_dma_error_pct",
    "resource_overlap_error_pct",
    "hardware_nc_p50_us",
    "nc_error_pct",
    "hardware_tensor_active_us",
    "tensor_error_pct",
    "hardware_total_dma_us",
    "dma_error_pct",
    "calibration_match",
    "dma_calibration_path",
    "dma_surface_match",
    "dma_surface_exact_count",
    "dma_surface_interpolated_count",
    "dma_surface_ood_count",
    "dma_surface_max_log_distance",
    "tensor_flops_domain_ood_count",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dma-read-surface-csv", type=Path, required=True)
    parser.add_argument("--dma-write-surface-csv", type=Path, required=True)
    parser.add_argument("--dma-transpose-surface-csv", type=Path)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--structured-control-csv", type=Path, required=True)
    parser.add_argument("--tensor-calibration-csv", type=Path)
    parser.add_argument("--structural-static-dma-csv", type=Path, required=True)
    parser.add_argument("--runtime-overhead-csv", type=Path)
    parser.add_argument("--strided-dma-csv", type=Path)
    parser.add_argument("--strict-calibration", action="store_true")
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
                dma_calibration=(
                    DmaCalibrationSurface.from_csv(
                        args.dma_read_surface_csv,
                        bandwidth_column="derived.read_gbps_dynamic_dma_active",
                        dtype_name=dtype,
                        duplicate_policy="median",
                    )
                    if args.dma_read_surface_csv else None
                ),
                dma_write_calibration=(
                    DmaCalibrationSurface.from_csv(
                        args.dma_write_surface_csv,
                        "dma_write_partition_surface",
                        "derived.write_gbps_dynamic_dma_active",
                        dtype,
                        required_repeat=16,
                    )
                    if args.dma_write_surface_csv else None
                ),
                dma_transpose_calibration=(
                    DmaCalibrationSurface.from_csv(
                        args.dma_transpose_surface_csv,
                        "dma_transpose_surface",
                        "derived.read_gbps_dynamic_dma_active",
                        dtype,
                    )
                    if args.dma_transpose_surface_csv else None
                ),
                compute_calibration=ComputeCalibration.from_csv(
                    args.compute_calibration_csv
                ),
                structured_control_lowering=StructuredControlCalibration.from_csv(
                    args.structured_control_csv
                ),
                tensor_calibration=(
                    TensorCalibrationSurface.from_csv(
                        args.tensor_calibration_csv,
                        benchmark_name="tensor_matmul_tiled",
                    )
                    if args.tensor_calibration_csv
                    else None
                ),
                structural_static_dma=StructuralStaticDmaCalibration.from_csv(
                    args.structural_static_dma_csv
                ),
                runtime_overhead_calibration=(
                    RuntimeOverheadCalibration.from_csv(args.runtime_overhead_csv)
                    if args.runtime_overhead_csv
                    else None
                ),
                strided_dma_calibration=(
                    StridedDmaCalibration.from_csv(args.strided_dma_csv)
                    if args.strided_dma_csv
                    else None
                ),
                strict_calibration=args.strict_calibration,
            )
        case_name = (
            f"{source['op']}__r{source['rows']}__c{source['cols']}__{source['dtype']}"
        )
        trace = args.root / case_name / "trace.jsonl"
        events = [json.loads(line) for line in trace.read_text().splitlines() if line]
        model_events, cse = eliminate_redundant_hbm_loads(events)
        result = simulate(model_events, models[dtype])
        # Audit the lowered compute events that CostModel actually prices.
        # Raw source primitives may have been replaced by structured lowering.
        matches = set()
        audit_events = _expand_lowering_groups(model_events, models[dtype])
        for event in audit_events:
            if event.get("op") not in {"binary", "compute", "reduce_sum"}:
                continue
            engine = _canonical_engine(
                str(event.get("engine") or ""), str(event.get("op") or "")
            )
            streams = _input_stream_count(event)
            free_dim = _free_dim(event)
            if free_dim is None:
                matches.add("missing_geometry")
                continue
            _, match = models[dtype].compute_calibration.instruction_lookup(
                engine,
                _compute_value_dtype(event),
                streams,
                free_dim,
                strict_dtype=args.strict_calibration,
            )
            matches.add(match)
        dynamic_us = result.engine_busy_ns.get("dma", 0.0) / 1000.0
        static_us = result.engine_busy_ns.get("static_dma", 0.0) / 1000.0
        predicted_us = dynamic_us + static_us
        hardware_us = float(source["hardware_dma_active_us"])
        predicted_total_us = result.predicted_latency_ns / 1000.0
        predicted_tensor_us = result.engine_busy_ns.get("tensor", 0.0) / 1000.0
        hardware_nc_us = float(source["hardware_nc_p50_us"])
        hardware_tensor_us = float(source.get("hardware_tensor_active_us") or 0.0)
        components = result.components_ns
        has_tensor_events = any(
            event.get("op") in {"dot", "tensor_transpose"}
            for event in model_events
        )
        dma_paths = set()
        dma_matches = set()
        for event in model_events:
            if event.get("op") not in {"load", "store", "transfer"}:
                continue
            lookup = models[dtype].dma_lookup(event)
            dma_paths.add(str(lookup["path"]))
            dma_matches.add(str(lookup["match"]))
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
                "predicted_tensor_us": predicted_tensor_us,
                "predicted_total_us": predicted_total_us,
                "predicted_compute_only_us": components["compute_only"] / 1000.0,
                "predicted_compute_dma_us": components["compute_plus_dma"] / 1000.0,
                "predicted_resource_overlap_us": components["resource_overlap_makespan"]
                / 1000.0,
                "compute_only_error_pct": (components["compute_only"] / 1000.0 - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "compute_dma_error_pct": (components["compute_plus_dma"] / 1000.0 - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "resource_overlap_error_pct": (
                    components["resource_overlap_makespan"] / 1000.0 - hardware_nc_us
                )
                / hardware_nc_us
                * 100,
                "hardware_nc_p50_us": hardware_nc_us,
                "nc_error_pct": (predicted_total_us - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "hardware_tensor_active_us": (
                    hardware_tensor_us if has_tensor_events else ""
                ),
                "tensor_error_pct": (
                    (predicted_tensor_us - hardware_tensor_us)
                    / hardware_tensor_us
                    * 100
                    if hardware_tensor_us and has_tensor_events
                    else ""
                ),
                "hardware_total_dma_us": hardware_us,
                "dma_error_pct": (predicted_us - hardware_us) / hardware_us * 100,
                "calibration_match": ";".join(sorted(matches)) or "not_applicable",
                "dma_calibration_path": ";".join(sorted(dma_paths))
                or "not_applicable",
                "dma_surface_match": ";".join(sorted(dma_matches))
                or "not_applicable",
                "dma_surface_ood_count": int(
                    components.get("dma_surface_ood_count", 0)
                ),
                "dma_surface_exact_count": int(
                    components.get("dma_surface_exact_count", 0)
                ),
                "dma_surface_interpolated_count": int(
                    components.get("dma_surface_interpolated_count", 0)
                ),
                "dma_surface_max_log_distance": components.get(
                    "dma_surface_max_log_distance", 0.0
                ),
                "tensor_flops_domain_ood_count": int(
                    components.get("tensor_flops_domain_ood_count", 0)
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    mape = statistics.mean(abs(float(row["dma_error_pct"])) for row in out)
    tensor_rows = [row for row in out if row.get("hardware_tensor_active_us")]
    tensor_mape = (
        statistics.mean(abs(float(row["tensor_error_pct"])) for row in tensor_rows)
        if tensor_rows
        else ""
    )
    print(
        f"Replayed {len(out)} cases; combined DMA MAPE={mape:.6f}%; "
        f"TensorE busy MAPE={tensor_mape if tensor_mape != '' else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
