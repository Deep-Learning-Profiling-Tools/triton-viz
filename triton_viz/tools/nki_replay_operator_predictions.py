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
    "predicted_gpsimd_us",
    "predicted_tensor_us",
    "predicted_total_us",
    "predicted_compute_only_us",
    "predicted_compute_dma_us",
    "predicted_resource_overlap_us",
    "predicted_without_completion_floor_us",
    "compute_only_error_pct",
    "compute_dma_error_pct",
    "resource_overlap_error_pct",
    "without_completion_floor_error_pct",
    "hardware_nc_p50_us",
    "nc_error_pct",
    "hardware_tensor_active_us",
    "tensor_error_pct",
    "hardware_vector_active_us",
    "hardware_vector_payload_us",
    "vector_payload_error_pct",
    "vector_error_pct",
    "hardware_scalar_active_us",
    "hardware_scalar_payload_us",
    "scalar_payload_error_pct",
    "scalar_error_pct",
    "hardware_gpsimd_active_us",
    "hardware_gpsimd_payload_us",
    "gpsimd_payload_error_pct",
    "gpsimd_error_pct",
    "hardware_dynamic_dma_us",
    "hardware_dynamic_dma_reference",
    "dynamic_dma_error_pct",
    "hardware_static_dma_us",
    "static_dma_error_pct",
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
    "completion_exact_count",
    "completion_interpolated_count",
    "completion_ood_count",
    "completion_excluded_partition_count",
    "completion_floor_activated",
    "micro_dag_vector_covered",
    "micro_dag_scalar_covered",
    "micro_dag_gpsimd_covered",
    "micro_dag_tensor_covered",
    "micro_dag_static_dma_covered",
    "micro_dag_unsupported_engine_events",
    "micro_dag_timing_exact_count",
    "micro_dag_timing_interpolated_count",
    "micro_dag_timing_aggregate_count",
    "micro_dag_source_region_count",
    "micro_dag_exact_region_count",
    "micro_dag_all_regions_covered",
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
    parser.add_argument(
        "--disable-structured-completion-floor",
        action="store_true",
        help="Ablation: report final latency without the structured completion floor.",
    )
    parser.add_argument(
        "--completion-exclude-free-dim",
        type=int,
        action="append",
        default=[],
        help="Leave-one-F-out completion audit; repeat for multiple free dimensions.",
    )
    parser.add_argument(
        "--completion-exclude-partition",
        type=int,
        action="append",
        default=[],
        help="Leave-one-partition-out completion audit.",
    )
    parser.add_argument(
        "--completion-exclude-calibration-key",
        action="append",
        default=[],
        help="Leave-one-grammar-control-out completion audit.",
    )
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
                enable_structured_completion_floor=(
                    not args.disable_structured_completion_floor
                ),
                completion_excluded_free_dims=frozenset(
                    args.completion_exclude_free_dim
                ),
                completion_excluded_partition_counts=frozenset(
                    args.completion_exclude_partition
                ),
                completion_excluded_calibration_keys=frozenset(
                    args.completion_exclude_calibration_key
                ),
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
        predicted_vector_us = result.engine_busy_ns.get("vector", 0.0) / 1000.0
        predicted_scalar_us = result.engine_busy_ns.get("scalar", 0.0) / 1000.0
        predicted_gpsimd_us = result.engine_busy_ns.get("gpsimd", 0.0) / 1000.0
        hardware_nc_us = float(source["hardware_nc_p50_us"])
        hardware_tensor_us = float(source.get("hardware_tensor_active_us") or 0.0)
        hardware_vector_us = float(source.get("hardware_vector_active_us") or 0.0)
        hardware_scalar_us = float(source.get("hardware_scalar_active_us") or 0.0)
        summary_path = args.root / case_name / "hardware/explorer_summary.json"
        profile = {}
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            profile = next(iter(summary.values()), {})
        hardware_gpsimd_us = float(
            profile.get("gpsimd_engine_active_time") or 0.0
        ) * 1e6
        counter_dynamic_dma_us = (
            float(profile.get("software_dynamic_dma_active_time") or 0.0)
            + float(profile.get("hardware_dynamic_dma_active_time") or 0.0)
        ) * 1e6
        hardware_static_dma_us = float(
            profile.get("static_dma_active_time") or 0.0
        ) * 1e6
        # Explorer exposes both explicit dynamic counters and the aggregate
        # DMA decomposition. Use the larger physically reconciled value:
        # older/compiler-specific profiles can under-report one of the dynamic
        # counter families, while dma_active - static_dma is the corresponding
        # aggregate residual. Preserve the choice in the CSV audit.
        residual_dynamic_dma_us = max(0.0, hardware_us - hardware_static_dma_us)
        hardware_dynamic_dma_us = max(
            counter_dynamic_dma_us, residual_dynamic_dma_us
        )
        hardware_dynamic_dma_reference = (
            "explicit_dynamic_counters"
            if counter_dynamic_dma_us >= residual_dynamic_dma_us
            else "dma_active_minus_static"
        )
        mapping_audit_path = (
            args.root / case_name / "hardware/source_mapping/audit.json"
        )
        mapping_audit = (
            json.loads(mapping_audit_path.read_text(encoding="utf-8"))
            if mapping_audit_path.is_file()
            else {}
        )

        def payload_us(engine: str) -> float:
            return (
                float(
                    (mapping_audit.get("engines", {}).get(engine, {})).get(
                        "payload_active_ns", 0.0
                    )
                )
                / 1000.0
            )

        hardware_vector_payload_us = payload_us("vector")
        hardware_scalar_payload_us = payload_us("scalar")
        hardware_gpsimd_payload_us = payload_us("gpsimd")
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
                "predicted_vector_us": predicted_vector_us,
                "predicted_scalar_us": predicted_scalar_us,
                "predicted_gpsimd_us": predicted_gpsimd_us,
                "predicted_tensor_us": predicted_tensor_us,
                "predicted_total_us": predicted_total_us,
                "predicted_compute_only_us": components["compute_only"] / 1000.0,
                "predicted_compute_dma_us": components["compute_plus_dma"] / 1000.0,
                "predicted_resource_overlap_us": components["resource_overlap_makespan"]
                / 1000.0,
                "predicted_without_completion_floor_us": components[
                    "without_structured_completion_floor"
                ]
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
                "without_completion_floor_error_pct": (
                    components["without_structured_completion_floor"] / 1000.0
                    - hardware_nc_us
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
                "hardware_vector_active_us": hardware_vector_us,
                "hardware_vector_payload_us": hardware_vector_payload_us,
                "vector_payload_error_pct": (
                    (predicted_vector_us - hardware_vector_payload_us)
                    / hardware_vector_payload_us
                    * 100
                    if hardware_vector_payload_us
                    else ""
                ),
                "vector_error_pct": (
                    (predicted_vector_us - hardware_vector_us)
                    / hardware_vector_us
                    * 100
                    if hardware_vector_us
                    else ""
                ),
                "hardware_scalar_active_us": hardware_scalar_us,
                "hardware_scalar_payload_us": hardware_scalar_payload_us,
                "scalar_payload_error_pct": (
                    (predicted_scalar_us - hardware_scalar_payload_us)
                    / hardware_scalar_payload_us
                    * 100
                    if hardware_scalar_payload_us
                    else ""
                ),
                "scalar_error_pct": (
                    (predicted_scalar_us - hardware_scalar_us)
                    / hardware_scalar_us
                    * 100
                    if hardware_scalar_us
                    else ""
                ),
                "hardware_gpsimd_active_us": hardware_gpsimd_us,
                "hardware_gpsimd_payload_us": hardware_gpsimd_payload_us,
                "gpsimd_payload_error_pct": (
                    (predicted_gpsimd_us - hardware_gpsimd_payload_us)
                    / hardware_gpsimd_payload_us
                    * 100
                    if hardware_gpsimd_payload_us
                    else ""
                ),
                "gpsimd_error_pct": (
                    (predicted_gpsimd_us - hardware_gpsimd_us)
                    / hardware_gpsimd_us
                    * 100
                    if hardware_gpsimd_us
                    else ""
                ),
                "hardware_dynamic_dma_us": hardware_dynamic_dma_us,
                "hardware_dynamic_dma_reference": hardware_dynamic_dma_reference,
                "dynamic_dma_error_pct": (
                    (dynamic_us - hardware_dynamic_dma_us)
                    / hardware_dynamic_dma_us
                    * 100
                    if hardware_dynamic_dma_us
                    else ""
                ),
                "hardware_static_dma_us": hardware_static_dma_us,
                "static_dma_error_pct": (
                    (static_us - hardware_static_dma_us)
                    / hardware_static_dma_us
                    * 100
                    if hardware_static_dma_us
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
                "completion_exact_count": int(
                    components.get("completion_exact_count", 0)
                ),
                "completion_interpolated_count": int(
                    components.get("completion_interpolated_count", 0)
                ),
                "completion_ood_count": int(
                    components.get("completion_ood_count", 0)
                ),
                "completion_excluded_partition_count": int(
                    components.get("completion_excluded_partition_count", 0)
                ),
                "completion_floor_activated": int(
                    components.get("structured_completion_floor_activated", 0)
                ),
                "micro_dag_vector_covered": int(
                    components.get("micro_dag_vector_covered", 0)
                ),
                "micro_dag_scalar_covered": int(
                    components.get("micro_dag_scalar_covered", 0)
                ),
                "micro_dag_gpsimd_covered": int(
                    components.get("micro_dag_gpsimd_covered", 0)
                ),
                "micro_dag_tensor_covered": int(
                    components.get("micro_dag_tensor_covered", 0)
                ),
                "micro_dag_static_dma_covered": int(
                    components.get("micro_dag_static_dma_covered", 0)
                ),
                "micro_dag_unsupported_engine_events": int(
                    components.get("micro_dag_unsupported_engine_events", 0)
                ),
                "micro_dag_timing_exact_count": int(
                    components.get("micro_dag_timing_exact_count", 0)
                ),
                "micro_dag_timing_interpolated_count": int(
                    components.get("micro_dag_timing_interpolated_count", 0)
                ),
                "micro_dag_timing_aggregate_count": int(
                    components.get("micro_dag_timing_aggregate_count", 0)
                ),
                "micro_dag_source_region_count": int(
                    components.get("micro_dag_source_region_count", 0)
                ),
                "micro_dag_exact_region_count": int(
                    components.get("micro_dag_exact_region_count", 0)
                ),
                "micro_dag_all_regions_covered": int(
                    components.get("micro_dag_all_regions_covered", 0)
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
