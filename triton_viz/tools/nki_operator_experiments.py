"""Validate the NKI cost model against real Tilebench operators.

Unlike ``nki_model_experiments`` (which builds separate hand-written trace and
hardware kernels), this driver traces and benchmarks the *exact same*
``@nki.jit`` kernel source shipped in Tilebench. That closes Status.md issue 1
(trace vs hardware kernel divergence): the CPU trace and the on-device NEFF are
produced from one Python function.

Flow per case:
  Tilebench impl_nki.<kernel>  (one source function)
    -> .func  -> triton_viz nki (nl.*) trace -> JSONL events -> cost model
    -> GenericKernel -> nki.benchmark -> NEFF/NTFF -> Neuron Explorer summary
    -> one CSV row comparing predicted vs measured latency and per-engine busy.

The same canonical NumPy arrays/scalars feed both paths so the traced and
compiled programs are identical, including weighted rmsnorm/layernorm.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

import triton_viz
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.trace import launches
from triton_viz.core.trace import trace as tv_trace
from triton_viz.tools.nki_cost_model import (
    CompositionalLoweringCalibration,
    ComputeCalibration,
    CostModel,
    DmaAffineCalibration,
    DmaCalibrationSurface,
    LoweringExpansionCalibration,
    RuntimeOverheadCalibration,
    StructuralStaticDmaCalibration,
    StridedDmaCalibration,
    StructuredControlCalibration,
    eliminate_redundant_hbm_loads,
    simulate,
)
from triton_viz.tools.nki_explorer import export_parquet
from triton_viz.tools.nki_instruction_source_mapping import write_case as write_mapping
from triton_viz.tools.nki_model_experiments import (
    _nc_p50,
    _percent_error,
    _profile_summary,
)
from triton_viz.tools.nki_provenance import write_experiment_manifest
from triton_viz.tools.nki_trace_dump import write_jsonl

TILEBENCH_OPS = Path(
    os.environ.get("TILEBENCH_OPS_DIR", "/home/ubuntu/Tilebench/benchmarks/operators")
)

_DTYPES = {
    "float32": np.float32,
    "fp32": np.float32,
    "float16": np.float16,
    "fp16": np.float16,
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "bf16": np.dtype(ml_dtypes.bfloat16),
}


def _randn(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    return np.random.randn(*shape).astype(_DTYPES[dtype])


def _matmul_inputs(m: int, k: int, dtype: str) -> list[Any]:
    """Square-output Tilebench matmul inputs plus frozen compile-time tiles."""
    tile_m, tile_n, tile_k = 128, 512, 128

    def tiles(dim: int, tile: int, preferred: int) -> int:
        return next(
            (count for count in range(preferred, 0, -1) if dim % (tile * count) == 0),
            1,
        )

    return [
        _randn((m, k), dtype),
        _randn((k, m), dtype),
        tiles(m, tile_m, 4),
        tiles(m, tile_n, 2),
        tiles(k, tile_k, 8),
        1,  # One Inf2 NeuronCore; Tilebench's wrapper selects cores by platform.
        False,  # Double-row is only applicable to FP8.
    ]


# Each operator: kernel attribute in impl_nki.py + a builder producing the
# canonical NumPy inputs from (rows, cols, dtype). These are fed identically to
# the trace and the hardware benchmark, so both compile the same program.
OPERATORS: dict[str, dict[str, Any]] = {
    "argmax": {
        "kernel": "argmax_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "dropout": {
        "kernel": "dropout_kernel",
        "inputs": lambda r, c, dt: [
            _randn((r, c), dt),
            (np.random.rand(r, c) > 0.25).astype(_DTYPES[dt]),
            0.25,
        ],
    },
    "fused_activation": {
        "kernel": "fused_activation_kernel",
        "inputs": lambda r, c, dt: [
            _randn((r, c), dt),
            _randn((r, c), dt),
            _randn((r, c), dt),
        ],
    },
    "interleave": {
        "kernel": "interleave_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt), _randn((r, c), dt)],
    },
    "kl_divergence": {
        "kernel": "kl_divergence_kernel",
        "inputs": lambda r, c, dt: [
            _randn((r, c), dt),
            np.abs(_randn((r, c), dt)),
        ],
    },
    "l2_norm": {
        "kernel": "l2_norm_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt), 1e-6],
    },
    "leaky_relu": {
        "kernel": "leaky_relu_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "matrix_copy": {
        "kernel": "matrix_copy_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "matrix_transpose": {
        "kernel": "transpose_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "matmul_fp32_fp16_fp8": {
        "kernel": "matmul_kernel",
        # Interpret rows=M=N and cols=K so the common rows/cols sweep also
        # spans contraction sizes without adding operator-specific CLI flags.
        "inputs": _matmul_inputs,
    },
    "mean_reduction": {
        "kernel": "mean_rowwise_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "reverse_array": {
        "kernel": "reverse_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "softmax": {
        "kernel": "softmax_online_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "swiglu": {
        "kernel": "swiglu_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt), _randn((r, c), dt)],
    },
    "relu": {
        "kernel": "relu_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "mul2": {
        "kernel": "mul2_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "sigmoid": {
        "kernel": "sigmoid_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
    },
    "vector_add": {
        "kernel": "add_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt), _randn((r, c), dt)],
    },
    "rmsnorm": {
        "kernel": "rmsnorm_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt), _randn((1, c), dt), 1e-6],
    },
    "layernorm": {
        "kernel": "layernorm_kernel",
        "inputs": lambda r, c, dt: [
            _randn((r, c), dt),
            _randn((1, c), dt),
            _randn((1, c), dt),
            1e-5,
        ],
    },
}


CSV_COLUMNS = [
    "op",
    "rows",
    "cols",
    "dtype",
    "status",
    "error",
    "event_count",
    "load_count",
    "store_count",
    "compute_count",
    "trace_hbm_read_bytes",
    "trace_hbm_write_bytes",
    "compiler_elided_load_count",
    "compiler_elided_load_bytes",
    "predicted_total_us",
    "predicted_dma_busy_us",
    "predicted_vector_busy_us",
    "predicted_scalar_busy_us",
    "hardware_nc_p50_us",
    "hardware_total_active_us",
    "hardware_dma_active_us",
    "hardware_vector_active_us",
    "hardware_scalar_active_us",
    "error_vs_nc_pct",
    "dma_busy_error_pct",
    "vector_busy_error_pct",
    "scalar_busy_error_pct",
    "vector_mapping_coverage_pct",
    "scalar_mapping_coverage_pct",
]


def _load_kernel(op: str):
    impl = TILEBENCH_OPS / op / "impl_nki.py"
    spec = importlib.util.spec_from_file_location(f"tilebench_{op}", impl)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, OPERATORS[op]["kernel"])


def _trace_events(
    op: str, inputs: list[np.ndarray], jsonl: Path
) -> list[dict[str, Any]]:
    kernel = _load_kernel(op)
    triton_viz.clear()
    traced = tv_trace(client=Tracer(), frontend="nki")(kernel.func)
    traced[(1,)](*inputs)
    return write_jsonl(launches[-1].records, jsonl)


def _hbm_bytes(events: list[dict[str, Any]]) -> tuple[int, int]:
    read = write = 0
    for e in events:
        if e.get("op") == "load":
            read += int(e.get("bytes", 0))
        elif e.get("op") == "store":
            write += int(e.get("bytes", 0))
    return read, write


def _run_hardware(
    op: str,
    inputs: list[np.ndarray],
    artifact_dir: Path,
    warmup: int,
    iters: int,
    kernel=None,
) -> tuple[float | None, dict[str, Any]]:
    from neuronxcc import nki

    kernel = kernel or _load_kernel(op)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(artifact_dir)
        bench = nki.benchmark(
            warmup=warmup,
            iters=iters,
            save_neff_name="file.neff",
            save_trace_name="profile.ntff",
            artifacts_dir="compiler_artifacts",
        )(kernel)
        bench(*inputs)
        summary = _profile_summary(
            artifact_dir / "file.neff",
            artifact_dir / "profile.ntff",
            artifact_dir / "explorer_summary.json",
        )
        return _nc_p50(bench), summary
    finally:
        os.chdir(old)


def run_case(
    op: str,
    rows: int,
    cols: int,
    dtype: str,
    out_dir: Path,
    model: CostModel,
    warmup: int,
    iters: int,
    hardware: bool,
    source_mapping: bool = False,
    compiler_load_cse: bool = False,
) -> dict[str, Any]:
    case_id = f"{op}__r{rows}__c{cols}__{dtype}"
    art = out_dir / case_id
    row: dict[str, Any] = {
        "op": op,
        "rows": rows,
        "cols": cols,
        "dtype": dtype,
        "status": "unknown",
        "error": "",
    }
    try:
        if art.exists():
            shutil.rmtree(art)
        art.mkdir(parents=True, exist_ok=True)
        inputs = OPERATORS[op]["inputs"](rows, cols, dtype)
        events = _trace_events(op, inputs, art / "trace.jsonl")
        read, write = _hbm_bytes(events)
        model_events = events
        cse_audit = {"eliminated_load_count": 0, "eliminated_load_bytes": 0}
        if compiler_load_cse:
            model_events, cse_audit = eliminate_redundant_hbm_loads(events)
        row.update(
            event_count=len(events),
            load_count=sum(e.get("op") == "load" for e in events),
            store_count=sum(e.get("op") == "store" for e in events),
            compute_count=sum(
                e.get("op") in ("compute", "binary", "reduce_sum") for e in events
            ),
            trace_hbm_read_bytes=read,
            trace_hbm_write_bytes=write,
            compiler_elided_load_count=cse_audit["eliminated_load_count"],
            compiler_elided_load_bytes=cse_audit["eliminated_load_bytes"],
        )
        sim = simulate(model_events, model)
        busy = sim.engine_busy_ns
        row["predicted_total_us"] = sim.predicted_latency_ns / 1000.0
        row["predicted_dma_busy_us"] = (
            busy.get("dma", 0.0) + busy.get("static_dma", 0.0)
        ) / 1000.0
        row["predicted_vector_busy_us"] = busy.get("vector", 0.0) / 1000.0
        row["predicted_scalar_busy_us"] = busy.get("scalar", 0.0) / 1000.0
        (art / "prediction.json").write_text(
            json.dumps(sim.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if hardware:
            nc_p50, prof = _run_hardware(op, inputs, art / "hardware", warmup, iters)
            row["hardware_nc_p50_us"] = nc_p50
            row["hardware_total_active_us"] = (
                float(prof.get("total_active_time", 0)) * 1e6
            )
            row["hardware_dma_active_us"] = float(prof.get("dma_active_time", 0)) * 1e6
            row["hardware_vector_active_us"] = (
                float(prof.get("vector_engine_active_time", 0)) * 1e6
            )
            row["hardware_scalar_active_us"] = (
                float(prof.get("scalar_engine_active_time", 0)) * 1e6
            )
            row["error_vs_nc_pct"] = _percent_error(row["predicted_total_us"], nc_p50)
            row["dma_busy_error_pct"] = _percent_error(
                row["predicted_dma_busy_us"], row["hardware_dma_active_us"]
            )
            row["vector_busy_error_pct"] = _percent_error(
                row["predicted_vector_busy_us"], row["hardware_vector_active_us"]
            )
            row["scalar_busy_error_pct"] = _percent_error(
                row["predicted_scalar_busy_us"], row["hardware_scalar_active_us"]
            )
            if source_mapping:
                export_parquet(art / "hardware")
                audit = write_mapping(art)
                row["vector_mapping_coverage_pct"] = audit["engines"]["vector"][
                    "mapped_payload_coverage_percent"
                ]
                row["scalar_mapping_coverage_pct"] = audit["engines"]["scalar"][
                    "mapped_payload_coverage_percent"
                ]
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - preserve every failed case in the CSV
        row["status"] = "error"
        row["error"] = repr(exc)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


def _mape(rows: list[dict[str, Any]], field: str) -> float | str:
    vals = [
        abs(float(r[field]))
        for r in rows
        if r.get("status") == "ok" and r.get(field) not in (None, "")
    ]
    return statistics.mean(vals) if vals else ""


def main(argv: list[str] | None = None) -> int:
    global TILEBENCH_OPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ops", nargs="*", default=["softmax"], choices=sorted(OPERATORS)
    )
    parser.add_argument("--rows", type=int, nargs="*", default=[128])
    parser.add_argument("--cols", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--dtype", default="float32", choices=sorted(_DTYPES))
    parser.add_argument(
        "--tilebench-ops-dir",
        type=Path,
        default=TILEBENCH_OPS,
        help="Tilebench benchmarks/operators directory (or set TILEBENCH_OPS_DIR)",
    )
    parser.add_argument("--dma-calibration-csv", type=Path, default=None)
    parser.add_argument("--dma-write-calibration-csv", type=Path, default=None)
    parser.add_argument(
        "--dma-bandwidth-column",
        default="derived.read_gbps_dma_active",
        help="Measured read bandwidth column; use dynamic-active columns to exclude static DMA.",
    )
    parser.add_argument(
        "--dma-write-bandwidth-column",
        default="derived.write_gbps_dma_active",
        help="Measured write bandwidth column.",
    )
    parser.add_argument(
        "--compiler-load-cse",
        action="store_true",
        help="Model exact repeated-HBM-load elimination and record eliminated bytes.",
    )
    parser.add_argument(
        "--structural-static-dma-csv",
        type=Path,
        default=None,
        help="Control-derived compiler Static DMA busy-time calibration.",
    )
    parser.add_argument("--dma-affine-read-csv", type=Path, default=None)
    parser.add_argument("--dma-affine-write-csv", type=Path, default=None)
    parser.add_argument("--kernel-overhead-us", type=float, default=0.0)
    parser.add_argument("--runtime-overhead-csv", type=Path, default=None)
    parser.add_argument("--strided-dma-csv", type=Path, default=None)
    parser.add_argument("--dma-queue-count", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-mapping",
        action="store_true",
        help="Export Explorer parquet and map payload instructions to source regions.",
    )
    parser.add_argument(
        "--compute-calibration-csv",
        type=Path,
        default=None,
        help="Level-B per-instruction VectorE/ScalarE cost surface.",
    )
    parser.add_argument(
        "--lowering-calibration-csv",
        type=Path,
        default=None,
        help="Level-A fusion-signature to per-engine expansion surface.",
    )
    parser.add_argument(
        "--compositional-lowering-csv",
        type=Path,
        default=None,
        help="Structured additive Level-A coefficients (no operator signatures).",
    )
    parser.add_argument(
        "--structured-control-csv",
        type=Path,
        default=None,
        help="Structural-family control points (no operator signatures).",
    )
    args = parser.parse_args(argv)
    if args.source_mapping and args.no_hardware:
        parser.error("--source-mapping requires hardware; remove --no-hardware")
    TILEBENCH_OPS = args.tilebench_ops_dir.resolve()
    if not TILEBENCH_OPS.is_dir():
        parser.error(f"Tilebench operators directory does not exist: {TILEBENCH_OPS}")

    calib = (
        DmaCalibrationSurface.from_csv(
            args.dma_calibration_csv,
            bandwidth_column=args.dma_bandwidth_column,
            dtype_name=args.dtype,
        )
        if args.dma_calibration_csv
        else None
    )
    write_calib = (
        DmaCalibrationSurface.from_csv(
            args.dma_write_calibration_csv,
            "dma_write_partition_surface",
            args.dma_write_bandwidth_column,
        )
        if args.dma_write_calibration_csv
        else None
    )
    compute_calib = (
        ComputeCalibration.from_csv(args.compute_calibration_csv)
        if args.compute_calibration_csv
        else None
    )
    lowering_calib = (
        LoweringExpansionCalibration.from_csv(args.lowering_calibration_csv)
        if args.lowering_calibration_csv
        else None
    )
    compositional = (
        CompositionalLoweringCalibration.from_csv(args.compositional_lowering_csv)
        if args.compositional_lowering_csv
        else None
    )
    structured_controls = (
        StructuredControlCalibration.from_csv(args.structured_control_csv)
        if args.structured_control_csv
        else None
    )
    structural_static_dma = (
        StructuralStaticDmaCalibration.from_csv(args.structural_static_dma_csv)
        if args.structural_static_dma_csv
        else None
    )
    runtime_overhead = (
        RuntimeOverheadCalibration.from_csv(args.runtime_overhead_csv)
        if args.runtime_overhead_csv
        else None
    )
    strided_dma = (
        StridedDmaCalibration.from_csv(args.strided_dma_csv)
        if args.strided_dma_csv
        else None
    )
    if bool(args.dma_affine_read_csv) != bool(args.dma_affine_write_csv):
        parser.error("--dma-affine-read-csv and --dma-affine-write-csv are paired")
    dma_affine = (
        DmaAffineCalibration.from_csvs(
            args.dma_affine_read_csv, args.dma_affine_write_csv, args.dtype
        )
        if args.dma_affine_read_csv
        else None
    )
    model = CostModel(
        dma_calibration=calib,
        dma_write_calibration=write_calib,
        compute_calibration=compute_calib,
        lowering_calibration=lowering_calib,
        compositional_lowering=compositional,
        structured_control_lowering=structured_controls,
        structural_static_dma=structural_static_dma,
        dma_affine_calibration=dma_affine,
        runtime_overhead_calibration=runtime_overhead,
        strided_dma_calibration=strided_dma,
        kernel_overhead_ns=max(0.0, args.kernel_overhead_us * 1000.0),
        dma_queue_count=max(1, args.dma_queue_count),
    )
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_manifest(
        out_dir,
        experiment="nki_operator_holdout",
        config={key: value for key, value in vars(args).items() if key != "resume"},
        repository_root=Path(__file__).resolve().parents[2],
    )
    results_path = out_dir / "operator_results.csv"
    rows: list[dict[str, Any]] = []
    if args.resume and results_path.is_file():
        with results_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    completed = {
        (str(row["op"]), int(row["rows"]), int(row["cols"]), str(row["dtype"]))
        for row in rows
        if row.get("status") == "ok"
    }
    for op in args.ops:
        for r in args.rows:
            for c in args.cols:
                key = (op, r, c, args.dtype)
                mapping_path = (
                    out_dir
                    / f"{op}__r{r}__c{c}__{args.dtype}"
                    / "hardware/source_mapping/instruction_mapping.csv"
                )
                if key in completed and (
                    not args.source_mapping or mapping_path.is_file()
                ):
                    print(
                        f"SKIP completed {op} rows={r} cols={c} {args.dtype}",
                        flush=True,
                    )
                    continue
                rows = [
                    row
                    for row in rows
                    if (
                        str(row.get("op")),
                        int(row.get("rows", -1)),
                        int(row.get("cols", -1)),
                        str(row.get("dtype")),
                    )
                    != key
                ]
                print(f"[{op}] rows={r} cols={c} {args.dtype}", flush=True)
                row = run_case(
                    op,
                    r,
                    c,
                    args.dtype,
                    out_dir,
                    model,
                    args.warmup,
                    args.iters,
                    not args.no_hardware,
                    args.source_mapping,
                    args.compiler_load_cse,
                )
                tail = ""
                if row.get("error_vs_nc_pct") not in (None, ""):
                    tail = f"  err_vs_nc={float(row['error_vs_nc_pct']):+.1f}%"
                print(f"  -> {row['status']}{tail}", flush=True)
                rows.append(row)
                write_csv(rows, results_path)

    print(
        "=== NC-p50 MAPE:",
        _mape(rows, "error_vs_nc_pct"),
        " vector-busy MAPE:",
        _mape(rows, "vector_busy_error_pct"),
        " scalar-busy MAPE:",
        _mape(rows, "scalar_busy_error_pct"),
        " dma-busy MAPE:",
        _mape(rows, "dma_busy_error_pct"),
    )
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
