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

import numpy as np

import triton_viz
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.trace import trace as tv_trace, launches
from triton_viz.tools.nki_trace_dump import write_jsonl
from triton_viz.tools.nki_cost_model import (
    CostModel,
    DmaCalibrationSurface,
    ComputeCalibration,
    LoweringExpansionCalibration,
    CompositionalLoweringCalibration,
    StructuredControlCalibration,
    simulate,
)
from triton_viz.tools.nki_model_experiments import (
    _profile_summary,
    _nc_p50,
    _percent_error,
)


TILEBENCH_OPS = Path(
    os.environ.get("TILEBENCH_OPS_DIR", "/home/ubuntu/Tilebench/benchmarks/operators")
)

_DTYPES = {"float32": np.float32, "fp32": np.float32,
           "float16": np.float16, "fp16": np.float16,
           "bfloat16": np.dtype("bfloat16"), "bf16": np.dtype("bfloat16")}


def _randn(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    return np.random.randn(*shape).astype(_DTYPES[dtype])


# Each operator: kernel attribute in impl_nki.py + a builder producing the
# canonical NumPy inputs from (rows, cols, dtype). These are fed identically to
# the trace and the hardware benchmark, so both compile the same program.
OPERATORS: dict[str, dict[str, Any]] = {
    "softmax": {
        "kernel": "softmax_online_kernel",
        "inputs": lambda r, c, dt: [_randn((r, c), dt)],
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
        "inputs": lambda r, c, dt: [
            _randn((r, c), dt), _randn((1, c), dt), 1e-6
        ],
    },
    "layernorm": {
        "kernel": "layernorm_kernel",
        "inputs": lambda r, c, dt: [
            _randn((r, c), dt), _randn((1, c), dt),
            _randn((1, c), dt), 1e-5
        ],
    },
}


CSV_COLUMNS = [
    "op", "rows", "cols", "dtype", "status", "error",
    "event_count", "load_count", "store_count", "compute_count",
    "trace_hbm_read_bytes", "trace_hbm_write_bytes",
    "predicted_total_us", "predicted_dma_busy_us",
    "predicted_vector_busy_us", "predicted_scalar_busy_us",
    "hardware_nc_p50_us", "hardware_total_active_us",
    "hardware_dma_active_us", "hardware_vector_active_us",
    "hardware_scalar_active_us",
    "error_vs_nc_pct", "dma_busy_error_pct",
    "vector_busy_error_pct", "scalar_busy_error_pct",
]


def _load_kernel(op: str):
    impl = TILEBENCH_OPS / op / "impl_nki.py"
    spec = importlib.util.spec_from_file_location(f"tilebench_{op}", impl)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, OPERATORS[op]["kernel"])


def _trace_events(op: str, inputs: list[np.ndarray], jsonl: Path) -> list[dict[str, Any]]:
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


def _run_hardware(op: str, inputs: list[np.ndarray], artifact_dir: Path,
                  warmup: int, iters: int, kernel=None) -> tuple[float | None, dict[str, Any]]:
    from neuronxcc import nki

    kernel = kernel or _load_kernel(op)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(artifact_dir)
        bench = nki.benchmark(
            warmup=warmup, iters=iters, save_neff_name="file.neff",
            save_trace_name="profile.ntff", artifacts_dir="compiler_artifacts",
        )(kernel)
        bench(*inputs)
        summary = _profile_summary(
            artifact_dir / "file.neff", artifact_dir / "profile.ntff",
            artifact_dir / "explorer_summary.json",
        )
        return _nc_p50(bench), summary
    finally:
        os.chdir(old)


def run_case(op: str, rows: int, cols: int, dtype: str, out_dir: Path,
             model: CostModel, warmup: int, iters: int,
             hardware: bool) -> dict[str, Any]:
    case_id = f"{op}__r{rows}__c{cols}__{dtype}"
    art = out_dir / case_id
    row: dict[str, Any] = {"op": op, "rows": rows, "cols": cols, "dtype": dtype,
                           "status": "unknown", "error": ""}
    try:
        if art.exists():
            shutil.rmtree(art)
        art.mkdir(parents=True, exist_ok=True)
        inputs = OPERATORS[op]["inputs"](rows, cols, dtype)
        events = _trace_events(op, inputs, art / "trace.jsonl")
        read, write = _hbm_bytes(events)
        row.update(
            event_count=len(events),
            load_count=sum(e.get("op") == "load" for e in events),
            store_count=sum(e.get("op") == "store" for e in events),
            compute_count=sum(e.get("op") in ("compute", "binary", "reduce_sum") for e in events),
            trace_hbm_read_bytes=read, trace_hbm_write_bytes=write,
        )
        sim = simulate(events, model)
        busy = sim.engine_busy_ns
        row["predicted_total_us"] = sim.predicted_latency_ns / 1000.0
        row["predicted_dma_busy_us"] = busy.get("dma", 0.0) / 1000.0
        row["predicted_vector_busy_us"] = busy.get("vector", 0.0) / 1000.0
        row["predicted_scalar_busy_us"] = busy.get("scalar", 0.0) / 1000.0
        (art / "prediction.json").write_text(
            json.dumps(sim.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if hardware:
            nc_p50, prof = _run_hardware(op, inputs, art / "hardware", warmup, iters)
            row["hardware_nc_p50_us"] = nc_p50
            row["hardware_total_active_us"] = float(prof.get("total_active_time", 0)) * 1e6
            row["hardware_dma_active_us"] = float(prof.get("dma_active_time", 0)) * 1e6
            row["hardware_vector_active_us"] = float(prof.get("vector_engine_active_time", 0)) * 1e6
            row["hardware_scalar_active_us"] = float(prof.get("scalar_engine_active_time", 0)) * 1e6
            row["error_vs_nc_pct"] = _percent_error(row["predicted_total_us"], nc_p50)
            row["dma_busy_error_pct"] = _percent_error(
                row["predicted_dma_busy_us"], row["hardware_dma_active_us"])
            row["vector_busy_error_pct"] = _percent_error(
                row["predicted_vector_busy_us"], row["hardware_vector_active_us"])
            row["scalar_busy_error_pct"] = _percent_error(
                row["predicted_scalar_busy_us"], row["hardware_scalar_active_us"])
        row["status"] = "ok"
    except Exception as exc:  # keep going; failures are recorded in the CSV
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
    vals = [abs(float(r[field])) for r in rows
            if r.get("status") == "ok" and r.get(field) not in (None, "")]
    return statistics.mean(vals) if vals else ""


def main(argv: list[str] | None = None) -> int:
    global TILEBENCH_OPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ops", nargs="*", default=["softmax"], choices=sorted(OPERATORS))
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
    parser.add_argument("--kernel-overhead-us", type=float, default=0.0)
    parser.add_argument("--dma-queue-count", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument("--compute-calibration-csv", type=Path, default=None,
                        help="Level-B per-instruction VectorE/ScalarE cost surface.")
    parser.add_argument("--lowering-calibration-csv", type=Path, default=None,
                        help="Level-A fusion-signature to per-engine expansion surface.")
    parser.add_argument("--compositional-lowering-csv", type=Path, default=None,
                        help="Structured additive Level-A coefficients (no operator signatures).")
    parser.add_argument("--structured-control-csv", type=Path, default=None,
                        help="Structural-family control points (no operator signatures).")
    args = parser.parse_args(argv)
    TILEBENCH_OPS = args.tilebench_ops_dir.resolve()
    if not TILEBENCH_OPS.is_dir():
        parser.error(f"Tilebench operators directory does not exist: {TILEBENCH_OPS}")

    calib = (DmaCalibrationSurface.from_csv(args.dma_calibration_csv)
             if args.dma_calibration_csv else None)
    write_calib = (
        DmaCalibrationSurface.from_csv(
            args.dma_write_calibration_csv,
            "dma_write_partition_surface",
            "derived.write_gbps_dma_active",
        )
        if args.dma_write_calibration_csv else None
    )
    compute_calib = (ComputeCalibration.from_csv(args.compute_calibration_csv)
                     if args.compute_calibration_csv else None)
    lowering_calib = (
        LoweringExpansionCalibration.from_csv(args.lowering_calibration_csv)
        if args.lowering_calibration_csv else None
    )
    compositional = (CompositionalLoweringCalibration.from_csv(args.compositional_lowering_csv)
                     if args.compositional_lowering_csv else None)
    structured_controls = (StructuredControlCalibration.from_csv(args.structured_control_csv)
                           if args.structured_control_csv else None)
    model = CostModel(
        dma_calibration=calib,
        dma_write_calibration=write_calib,
        compute_calibration=compute_calib,
        lowering_calibration=lowering_calib,
        compositional_lowering=compositional,
        structured_control_lowering=structured_controls,
        kernel_overhead_ns=max(0.0, args.kernel_overhead_us * 1000.0),
        dma_queue_count=max(1, args.dma_queue_count),
    )
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for op in args.ops:
        for r in args.rows:
            for c in args.cols:
                print(f"[{op}] rows={r} cols={c} {args.dtype}", flush=True)
                row = run_case(op, r, c, args.dtype, out_dir, model,
                               args.warmup, args.iters, not args.no_hardware)
                tail = ""
                if row.get("error_vs_nc_pct") not in (None, ""):
                    tail = f"  err_vs_nc={float(row['error_vs_nc_pct']):+.1f}%"
                print(f"  -> {row['status']}{tail}", flush=True)
                rows.append(row)
                write_csv(rows, out_dir / "operator_results.csv")

    print("=== NC-p50 MAPE:", _mape(rows, "error_vs_nc_pct"),
          " vector-busy MAPE:", _mape(rows, "vector_busy_error_pct"),
          " scalar-busy MAPE:", _mape(rows, "scalar_busy_error_pct"),
          " dma-busy MAPE:", _mape(rows, "dma_busy_error_pct"))
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
