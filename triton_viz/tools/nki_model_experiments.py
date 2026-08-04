"""Run trace-driven Inf2 NKI model validation over a parameterized workload set.

For every case this tool creates a Triton-Viz beta2 trace, writes ``trace.jsonl``,
runs the calibrated cost model, optionally executes the equivalent kernel on one
Inf2 NeuronCore with ``nki.benchmark``, exports Neuron Explorer summary data, and
writes one stable aggregate CSV row.  Case failures are retained in the CSV.

The supported workloads intentionally distinguish three data paths:

* ``tensor_add``: two HBM copies + VectorE add + one HBM store;
* ``dma_transpose``: HBM DMA-transpose + one HBM store;
* ``sbuf_transpose``: ordinary HBM copy + Static-DMA SBUF scatter + HBM store.

The third path additionally uses a paired Static DMA latency surface.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Any, Callable

import numpy as np

import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_cost_model import (
    CostModel,
    DmaCalibrationSurface,
    StaticDmaCalibrationSurface,
    simulate,
)
from triton_viz.tools.nki_trace_dump import records_to_events, summarize_events, write_jsonl


SUPPORTED_KINDS = ("tensor_add", "dma_transpose", "sbuf_transpose")
CSV_COLUMNS = [
    "case_id", "kind", "status", "error", "p", "f", "x", "y", "dtype",
    "transpose_impl", "trace_path", "event_count", "transfer_count",
    "binary_count", "trace_hbm_read_bytes", "trace_hbm_write_bytes",
    "trace_onchip_bytes", "dma_geometry", "predicted_source_us",
    "kernel_overhead_us", "predicted_total_us",
    "predicted_dma_busy_us", "predicted_vector_busy_us", "hardware_nc_p50_us",
    "predicted_static_dma_group_us",
    "hardware_total_exec_us", "hardware_total_active_us",
    "hardware_dma_active_us", "hardware_vector_active_us",
    "hardware_static_dma_active_us",
    "prediction_error_vs_nc_pct",
    "prediction_error_vs_exec_pct", "prediction_error_vs_active_pct",
    "dma_busy_error_pct", "vector_busy_error_pct",
    "artifact_dir",
]


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number < 1:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _case_id(case: dict[str, Any]) -> str:
    kind = case["kind"]
    if kind in {"tensor_add", "dma_transpose"}:
        return f"{kind}__p{case['p']}__f{case['f']}__{case['dtype']}"
    return (
        f"{kind}__p{case['p']}__x{case['x']}__y{case['y']}__{case['dtype']}"
    )


def _validate_case(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each case must be an object")
    kind = str(raw.get("kind", ""))
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported kind {kind!r}; expected one of {SUPPORTED_KINDS}")
    dtype = str(raw.get("dtype", "float32"))
    if dtype != "float32":
        raise ValueError("current calibration surfaces are FP32; dtype must be 'float32'")
    p = _positive_int(raw.get("p"), "p")
    if p > 128:
        raise ValueError("p must not exceed 128 NeuronCore partitions")
    case: dict[str, Any] = {"kind": kind, "dtype": dtype, "p": p}
    if kind in {"tensor_add", "dma_transpose"}:
        case["f"] = _positive_int(raw.get("f"), "f")
    else:
        case["x"] = _positive_int(raw.get("x"), "x")
        case["y"] = _positive_int(raw.get("y"), "y")
        case["f"] = case["x"] * case["y"]
        # The sample emits one tensor_copy per element; protect users from an
        # accidental enormous unrolled kernel while covering diverse layouts.
        if case["f"] > 512:
            raise ValueError("sbuf_transpose x*y must not exceed 512")
    case["id"] = _case_id(case)
    return case


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("config must contain a non-empty 'cases' list")
    cases: list[dict[str, Any]] = []
    for entry in entries:
        matrix = entry.get("matrix") or {}
        if not isinstance(matrix, dict):
            raise ValueError("case matrix must be an object")
        keys = list(matrix)
        values = []
        for key in keys:
            choices = matrix[key]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"matrix field {key!r} must be a non-empty list")
            values.append(choices)
        combinations = itertools.product(*values) if keys else [()]
        for combination in combinations:
            raw = {key: value for key, value in entry.items() if key != "matrix"}
            raw.update(dict(zip(keys, combination)))
            cases.append(_validate_case(raw))
    ids = [case["id"] for case in cases]
    duplicates = sorted(
        case_id for case_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
    return sorted(cases, key=lambda case: case["id"])


def _trace_kernel(case: dict[str, Any]):
    import nki.isa as nisa
    import nki.language as nl

    p, f = case["p"], case["f"]
    if case["kind"] == "tensor_add":
        def kernel(a, b, out):
            a_tile = nl.ndarray((nl.par_dim(p), f), dtype=a.dtype, buffer=nl.sbuf)
            b_tile = nl.ndarray((nl.par_dim(p), f), dtype=b.dtype, buffer=nl.sbuf)
            c_tile = nl.ndarray((nl.par_dim(p), f), dtype=out.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a)
            nisa.dma_copy(dst=b_tile, src=b)
            nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)
            nisa.dma_copy(dst=out, src=c_tile)
        inputs = [np.ones((p, f), np.float32), np.full((p, f), 2.0, np.float32)]
        output = np.empty((p, f), np.float32)
        return kernel, inputs + [output]

    if case["kind"] == "dma_transpose":
        def kernel(src, out):
            tile = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=tile, src=src)
            nisa.dma_copy(dst=out, src=tile)
        src = np.arange(p * f, dtype=np.float32).reshape(f, p)
        return kernel, [src, np.empty((p, f), np.float32)]

    x, y = case["x"], case["y"]
    def kernel(src, out):
        src_tile = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
        dst_tile = nl.ndarray((nl.par_dim(p), f), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=src_tile, src=src)
        for i in nl.affine_range(x):
            for j in nl.affine_range(y):
                nisa.tensor_copy(
                    dst=dst_tile[:, nl.ds(j * x + i, 1)],
                    src=src_tile[:, nl.ds(i * y + j, 1)],
                )
        nisa.dma_copy(dst=out, src=dst_tile)
    src = np.arange(p * f, dtype=np.float32).reshape(p, f)
    return kernel, [src, np.empty_like(src)]


def _hardware_kernel(case: dict[str, Any]):
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa

    p, f = case["p"], case["f"]
    if case["kind"] == "tensor_add":
        def kernel(a, b):
            out = nl.ndarray((p, f), dtype=a.dtype, buffer=nl.shared_hbm)
            a_tile = nl.ndarray((nl.par_dim(p), f), dtype=a.dtype, buffer=nl.sbuf)
            b_tile = nl.ndarray((nl.par_dim(p), f), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a)
            nisa.dma_copy(dst=b_tile, src=b)
            c_tile = nisa.tensor_tensor(a_tile, b_tile, op=nl.add, engine=nisa.engine.vector)
            nisa.dma_copy(dst=out, src=c_tile)
            return out
        return kernel, [np.ones((p, f), np.float32), np.full((p, f), 2.0, np.float32)]

    if case["kind"] == "dma_transpose":
        def kernel(src):
            out = nl.ndarray((p, f), dtype=src.dtype, buffer=nl.shared_hbm)
            tile = nisa.dma_transpose(src=src)
            nisa.dma_copy(dst=out, src=tile)
            return out
        return kernel, [np.arange(p * f, dtype=np.float32).reshape(f, p)]

    x, y = case["x"], case["y"]
    def kernel(src):
        out = nl.ndarray((p, f), dtype=src.dtype, buffer=nl.shared_hbm)
        src_tile = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
        dst_tile = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=src_tile, src=src)
        for i in nl.affine_range(x):
            for j in nl.affine_range(y):
                dst_tile[:, nl.ds(j * x + i, 1)] = nisa.tensor_copy(
                    src_tile[:, nl.ds(i * y + j, 1)]
                )
        nisa.dma_copy(dst=out, src=dst_tile)
        return out
    return kernel, [np.arange(p * f, dtype=np.float32).reshape(p, f)]


def _write_trace(case: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    kernel, args = _trace_kernel(case)
    triton_viz.clear()
    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    traced[(1,)](*args, pre_trace=False)
    return write_jsonl(launches[-1].records, path)


def _profile_summary(neff: Path, ntff: Path, output: Path) -> dict[str, Any]:
    command = [
        "neuron-explorer", "view", "-n", str(neff), "-s", str(ntff),
        "--output-format", "summary-json", "--disable-ui", "--ignore-event-trace",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (output.parent / "explorer_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output.parent / "explorer_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    start, end = completed.stdout.find("{"), completed.stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Neuron Explorer output contains no JSON")
    data = json.loads(completed.stdout[start:end + 1])
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    models = [value for value in data.values() if isinstance(value, dict)]
    if not models:
        raise ValueError("Neuron Explorer summary contains no model")
    return models[0]


def _nc_p50(benchmark: Any) -> float | None:
    result = getattr(benchmark, "benchmark_result", None)
    full = getattr(result, "full_results", None)
    if isinstance(full, dict):
        values = full.get("nc_latency")
        if isinstance(values, dict) and values.get("50") is not None:
            return float(values["50"])
    metric = getattr(result, "nc_latency", None)
    if metric is not None:
        try:
            return float(metric.get_latency_percentile(50))
        except Exception:
            pass
    return None


def _existing_nc_p50(hardware_dir: Path) -> float | None:
    """Read the prior neuron-bench NC latency samples from an artifact bundle."""
    candidates = sorted(
        (hardware_dir / "compiler_artifacts").rglob("nc_latency_data.json")
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            groups = data.get("latency_data")
            if not isinstance(groups, dict):
                continue
            values = [
                float(value)
                for samples in groups.values()
                if isinstance(samples, list)
                for value in samples
            ]
            if values:
                return float(np.median(values))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _existing_profile(hardware_dir: Path) -> dict[str, Any]:
    path = hardware_dir / "explorer_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    models = [value for value in data.values() if isinstance(value, dict)]
    if not models:
        raise ValueError(f"{path} contains no model summary")
    return models[0]


def _run_hardware(case: dict[str, Any], artifact_dir: Path, warmup: int, iters: int) -> tuple[float | None, dict[str, Any]]:
    from neuronxcc import nki

    kernel, args = _hardware_kernel(case)
    old_cwd = Path.cwd()
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        os.chdir(artifact_dir)
        benchmark = nki.benchmark(
            warmup=warmup, iters=iters, save_neff_name="file.neff",
            save_trace_name="profile.ntff", artifacts_dir="compiler_artifacts",
        )(kernel)
        benchmark(*args)
        summary = _profile_summary(
            artifact_dir / "file.neff", artifact_dir / "profile.ntff",
            artifact_dir / "explorer_summary.json",
        )
        return _nc_p50(benchmark), summary
    finally:
        os.chdir(old_cwd)


def _event_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    hbm_read = hbm_write = onchip = 0
    geometry: list[dict[str, Any]] = []
    for event in events:
        if event.get("op") != "transfer":
            continue
        if event.get("mem_src") == "hbm":
            hbm_read += int(event.get("bytes", 0))
        if event.get("mem_dst") == "hbm":
            hbm_write += int(event.get("bytes", 0))
        if event.get("mem_src") in {"sbuf", "psum"} and event.get("mem_dst") in {"sbuf", "psum"}:
            onchip += int(event.get("bytes", 0))
        if event.get("mem_src") == "hbm" or event.get("mem_dst") == "hbm":
            geometry.append({key: event.get(key) for key in (
                "seq", "mem_src", "mem_dst", "bytes", "partition_count",
                "free_bytes_per_partition", "partition_axis", "dma_pattern",
            )})
    return {
        "event_count": len(events),
        "transfer_count": sum(event.get("op") == "transfer" for event in events),
        "binary_count": sum(event.get("op") == "binary" for event in events),
        "trace_hbm_read_bytes": hbm_read,
        "trace_hbm_write_bytes": hbm_write,
        "trace_onchip_bytes": onchip,
        "dma_geometry": json.dumps(geometry, separators=(",", ":"), sort_keys=True),
    }


def _percent_error(predicted_us: float | None, actual_us: float | None) -> float | None:
    if predicted_us is None or actual_us is None or actual_us == 0:
        return None
    return (predicted_us / actual_us - 1.0) * 100.0


def run_case(
    case: dict[str, Any], output_dir: Path, model: CostModel, *, warmup: int,
    iters: int, hardware: bool, reuse_existing: bool = False,
) -> dict[str, Any]:
    artifact_dir = output_dir / case["id"]
    row: dict[str, Any] = {
        "case_id": case["id"], "kind": case["kind"], "status": "unknown",
        "error": "", "p": case["p"], "f": case["f"],
        "x": case.get("x", ""), "y": case.get("y", ""),
        "dtype": case["dtype"],
        "transpose_impl": (
            "dma_transpose" if case["kind"] == "dma_transpose"
            else "sbuf_scatter" if case["kind"] == "sbuf_transpose" else "none"
        ),
        "artifact_dir": str(artifact_dir),
    }
    try:
        artifact_dir.mkdir(parents=True, exist_ok=reuse_existing)
        trace_path = artifact_dir / "trace.jsonl"
        if reuse_existing and trace_path.is_file():
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if (
                case["kind"] == "sbuf_transpose"
                and not any(event.get("engine") == "static_dma" for event in events)
            ):
                # Regenerate legacy traces written before scalar SBUF copies
                # were classified and grouped as compiler-lowered Static DMA.
                events = _write_trace(case, trace_path)
        else:
            events = _write_trace(case, trace_path)
        row["trace_path"] = str(trace_path)
        row.update(_event_metrics(events))
        simulation = simulate(events, model)
        result = simulation.as_dict()
        overhead_us = max(0.0, model.kernel_overhead_ns) / 1000.0
        row["kernel_overhead_us"] = overhead_us
        row["predicted_source_us"] = (
            simulation.predicted_latency_ns / 1000.0 - overhead_us
        )
        row["predicted_total_us"] = simulation.predicted_latency_ns / 1000.0
        row["predicted_dma_busy_us"] = simulation.engine_busy_ns.get("dma", 0.0) / 1000.0
        row["predicted_vector_busy_us"] = simulation.engine_busy_ns.get("vector", 0.0) / 1000.0
        row["predicted_static_dma_group_us"] = (
            simulation.engine_busy_ns.get("static_dma", 0.0) / 1000.0
        )
        (artifact_dir / "prediction.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if hardware:
            hardware_dir = artifact_dir / "hardware"
            if reuse_existing and (hardware_dir / "explorer_summary.json").is_file():
                nc_p50 = _existing_nc_p50(hardware_dir)
                profile = _existing_profile(hardware_dir)
            else:
                if reuse_existing and hardware_dir.exists():
                    shutil.rmtree(hardware_dir)
                nc_p50, profile = _run_hardware(case, hardware_dir, warmup, iters)
            row["hardware_nc_p50_us"] = nc_p50
            row["hardware_total_exec_us"] = float(profile.get("total_exec_time", 0)) * 1e6
            row["hardware_total_active_us"] = float(profile.get("total_active_time", 0)) * 1e6
            row["hardware_dma_active_us"] = float(profile.get("dma_active_time", 0)) * 1e6
            row["hardware_vector_active_us"] = float(profile.get("vector_engine_active_time", 0)) * 1e6
            row["hardware_static_dma_active_us"] = float(
                profile.get("static_dma_active_time", 0)
            ) * 1e6
            row["prediction_error_vs_nc_pct"] = _percent_error(
                row["predicted_total_us"], row["hardware_nc_p50_us"]
            )
            row["prediction_error_vs_exec_pct"] = _percent_error(
                row["predicted_total_us"], row["hardware_total_exec_us"]
            )
            row["prediction_error_vs_active_pct"] = _percent_error(
                row["predicted_total_us"], row["hardware_total_active_us"]
            )
            row["dma_busy_error_pct"] = _percent_error(
                row["predicted_dma_busy_us"], row["hardware_dma_active_us"]
            )
            row["vector_busy_error_pct"] = _percent_error(
                row["predicted_vector_busy_us"], row["hardware_vector_active_us"]
            )
            if not row["predicted_vector_busy_us"]:
                row["vector_busy_error_pct"] = None
            if case["kind"] == "sbuf_transpose":
                # Explorer's dma_active_time includes Static DMA, while the
                # predicted DMA field intentionally contains HBM DMA only.
                row["dma_busy_error_pct"] = None
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


SUMMARY_COLUMNS = [
    "kind", "cases", "nc_mape_pct", "nc_median_abs_pct", "nc_max_abs_pct",
    "dma_mape_pct", "dma_median_abs_pct", "vector_mape_pct",
    "mean_source_us", "mean_adjusted_us", "mean_nc_p50_us",
]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate successful workload rows into stable per-kind error metrics."""
    summary: list[dict[str, Any]] = []
    for kind in sorted({str(row.get("kind", "")) for row in rows}):
        selected = [
            row for row in rows
            if row.get("kind") == kind and row.get("status") == "ok"
        ]
        if not selected:
            continue

        def numbers(field: str) -> list[float]:
            result: list[float] = []
            for row in selected:
                value = row.get(field)
                if value not in (None, ""):
                    result.append(float(value))
            return result

        nc = [abs(value) for value in numbers("prediction_error_vs_nc_pct")]
        dma = [abs(value) for value in numbers("dma_busy_error_pct")]
        vector = [abs(value) for value in numbers("vector_busy_error_pct")]
        source = numbers("predicted_source_us")
        adjusted = numbers("predicted_total_us")
        hardware = numbers("hardware_nc_p50_us")
        summary.append({
            "kind": kind,
            "cases": len(selected),
            "nc_mape_pct": statistics.mean(nc) if nc else "",
            "nc_median_abs_pct": statistics.median(nc) if nc else "",
            "nc_max_abs_pct": max(nc) if nc else "",
            "dma_mape_pct": statistics.mean(dma) if dma else "",
            "dma_median_abs_pct": statistics.median(dma) if dma else "",
            "vector_mape_pct": statistics.mean(vector) if vector else "",
            "mean_source_us": statistics.mean(source) if source else "",
            "mean_adjusted_us": statistics.mean(adjusted) if adjusted else "",
            "mean_nc_p50_us": statistics.mean(hardware) if hardware else "",
        })
    return summary


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    summary = summarize_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "microbench/inf2_nki/configs/modeling_workloads.json"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--dma-calibration-csv", type=Path, required=True)
    parser.add_argument("--dma-transpose-calibration-csv", type=Path, required=True)
    parser.add_argument("--static-dma-calibration-csv", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument(
        "--kernel-overhead-us",
        type=float,
        default=0.0,
        help="Optional fixed overhead added only to adjusted end-to-end prediction.",
    )
    parser.add_argument(
        "--dma-queue-count",
        type=int,
        default=1,
        help="Number of parallel DMA queues; independent transfers overlap across them.",
    )
    parser.add_argument(
        "--cross-engine-sync-us",
        type=float,
        default=0.0,
        help="Per-dependency handoff latency charged only across engine boundaries.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse trace/hardware artifacts already present in output-dir.",
    )
    parser.add_argument("--kinds", nargs="*", choices=SUPPORTED_KINDS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.iters < 1:
        parser.error("warmup must be >= 0 and iters must be >= 1")
    cases = load_cases(args.config)
    if args.kinds:
        selected = set(args.kinds)
        cases = [case for case in cases if case["kind"] in selected]
    if args.limit is not None:
        if args.limit < 0:
            parser.error("limit must be >= 0")
        cases = cases[:args.limit]
    if not cases:
        parser.error("no cases selected")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=args.reuse_existing)
    copy_surface = DmaCalibrationSurface.from_csv(args.dma_calibration_csv)
    transpose_surface = DmaCalibrationSurface.from_csv(
        args.dma_transpose_calibration_csv, "dma_transpose_surface"
    )
    static_dma_surface = (
        StaticDmaCalibrationSurface.from_csv(args.static_dma_calibration_csv)
        if args.static_dma_calibration_csv else None
    )
    model = CostModel(
        dma_calibration=copy_surface,
        dma_transpose_calibration=transpose_surface,
        static_dma_calibration=static_dma_surface,
        kernel_overhead_ns=max(0.0, args.kernel_overhead_us * 1000.0),
        dma_queue_count=max(1, args.dma_queue_count),
        cross_engine_sync_ns=max(0.0, args.cross_engine_sync_us * 1000.0),
    )
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        row = run_case(
            case, output_dir, model, warmup=args.warmup, iters=args.iters,
            hardware=not args.no_hardware, reuse_existing=args.reuse_existing,
        )
        print(f"  -> {row['status']}", flush=True)
        rows.append(row)
        write_csv(rows, args.output_csv or output_dir / "prediction_results.csv")

    csv_path = args.output_csv or output_dir / "prediction_results.csv"
    summary_path = output_dir / "prediction_summary.csv"
    summary = write_summary_csv(rows, summary_path)
    manifest = {
        "config": str(args.config.resolve()),
        "dma_calibration_csv": str(args.dma_calibration_csv.resolve()),
        "dma_transpose_calibration_csv": str(args.dma_transpose_calibration_csv.resolve()),
        "static_dma_calibration_csv": (
            str(args.static_dma_calibration_csv.resolve())
            if args.static_dma_calibration_csv else None
        ),
        "hardware": not args.no_hardware,
        "kernel_overhead_us": max(0.0, args.kernel_overhead_us),
        "reuse_existing": args.reuse_existing,
        "warmup": args.warmup, "iters": args.iters,
        "cases": len(rows), "ok": sum(row["status"] == "ok" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "csv": str(csv_path),
        "summary_csv": str(summary_path),
        "summary_rows": len(summary),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
