"""Shared workload-case helpers extracted from ``nki_model_experiments``.

Keeps ``load_cases``/``write_csv``/small profiling helpers available to
``nki_operator_experiments`` and the test suite after the legacy
``nki_model_experiments`` module was removed. This module only carries
dependencies of those helpers; the legacy run_case/_trace_kernel workload
driver was intentionally dropped.
"""
from __future__ import annotations

import csv
from collections import Counter
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any

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


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _profile_summary(neff: Path, ntff: Path, output: Path) -> dict[str, Any]:
    command = [
        "neuron-explorer", "view", "-n", str(neff), "-s", str(ntff),
        "--output-format", "summary-json", "--disable-ui", "--ignore-event-trace",
    ]
    # Neuron Explorer creates DuckDB metadata stores relative to its working
    # directory.  Use the per-case hardware directory so concurrently profiled
    # controls/holdouts never contend on repository-global ``tables/*.duckdb``.
    completed = subprocess.run(
        command,
        cwd=output.parent,
        text=True,
        capture_output=True,
        check=False,
    )
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


def _percent_error(predicted_us: float | None, actual_us: float | None) -> float | None:
    if predicted_us is None or actual_us is None or actual_us == 0:
        return None
    return (predicted_us / actual_us - 1.0) * 100.0
