"""Fit mechanism-level NC runtime overhead from orthogonal microbenchmarks."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import triton_viz
from scipy.optimize import least_squares

from microbench.inf2_nki.common.inputs import make_input
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_cost_model import (
    ComputeCalibration,
    CostModel,
    DmaAffineCalibration,
    simulate,
)
from triton_viz.tools.nki_trace_dump import records_to_events

FIELDS = [
    "sequencer_base_ns",
    "vector_activation_ns",
    "scalar_activation_ns",
    "tensor_activation_ns",
    "cross_engine_sync_ns",
    "partition_log2_ns",
    "dma_packet_log2_ns",
]
DOMAIN_FIELDS = [
    "partition_min",
    "partition_max",
    "free_access_min",
    "free_access_max",
]


def _nonnegative_lstsq(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Small exact active-set NNLS without adding a SciPy dependency."""
    best = np.zeros(matrix.shape[1])
    best_error = float("inf")
    for size in range(1, matrix.shape[1] + 1):
        for active in itertools.combinations(range(matrix.shape[1]), size):
            values, *_ = np.linalg.lstsq(matrix[:, active], target, rcond=None)
            if np.any(values < 0):
                continue
            candidate = np.zeros(matrix.shape[1])
            candidate[list(active)] = values
            error = float(np.square(matrix @ candidate - target).sum())
            if error < best_error:
                best, best_error = candidate, error
    return best


def _fit_runtime_path(
    matrix: np.ndarray, scheduler: np.ndarray, measured: np.ndarray
) -> np.ndarray:
    initial = _nonnegative_lstsq(matrix, measured)
    result = least_squares(
        lambda coefficients: np.maximum(
            scheduler, matrix @ coefficients
        ) - measured,
        initial,
        bounds=(0.0, np.inf),
    )
    return result.x


def _trace(spec: dict) -> list[dict]:
    import nki.isa as nisa
    import nki.language as nl

    p, f, mode = int(spec["p"]), int(spec["f"]), str(spec["mode"])

    def kernel(src, out):
        loaded = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
        first = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
        second = nl.ndarray((nl.par_dim(p), f), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(loaded, src)
        value = loaded
        if mode == "vector":
            nisa.tensor_scalar(first, loaded, nl.add, 1.0)
            value = first
        elif mode == "scalar":
            nisa.activation(first, nl.exp, loaded, scale=0.001)
            value = first
        elif mode == "vector_scalar_chain":
            nisa.tensor_scalar(first, loaded, nl.add, 1.0)
            nisa.activation(second, nl.exp, first, scale=0.001)
            value = second
        elif mode == "scalar_vector_chain":
            nisa.activation(first, nl.exp, loaded, scale=0.001)
            nisa.tensor_scalar(second, first, nl.add, 1.0)
            value = second
        nisa.dma_copy(out, value)

    src = make_input((p, f), str(spec["dtype"]), seed=0)
    out = np.empty_like(src)
    triton_viz.clear()
    triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)[(1,)](
        src, out, pre_trace=False
    )
    return records_to_events(launches[-1].records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--dma-affine-read-csv", type=Path, required=True)
    parser.add_argument("--dma-affine-write-csv", type=Path, required=True)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args(argv)

    models: dict[str, CostModel] = {}
    rows = []
    for line in args.results_jsonl.read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        spec = result.get("spec") or {}
        mode = str(spec.get("mode", ""))
        if result.get("status") != "ok" or mode == "empty":
            continue
        dtype = str(spec["dtype"])
        if dtype not in models:
            models[dtype] = CostModel(
                dma_affine_calibration=DmaAffineCalibration.from_csvs(
                    args.dma_affine_read_csv,
                    args.dma_affine_write_csv,
                    dtype,
                ),
                compute_calibration=ComputeCalibration.from_csv(
                    args.compute_calibration_csv
                ),
            )
        events = _trace(spec)
        schedule = simulate(events, models[dtype])
        measured_ns = float(
            result["latency_percentiles"]["nc_latency"]["p50_us"]
        ) * 1000.0
        feature = [
            1.0,
            float("vector" in mode),
            float("scalar" in mode),
            0.0,
            float(mode.endswith("_chain")),
            float(np.log2(max(1, int(spec["p"])))),
            float(np.log2(max(1, int(spec["f"])) / 128.0)),
        ]
        rows.append(
            {
                "case": result["id"],
                "dtype": dtype,
                "p": int(spec["p"]),
                "f": int(spec["f"]),
                "mode": mode,
                "measured_ns": measured_ns,
                "scheduler_ns": schedule.components_ns[
                    "resource_overlap_makespan"
                ],
                "target_overhead_ns": measured_ns,
                "feature": feature,
            }
        )
    if not rows:
        raise ValueError("no successful runtime controls")

    matrix = np.asarray([row["feature"] for row in rows], dtype=float)
    target = np.asarray([row["target_overhead_ns"] for row in rows], dtype=float)
    scheduler = np.asarray([row["scheduler_ns"] for row in rows], dtype=float)
    measured = np.asarray([row["measured_ns"] for row in rows], dtype=float)
    coefficients = _fit_runtime_path(matrix, scheduler, measured)
    runtime_predictions = matrix @ coefficients
    predictions = np.maximum(scheduler, runtime_predictions)
    for row, runtime_prediction, prediction in zip(
        rows, runtime_predictions, predictions
    ):
        row["predicted_overhead_ns"] = float(runtime_prediction)
        row["error_ns"] = float(prediction - row["measured_ns"])
        row.pop("feature")

    partition_validation = {}
    partitions = sorted({int(row["p"]) for row in rows})
    for partition in partitions:
        held = np.asarray([int(row["p"]) == partition for row in rows])
        if not held.any() or held.all():
            continue
        slice_coefficients = _fit_runtime_path(
            matrix[~held], scheduler[~held], measured[~held]
        )
        final_prediction = np.maximum(
            np.asarray([row["scheduler_ns"] for row in rows], dtype=float)[held],
            matrix[held] @ slice_coefficients,
        )
        held_measured = measured[held]
        partition_validation[str(partition)] = {
            "cases": int(held.sum()),
            "nc_mape_pct": float(
                np.mean(np.abs(final_prediction - held_measured) / held_measured)
                * 100
            ),
            "nc_max_abs_error_pct": float(
                np.max(np.abs(final_prediction - held_measured) / held_measured)
                * 100
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS + DOMAIN_FIELDS)
        writer.writeheader()
        writer.writerow({
            **dict(zip(FIELDS, coefficients)),
            "partition_min": min(int(row["p"]) for row in rows),
            "partition_max": max(int(row["p"]) for row in rows),
            "free_access_min": min(int(row["f"]) for row in rows),
            "free_access_max": max(int(row["f"]) for row in rows),
        })
    diagnostics = args.diagnostics or args.output.with_suffix(".diagnostics.json")
    diagnostics.write_text(
        json.dumps(
            {
                "schema": "triton-viz.nki-runtime-overhead-v1",
                "fit_cases": len(rows),
                "rmse_ns": float(np.sqrt(np.square(predictions - target).mean())),
                "max_abs_error_ns": float(np.max(np.abs(predictions - target))),
                "coefficients": dict(zip(FIELDS, coefficients)),
                "cases": rows,
                "leave_one_partition_out": partition_validation,
                "identifiability": {
                    "sequencer_base": "joint launch/sequencer/completion; empty kernel is compiler-invalid",
                    "dma_activation": "absorbed by sequencer_base because every observable kernel performs HBM IO",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Fitted {len(rows)} controls; RMSE={np.sqrt(np.square(predictions-target).mean()):.3f} ns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
