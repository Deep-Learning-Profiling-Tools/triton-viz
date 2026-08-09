"""Analytical cost model + timeline scheduler for NKI tile trace events.

This module consumes the JSONL event stream produced by
``triton_viz.tools.nki_trace_dump`` and turns it into a *predicted* per-engine
timeline.  It closes the loop that the trace dumper only starts:

    NKI kernel
      -> triton_viz.trace(client=Tracer(), frontend="nki_beta2")
      -> launches[-1].records
      -> nki_trace_dump.records_to_events        (WHAT ran: op/engine/memory/bytes)
      -> nki_cost_model.simulate                  (HOW LONG it takes + overlap)
      -> per-engine timeline + predicted latency  (this file)

The DMA path supports both a geometry-aware analytical fallback and an Inf2
hardware calibration surface loaded from the microbenchmark CSV. Compute
constants remain provisional until their per-engine calibration is integrated.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from triton_viz.tools.nki_features import AccessPattern, ComputeRegion

# ---------------------------------------------------------------------------
# Engine model
# ---------------------------------------------------------------------------
# NeuronCore exposes several independent engines plus DMA queues.  We model each
# as a single serial resource: instructions issued to the same engine run one
# after another, while different engines run in parallel.  This is deliberately
# coarse; it is enough to demonstrate compute/DMA overlap in the timeline.
ENGINE_DMA = "dma"
ENGINE_TENSOR = "tensor"
ENGINE_VECTOR = "vector"
ENGINE_SCALAR = "scalar"
ENGINE_GPSIMD = "gpsimd"
ENGINE_STATIC_DMA = "static_dma"


@dataclass
class DmaCalibrationSurface:
    """Measured aggregate GB/s indexed by (partitions, free bytes/partition)."""

    points: dict[tuple[int, int], float]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        benchmark_name: str = "dma_partition_surface",
        bandwidth_column: str = "derived.read_gbps_dma_active",
        dtype_name: str | None = None,
    ) -> DmaCalibrationSurface:
        points: dict[tuple[int, int], float] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("row_type") != "benchmark" or row.get("status") != "ok":
                    continue
                if row.get("spec.name") != benchmark_name:
                    continue
                if dtype_name is not None:
                    expected = {
                        "bf16": "bfloat16",
                        "fp32": "float32",
                        "fp16": "float16",
                    }.get(dtype_name, dtype_name)
                    observed = {
                        "bf16": "bfloat16",
                        "fp32": "float32",
                        "fp16": "float16",
                    }.get(
                        str(row.get("spec.dtype", "")), str(row.get("spec.dtype", ""))
                    )
                    if observed != expected:
                        continue
                try:
                    key = (
                        int(float(row["work.partition_count"])),
                        int(float(row["work.free_bytes_per_partition"])),
                    )
                    bandwidth = float(row[bandwidth_column])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    key[0] <= 0
                    or key[1] <= 0
                    or not math.isfinite(bandwidth)
                    or bandwidth <= 0
                ):
                    continue
                previous = points.get(key)
                if previous is not None and not math.isclose(
                    previous, bandwidth, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"Conflicting calibration rows for {key}: "
                        f"{previous} versus {bandwidth}"
                    )
                points[key] = bandwidth
        if not points:
            raise ValueError(
                f"No {benchmark_name} calibration rows with {bandwidth_column}"
                f" for dtype={dtype_name or '*'} in {path}"
            )
        return cls(points)

    def bandwidth_gbps(self, partitions: int, free_bytes: int) -> float:
        """Log-space inverse-distance interpolation, clamped to measured bounds."""
        exact = self.points.get((partitions, free_bytes))
        if exact is not None:
            return exact
        if partitions <= 0 or free_bytes <= 0:
            raise ValueError(
                f"DMA geometry must be positive, received "
                f"partitions={partitions}, free_bytes={free_bytes}"
            )
        measured_p = [point[0] for point in self.points]
        measured_f = [point[1] for point in self.points]
        # Clamp extrapolation to the measured rectangle. This keeps the
        # interpolation stable for workload shapes slightly outside the sweep,
        # while still exposing that those shapes are not exact calibration
        # points to callers through ``points``.
        clamped_p = min(max(partitions, min(measured_p)), max(measured_p))
        clamped_f = min(max(free_bytes, min(measured_f)), max(measured_f))
        target_p = math.log2(clamped_p)
        target_f = math.log2(clamped_f)
        nearest = sorted(
            self.points.items(),
            key=lambda item: (
                (math.log2(item[0][0]) - target_p) ** 2
                + (math.log2(item[0][1]) - target_f) ** 2
            ),
        )[:4]
        weighted = 0.0
        weight_sum = 0.0
        for (p, f), bandwidth in nearest:
            distance = math.hypot(math.log2(p) - target_p, math.log2(f) - target_f)
            weight = 1.0 / max(distance, 1e-9)
            weighted += weight * bandwidth
            weight_sum += weight
        return weighted / weight_sum


@dataclass(frozen=True)
class DmaAffineCalibration:
    """Kernel-level DMA startup plus directional byte slopes."""

    startup_ns: float
    read_ns_per_byte: float
    write_ns_per_byte: float

    @staticmethod
    def _fit_direction(
        path: str | Path,
        benchmark_name: str,
        dtype_name: str,
        byte_column: str,
        partition_count: int = 128,
    ) -> tuple[float, float]:
        expected_dtype = {
            "bf16": "bfloat16",
            "fp32": "float32",
            "fp16": "float16",
        }.get(dtype_name, dtype_name)
        samples: list[tuple[float, float]] = []
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if (
                    row.get("row_type") != "benchmark"
                    or row.get("status") != "ok"
                    or row.get("spec.name") != benchmark_name
                    or row.get("spec.dtype") != expected_dtype
                ):
                    continue
                try:
                    if int(float(row["work.partition_count"])) != partition_count:
                        continue
                    samples.append(
                        (
                            float(row[byte_column]),
                            float(row["profile.software_dynamic_dma_active_time"])
                            * 1e9,
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        if len(samples) < 2:
            raise ValueError(
                f"Need at least two {benchmark_name}/{expected_dtype}/p{partition_count} "
                f"dynamic DMA samples in {path}"
            )
        count = float(len(samples))
        sum_x = sum(x for x, _ in samples)
        sum_y = sum(y for _, y in samples)
        sum_xx = sum(x * x for x, _ in samples)
        sum_xy = sum(x * y for x, y in samples)
        denominator = count * sum_xx - sum_x * sum_x
        if denominator <= 0:
            raise ValueError(f"Degenerate DMA affine samples in {path}")
        slope = (count * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / count
        return max(0.0, intercept), max(0.0, slope)

    @classmethod
    def from_csvs(
        cls,
        read_path: str | Path,
        write_path: str | Path,
        dtype_name: str,
    ) -> DmaAffineCalibration:
        read_startup, read_slope = cls._fit_direction(
            read_path,
            "dma_partition_surface",
            dtype_name,
            "work.hbm_read_bytes",
        )
        _, write_slope = cls._fit_direction(
            write_path,
            "dma_write_partition_surface",
            dtype_name,
            "work.hbm_write_bytes",
        )
        return cls(read_startup, read_slope, write_slope)


@dataclass
class StaticDmaCalibrationSurface:
    """Measured incremental SBUF scatter latency indexed by ``(p, x, y)``."""

    points: dict[tuple[int, int, int], float]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        benchmark_name: str = "static_dma_surface",
    ) -> StaticDmaCalibrationSurface:
        scatter: dict[tuple[int, int, int], float] = {}
        baseline: dict[tuple[int, int, int], float] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("row_type") != "benchmark" or row.get("status") != "ok":
                    continue
                if row.get("spec.name") != benchmark_name:
                    continue
                try:
                    key = (
                        int(float(row["work.partition_count"])),
                        int(float(row["work.scatter_rows"])),
                        int(float(row["work.scatter_columns"])),
                    )
                    latency_ns = float(row["latency.nc_latency.p50_us"]) * 1000.0
                except (KeyError, TypeError, ValueError):
                    continue
                if min(key) <= 0 or not math.isfinite(latency_ns) or latency_ns <= 0:
                    continue
                mode = row.get("mode") or row.get("spec.mode")
                target = baseline if mode == "hbm_roundtrip_baseline" else scatter
                previous = target.get(key)
                if previous is not None and not math.isclose(
                    previous, latency_ns, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"Conflicting Static DMA calibration rows for {key}: "
                        f"{previous} versus {latency_ns}"
                    )
                target[key] = latency_ns
        points = {
            key: max(0.0, latency_ns - baseline[key])
            for key, latency_ns in scatter.items()
            if key in baseline
        }
        if not points:
            raise ValueError(f"No paired static_dma_surface calibration rows in {path}")
        return cls(points)

    def latency_ns(self, partitions: int, x: int, y: int) -> float:
        """Return exact or log-space inverse-distance interpolated group latency."""
        exact = self.points.get((partitions, x, y))
        if exact is not None:
            return exact
        if partitions <= 0 or x <= 0 or y <= 0:
            raise ValueError(
                "Static DMA geometry must be positive, received "
                f"partitions={partitions}, x={x}, y={y}"
            )
        bounds = tuple(
            (
                min(point[axis] for point in self.points),
                max(point[axis] for point in self.points),
            )
            for axis in range(3)
        )
        target = (
            min(max(partitions, bounds[0][0]), bounds[0][1]),
            min(max(x, bounds[1][0]), bounds[1][1]),
            min(max(y, bounds[2][0]), bounds[2][1]),
        )
        log_target = tuple(math.log2(value) for value in target)
        nearest = sorted(
            self.points.items(),
            key=lambda item: sum(
                (math.log2(item[0][axis]) - log_target[axis]) ** 2 for axis in range(3)
            ),
        )[:8]
        weighted = 0.0
        weight_sum = 0.0
        for point, latency in nearest:
            distance = math.sqrt(
                sum(
                    (math.log2(point[axis]) - log_target[axis]) ** 2
                    for axis in range(3)
                )
            )
            weight = 1.0 / max(distance, 1e-9)
            weighted += weight * latency
            weight_sum += weight
        return weighted / weight_sum


@dataclass
class StructuralStaticDmaCalibration:
    """Compiler-generated Static DMA busy time keyed by structural grammar."""

    points: dict[tuple[str, int, int], float]

    @classmethod
    def from_csv(cls, path: str | Path) -> StructuralStaticDmaCalibration:
        points: dict[tuple[str, int, int], float] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    key = (
                        row["structural_rule_sequence"],
                        int(row["element_bytes"]),
                        int(row["logical_free_dim"]),
                    )
                    value = float(row["static_dma_ns"])
                except (KeyError, TypeError, ValueError):
                    continue
                if key[0] and min(key[1:]) > 0 and value >= 0:
                    points[key] = value
        if not points:
            raise ValueError(f"No structural Static DMA calibration rows in {path}")
        return cls(points)

    def predict_ns(self, events: Iterable[dict[str, Any]]) -> float:
        regions: dict[int, dict[str, Any]] = {}
        element_bytes = 0
        for event in events:
            if event.get("region_ir") is not None:
                regions[int(event["fusion_group"])] = event["region_ir"]
            if not element_bytes and event.get("op") in {"load", "store"}:
                lanes = int(event.get("active_lanes") or 0)
                nbytes = int(event.get("bytes") or 0)
                if lanes > 0 and nbytes > 0 and nbytes % lanes == 0:
                    element_bytes = nbytes // lanes
        if not regions or element_bytes <= 0:
            return 0.0
        from triton_viz.tools.nki_region_ir import match_structural_family

        sequence = ";".join(
            match_structural_family(regions[group]).rule_id for group in sorted(regions)
        )
        free_dim = max(
            int(region.get("logical_free_dim") or 0) for region in regions.values()
        )
        candidates = [
            (point_free_dim, value)
            for (
                point_sequence,
                point_bytes,
                point_free_dim,
            ), value in self.points.items()
            if point_sequence == sequence and point_bytes == element_bytes
        ]
        if not candidates or free_dim <= 0:
            return 0.0
        return min(
            candidates,
            key=lambda item: abs(math.log2(item[0]) - math.log2(free_dim)),
        )[1]


@dataclass
class ComputeCalibration:
    """Measured per-*instruction* cost for VectorE/ScalarE compute engines.

    Level B of the lowering-aware compute model (Status.md goal 1). Each hardware
    compute instruction over a tile with free dimension ``F`` costs
    ``startup_ns + F * ns_per_free_elem`` (partition-parallel: cost tracks the
    free axis, not total elements). Keyed by ``(engine, dtype, input_streams)``
    so one-input vs two-input and FP32 vs BF16 can differ. This is the cost of a
    *single* lowered instruction; how many instructions a source ``nl.*`` op
    expands into is supplied separately by the Level-A expansion table.
    """

    points: dict

    @classmethod
    def from_csv(cls, path):
        points = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    key = (
                        str(row["engine"]).strip().lower(),
                        str(row["dtype"]).strip().lower(),
                        int(float(row["input_stream_count"])),
                    )
                    value = (float(row["startup_ns"]), float(row["ns_per_free_elem"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if value[0] < 0 or value[1] < 0:
                    continue
                points[key] = value
        if not points:
            raise ValueError(f"No compute calibration rows in {path}")
        return cls(points)

    @staticmethod
    def _norm_dtype(dtype):
        d = (dtype or "").lower()
        if d in ("float32", "fp32", "f32"):
            return "float32"
        if d in ("float16", "fp16", "f16"):
            return "float16"
        if d in ("bfloat16", "bf16"):
            return "bfloat16"
        return d or "float32"

    def instruction_ns(self, engine, dtype, input_streams, free_dim):
        """Return one lowered instruction's cost, or None if uncalibrated."""
        engine = (engine or "").lower()
        dtype = self._norm_dtype(dtype)
        streams = max(1, int(input_streams))
        for key in (
            (engine, dtype, streams),
            (engine, dtype, 2 if streams >= 2 else 1),
            (engine, "float32", streams),
            (engine, "float32", 2 if streams >= 2 else 1),
        ):
            hit = self.points.get(key)
            if hit is not None:
                startup, per_elem = hit
                return startup + max(0, int(free_dim)) * per_elem
        return None


@dataclass
class LoweringExpansionCalibration:
    """Level-A source fusion signature to effective per-engine instructions.

    Points are keyed by ``(signature, dtype, engine, free_dim)`` and carry an
    effective instruction count plus the input-stream class used by Level B.
    Effective counts may be fractional: they represent measured engine active
    time divided by the calibrated single-instruction cost at that shape.
    """

    points: dict[tuple[str, str, str, int], tuple[float, int]]
    fixed_points: dict[tuple[str, str, str, int], float] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> LoweringExpansionCalibration:
        points: dict[tuple[str, str, str, int], tuple[float, int]] = {}
        fixed_points: dict[tuple[str, str, str, int], float] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    key = (
                        str(row["fusion_signature"]).strip(),
                        ComputeCalibration._norm_dtype(row.get("dtype")),
                        str(row["engine"]).strip().lower(),
                        int(float(row["free_dim"])),
                    )
                    value = (
                        float(row["effective_instruction_count"]),
                        int(float(row.get("input_stream_count") or 1)),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if key[0] and key[2] and key[3] > 0 and value[0] > 0:
                    points[key] = value
                    try:
                        fixed_points[key] = max(
                            0.0, float(row.get("kernel_control_active_ns") or 0)
                        )
                    except (TypeError, ValueError):
                        fixed_points[key] = 0.0
        if not points:
            raise ValueError(f"No lowering expansion calibration rows in {path}")
        return cls(points, fixed_points)

    def fixed_ns(self, signature: str, dtype: str, engine: str, free_dim: int) -> float:
        """Return instruction-audited fixed kernel-control work for a region."""
        dtype = ComputeCalibration._norm_dtype(dtype)
        rows = [
            (free, value)
            for (
                sig,
                calibrated_dtype,
                calibrated_engine,
                free,
            ), value in self.fixed_points.items()
            if sig == signature
            and calibrated_dtype == dtype
            and calibrated_engine == engine
        ]
        if not rows:
            return 0.0
        exact = next((value for free, value in rows if free == free_dim), None)
        return (
            exact
            if exact is not None
            else min(rows, key=lambda row: abs(row[0] - free_dim))[1]
        )

    def expansions(
        self, signature: str, dtype: str, free_dim: int, pattern: str | None = None
    ) -> dict[str, tuple[float, int]]:
        """Return exact/interpolated target-engine expansions for one group."""
        dtype = ComputeCalibration._norm_dtype(dtype)
        by_dtype: dict[tuple[str, str], list[tuple[int, float, int]]] = {}
        lookup_signature = signature
        if not any(key[0] == signature for key in self.points) and pattern:
            lookup_signature = f"pattern:{pattern}"
        for (
            sig,
            calibrated_dtype,
            engine,
            calibrated_free,
        ), value in self.points.items():
            if sig == lookup_signature:
                by_dtype.setdefault((engine, calibrated_dtype), []).append(
                    (calibrated_free, value[0], value[1])
                )
        result: dict[str, tuple[float, int]] = {}
        engines = {engine for engine, _ in by_dtype}
        for engine in engines:
            rows = (
                by_dtype.get((engine, dtype)) or by_dtype.get((engine, "float32")) or []
            )
            exact = next((row for row in rows if row[0] == free_dim), None)
            if exact is not None:
                result[engine] = (exact[1], exact[2])
                continue
            rows = sorted(rows)
            target = min(max(free_dim, rows[0][0]), rows[-1][0])
            lower = max(
                (row for row in rows if row[0] <= target), key=lambda row: row[0]
            )
            upper = min(
                (row for row in rows if row[0] >= target), key=lambda row: row[0]
            )
            if lower[0] == upper[0]:
                result[engine] = (lower[1], lower[2])
            else:
                weight = (math.log2(target) - math.log2(lower[0])) / (
                    math.log2(upper[0]) - math.log2(lower[0])
                )
                count = lower[1] + weight * (upper[1] - lower[1])
                result[engine] = (count, lower[2] if weight < 0.5 else upper[2])
        return result


@dataclass
class CompositionalLoweringCalibration:
    """Additive Level-A coefficients over structured region IR features."""

    coefficients: dict[tuple[str, str, str], dict[str, float]]

    @classmethod
    def from_csv(cls, path: str | Path) -> CompositionalLoweringCalibration:
        coefficients = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    str(row["engine"]),
                    ComputeCalibration._norm_dtype(row["dtype"]),
                    str(row["target"]),
                )
                coefficients.setdefault(key, {})[str(row["feature"])] = float(
                    row["coefficient"]
                )
        if not coefficients:
            raise ValueError(f"No compositional lowering coefficients in {path}")
        return cls(coefficients)

    def predict(self, region_ir: dict[str, Any]) -> dict[str, tuple[float, int, float]]:
        from triton_viz.tools.nki_region_ir import compositional_features

        features = compositional_features(region_ir)
        dtype = ComputeCalibration._norm_dtype(region_ir.get("dtype"))
        result = {}
        for engine, streams in (("vector", 2), ("scalar", 1)):

            def value(target, engine=engine):
                weights = self.coefficients.get((engine, dtype, target))
                if not weights:
                    return 0.0
                return max(
                    0.0,
                    sum(
                        weights.get(name, 0.0) * amount
                        for name, amount in features.items()
                    ),
                )

            count = value("effective_count")
            if count > 0:
                result[engine] = (count, streams, value("fixed_ns"))
        return result


@dataclass
class StructuredControlCalibration:
    """Interpolated points keyed by reusable structural grammar families."""

    points: dict[tuple[str, str, str], list[tuple[int, float, int, float]]]

    @classmethod
    def from_csv(cls, path: str | Path) -> StructuredControlCalibration:
        points = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    row.get("calibration_key") or row["family"],
                    row["engine"],
                    ComputeCalibration._norm_dtype(row["dtype"]),
                )
                points.setdefault(key, []).append(
                    (
                        int(row["free_dim"]),
                        float(row["effective_count"]),
                        int(row["instruction_count"]),
                        float(row["fixed_ns"]),
                    )
                )
        return cls(points)

    def predict_points(
        self, region_ir: dict[str, Any]
    ) -> dict[str, tuple[float, int, float]]:
        """Return effective count, real ISA count and fixed time per engine."""
        from triton_viz.tools.nki_region_ir import (
            structural_calibration_key,
            structural_family,
        )

        family = structural_family(region_ir)
        calibration_key, dtype = (
            structural_calibration_key(region_ir),
            ComputeCalibration._norm_dtype(region_ir.get("dtype")),
        )
        # Masked/padded kernels lower for the active logical width. Using the
        # allocation tile width here spuriously extrapolates (Tilebench uses a
        # 16K backing tile even for 128 active columns).
        free = int(region_ir.get("logical_free_dim") or region_ir.get("free_dim") or 1)
        result = {}
        for engine, streams in (("vector", 2), ("scalar", 1)):
            rows = sorted(self.points.get((calibration_key, engine, dtype), []))
            # Backward compatibility for pre-key tables. New tables must not
            # silently cross primitive-specific instruction-selection paths.
            if not rows and (family, engine, dtype) in self.points:
                rows = sorted(self.points[(family, engine, dtype)])
            if not rows and "__" in family:
                rows = sorted(
                    self.points.get((family.split("__", 1)[0], engine, dtype), [])
                )
            if not rows:
                continue
            exact = [row for row in rows if row[0] == free]
            if exact:
                row = exact[0]
            else:
                lower = max(
                    (row for row in rows if row[0] <= free),
                    default=rows[0],
                    key=lambda x: x[0],
                )
                upper = min(
                    (row for row in rows if row[0] >= free),
                    default=rows[-1],
                    key=lambda x: x[0],
                )
                if lower[0] == upper[0]:
                    row = lower
                else:
                    weight = (math.log2(free) - math.log2(lower[0])) / (
                        math.log2(upper[0]) - math.log2(lower[0])
                    )
                    row = (
                        free,
                        lower[1] + weight * (upper[1] - lower[1]),
                        round(lower[2] + weight * (upper[2] - lower[2])),
                        lower[3] + weight * (upper[3] - lower[3]),
                    )
            if row[1] > 0:
                result[engine] = (row[1], row[2], row[3])
        return result

    def predict(self, region_ir: dict[str, Any]) -> dict[str, tuple[float, int, float]]:
        points = self.predict_points(region_ir)
        return {
            engine: (value[0], 2 if engine == "vector" else 1, value[2])
            for engine, value in points.items()
        }


@dataclass
class NcLatencyCalibration:
    """Kernel-level dispatch residual keyed by structural lowering evidence."""

    points: dict[tuple[str, str], list[tuple[int, float]]]

    @classmethod
    def from_csv(cls, path: str | Path) -> NcLatencyCalibration:
        points: dict[tuple[str, str], list[tuple[int, float]]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (row["calibration_key"], ComputeCalibration._norm_dtype(row["dtype"]))
                points.setdefault(key, []).append(
                    (int(row["free_dim"]), float(row["residual_ns"]))
                )
        return cls(points)

    def predict_ns(self, region_ir: dict[str, Any]) -> float | None:
        from triton_viz.tools.nki_region_ir import structural_calibration_key

        key = (
            structural_calibration_key(region_ir),
            ComputeCalibration._norm_dtype(region_ir.get("dtype")),
        )
        rows = sorted(self.points.get(key, []))
        if not rows:
            return None
        free = int(region_ir.get("logical_free_dim") or region_ir.get("free_dim") or 1)
        return min(rows, key=lambda row: abs(math.log2(max(1, row[0]) / free)))[1]


@dataclass
class RuntimeOverheadCalibration:
    """Mechanism-level NC runtime costs fitted from orthogonal controls."""

    sequencer_base_ns: float
    vector_activation_ns: float = 0.0
    scalar_activation_ns: float = 0.0
    tensor_activation_ns: float = 0.0
    cross_engine_sync_ns: float = 0.0
    partition_log2_ns: float = 0.0
    dma_packet_log2_ns: float = 0.0
    partition_min: float = 1.0
    partition_max: float = 128.0
    free_access_min: float = 128.0
    free_access_max: float = 2048.0

    @classmethod
    def from_csv(cls, path: str | Path) -> RuntimeOverheadCalibration:
        with Path(path).open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 1:
            raise ValueError("runtime calibration CSV must contain exactly one row")
        row = rows[0]
        return cls(
            **{
                field.name: float(row.get(field.name) or 0.0)
                for field in fields(cls)
            }
        )

    def predict_ns(
        self,
        engine_busy_ns: dict[str, float],
        cross_engine_edges: int,
        partition_count: int,
        free_access_count: int,
    ) -> float:
        return max(
            0.0,
            self.sequencer_base_ns
            + self.vector_activation_ns * int(ENGINE_VECTOR in engine_busy_ns)
            + self.scalar_activation_ns * int(ENGINE_SCALAR in engine_busy_ns)
            + self.tensor_activation_ns * int(ENGINE_TENSOR in engine_busy_ns)
            + self.cross_engine_sync_ns * max(0, cross_engine_edges)
            + self.partition_log2_ns * math.log2(max(1, partition_count))
            + self.dma_packet_log2_ns
            * math.log2(max(1, free_access_count) / 128.0),
        )

    def in_domain(self, partition_count: int, free_access_count: int) -> bool:
        """Whether runtime geometry lies inside the measured control box."""
        return (
            self.partition_min <= partition_count <= self.partition_max
            and self.free_access_min <= free_access_count <= self.free_access_max
        )


@dataclass
class StridedDmaCalibration:
    """Kernel-level DMA and completion calibration for affine strided stores."""

    points: dict[tuple[str, int, int], list[tuple[int, float, float]]]

    @classmethod
    def from_csv(cls, path: str | Path) -> StridedDmaCalibration:
        points: dict[tuple[str, int, int], list[tuple[int, float, float]]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    ComputeCalibration._norm_dtype(row["dtype"]),
                    int(row["stride_items"]),
                    int(row["partition_count"]),
                )
                points.setdefault(key, []).append(
                    (
                        int(row["free_dim"]),
                        float(row["dma_active_ns"]),
                        float(row["completion_residual_ns"]),
                    )
                )
        return cls(points)

    def predict(self, events: Iterable[dict[str, Any]]) -> tuple[float, float] | None:
        patterns = [
            (event, pattern)
            for event in events
            if (pattern := AccessPattern.from_event(event)) is not None
            and pattern.dst_space == "hbm"
            and pattern.layout_family == "strided"
        ]
        if not patterns:
            return None
        first, first_pattern = patterns[0]
        stride = first_pattern.free_stride_items
        partitions = first_pattern.partition_count
        active = sum(pattern.active_access_count for _, pattern in patterns)
        free = max(1, active // max(1, partitions * len(patterns)))
        dtype = ComputeCalibration._norm_dtype(
            first.get("dtype")
            or ("float16" if first_pattern.item_bytes == 2 else "float32")
        )
        rows = self.points.get((dtype, stride, partitions), [])
        if not rows:
            return None
        rows = sorted(rows)
        lower = max((row for row in rows if row[0] <= free), default=rows[0])
        upper = min((row for row in rows if row[0] >= free), default=rows[-1])
        if lower[0] == upper[0]:
            return lower[1], lower[2]
        # Interpolate only between independent control sizes. Values outside
        # their measured range remain clamped and are reported as OOD by the
        # experiment layer rather than silently extrapolated.
        weight = (free - lower[0]) / (upper[0] - lower[0])
        return (
            lower[1] + weight * (upper[1] - lower[1]),
            lower[2] + weight * (upper[2] - lower[2]),
        )


def _canonical_engine(raw_engine: str, op: str) -> str:
    """Map a coarse dumper engine tag onto a concrete NeuronCore engine.

    The trace dumper only knows rough categories (e.g. it cannot yet tell a
    PSUM/SBUF copy apart into VectorE vs ScalarE).  We resolve those here so the
    scheduler works on a fixed set of resources.  This mapping is the natural
    place to refine once the dumper carries the explicit ``engine`` argument.
    """
    if raw_engine == ENGINE_DMA or raw_engine == "dma_or_vector_load":
        return ENGINE_DMA
    if raw_engine == "dma_or_vector_store":
        return ENGINE_DMA
    if raw_engine == ENGINE_TENSOR:
        return ENGINE_TENSOR
    if raw_engine == ENGINE_STATIC_DMA:
        return ENGINE_STATIC_DMA
    if raw_engine == ENGINE_SCALAR:
        return ENGINE_SCALAR
    if raw_engine == ENGINE_VECTOR:
        return ENGINE_VECTOR
    if raw_engine == ENGINE_GPSIMD:
        return ENGINE_GPSIMD
    if raw_engine == "tensor_or_vector_copy":
        # On-chip PSUM<->SBUF copies are commonly issued on a vector/scalar
        # engine; assume VectorE until the dumper records the real engine token.
        return ENGINE_VECTOR
    if raw_engine == "vector_or_scalar":
        return ENGINE_VECTOR
    # Fallback: keep unknown work off the compute engines so it does not hide
    # real bottlenecks.  Treat it as a generic vector op.
    return ENGINE_VECTOR


def _free_dim(event: dict[str, Any]) -> int | None:
    """Return the free-dimension length (elements per partition) for a compute op.

    VectorE/ScalarE are partition-parallel: the partition axis maps onto lanes,
    so op latency tracks the free (last) axis of the *processed* tile, not the
    total element count. For a reduction the input free dimension is larger than
    the output's (which collapses to 1), and the engine still streams the whole
    input, so we take the max last-dim across the input and output shapes. This
    is exact for elementwise ops (input == output) and correct for reductions.
    Returns ``None`` when no shape is available so the caller can fall back to
    the element-count model.
    """
    free = None
    for key in ("output_shape", "input_shape", "other_shape"):
        shape = event.get(key)
        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
            try:
                candidate = int(shape[-1])
            except (TypeError, ValueError):
                continue
            free = candidate if free is None else max(free, candidate)
    return free


def _input_stream_count(event):
    """Number of distinct tile inputs a compute op streams (1-input vs 2-input).

    Two-input ops (tensor_tensor add/mul/sub/div/max between two tiles) read two
    SBUF streams and cost more per element than one-input ops (activations,
    scalar ops, reductions). Inferred from recorded input pointers, then the op
    name, defaulting to 1.
    """
    explicit = event.get("input_stream_count")
    if explicit is not None:
        return max(1, int(explicit))
    ptrs = event.get("input_ptrs")
    if isinstance(ptrs, (list, tuple)) and len(ptrs) >= 1:
        return 2 if len(ptrs) >= 2 else 1
    if event.get("op") == "binary":
        return 2
    api = str(event.get("api_op") or "")
    two_input = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "greater",
        "less",
        "tensor_tensor",
        "where",
    }
    return 2 if api in two_input else 1


@dataclass
class CostModel:
    """Placeholder analytical cost model (units: nanoseconds).

    All constants are illustrative defaults, NOT measured hardware values.
    Replace them with neuron-profile-calibrated numbers per NeuronCore version.
    """

    # Inf2/NCv2 DMA geometry. Each engine owns up to eight SBUF partitions.
    # The analytical fallback is startup + bytes / active-engine peak; a
    # measured surface can replace it for small/partial tiles.
    dma_startup_ns: float = 300.0
    dma_engine_bytes_per_ns: float = 17.0
    dma_max_engines: int = 16
    dma_bytes_per_ns: float | None = None  # legacy explicit aggregate override
    dma_calibration: DmaCalibrationSurface | None = None
    dma_write_calibration: DmaCalibrationSurface | None = None
    dma_transpose_calibration: DmaCalibrationSurface | None = None
    dma_affine_calibration: DmaAffineCalibration | None = None
    static_dma_calibration: StaticDmaCalibrationSurface | None = None
    structural_static_dma: StructuralStaticDmaCalibration | None = None
    # Level-B per-instruction compute cost (VectorE/ScalarE). When present,
    # NkiCompute/binary/reduce events cost their lowered-instruction count times
    # the measured single-instruction cost instead of the hardcoded VectorE fit.
    compute_calibration: ComputeCalibration | None = None
    lowering_calibration: LoweringExpansionCalibration | None = None
    compositional_lowering: CompositionalLoweringCalibration | None = None
    structured_control_lowering: StructuredControlCalibration | None = None
    nc_latency_calibration: NcLatencyCalibration | None = None
    runtime_overhead_calibration: RuntimeOverheadCalibration | None = None
    strided_dma_calibration: StridedDmaCalibration | None = None

    # On-chip copy (VectorE/ScalarE moving PSUM<->SBUF): cheaper per byte.
    onchip_startup_ns: float = 100.0
    onchip_bytes_per_ns: float = 1000.0

    # TensorE: modeled via a throughput (initiation-interval) style estimate.
    # NOT a single-instruction end-to-end latency, because TensorE is deeply
    # pipelined.  cost ~= startup + flops / peak_flops_per_ns.
    tensor_startup_ns: float = 200.0
    tensor_flops_per_ns: float = 90000.0  # placeholder peak MAC throughput

    # VectorE processes all SBUF partitions in parallel across its lanes, so its
    # latency scales with the *free* dimension (elements per partition), NOT the
    # total element count. The primary model is therefore
    #   cost = vector_free_startup_ns + free_dim / vector_free_per_ns
    # with defaults fit to the Inf2 tensor_add sweep (two-input fp32
    # ``tensor_tensor``): active time = 0.954 us + 2.585 ns * free_dim across
    # p = 1..128, f = 32..1024 (max abs residual 0.32 us). Because it is
    # partition-parallel this fit is flat in p, which is why the old
    # element-count model was ~30x low at small p.
    vector_free_startup_ns: float = 953.7
    # Fit slope is 2.5854 ns per free-element, i.e. ~0.387 free-elems/ns.
    vector_free_per_ns: float = 1.0 / 2.5854  # ~= 0.387 free-elems/ns
    # Legacy element-count fallback, used only when an event carries no output
    # shape to derive the free dimension from (older traces / synthetic events).
    vector_startup_ns: float = 100.0
    vector_elements_per_ns: float = 128.0 * 1.12 / 2.0

    # Optional end-to-end launch/sequencer overhead. This is deliberately
    # separate from engine busy times and defaults to zero for backward
    # compatibility. Inf2 workload validation currently observes an ~8 us
    # residual between the source-operation timeline and NC p50 latency.
    kernel_overhead_ns: float = 0.0

    # Cross-engine handoff latency. When a consumer op depends on a value
    # produced on a *different* engine, the NeuronCore must synchronize the two
    # engines (semaphore set/wait) before the consumer can start. This models
    # that fixed per-dependency cost; it is only charged when the dependency
    # crosses an engine boundary, never within the same serial engine (that is
    # already captured by program-order queueing). Defaults to zero so existing
    # callers and tests are unaffected until a value is calibrated.
    cross_engine_sync_ns: float = 0.0

    # Number of independent DMA queues. NeuronCore-v2 issues DMA on several
    # queues in parallel, so *independent* transfers (no data hazard between
    # them, e.g. loading two distinct operands) overlap instead of serializing.
    # Modeling the DMA engine as this many parallel slots removes a systematic
    # over-count of DMA busy time for multi-input kernels. Data-dependent
    # transfers still serialize through the RAW/WAR/WAW hazard logic. Compute
    # engines stay single-slot. Default 1 preserves prior behavior/tests.
    dma_queue_count: int = 1
    # NCv2 exposes 16 DMA engines. A transfer consumes min(partitions, 16)
    # tokens for its lifetime; only disjoint remaining tokens may issue other
    # transfers concurrently. ``dma_queue_count`` remains a compatibility
    # fallback when this is set to zero.
    dma_resource_count: int = 16
    # Aggregate HBM cap shared by all DMA engines. The analytical path applies
    # it to per-transfer bandwidth; calibrated aggregate slopes already include
    # the cap and are therefore not rescaled.
    hbm_bandwidth_bytes_per_ns: float = 272.0

    def _dma_cost_ns(self, event: dict[str, Any], nbytes: int) -> float:
        if self.dma_affine_calibration is not None:
            is_write = (
                str(event.get("mem_src", "")).lower() == "sbuf"
                and str(event.get("mem_dst", "")).lower() == "hbm"
            )
            slope = (
                self.dma_affine_calibration.write_ns_per_byte
                if is_write
                else self.dma_affine_calibration.read_ns_per_byte
            )
            return nbytes * slope + float(event.get("dma_kernel_startup_ns") or 0.0)
        partitions = int(event.get("partition_count") or 0)
        free_bytes = int(event.get("free_bytes_per_partition") or 0)
        if event.get("dma_pattern") == "transpose":
            calibration = self.dma_transpose_calibration
        elif (
            str(event.get("mem_src", "")).lower() == "sbuf"
            and str(event.get("mem_dst", "")).lower() == "hbm"
        ):
            calibration = self.dma_write_calibration or self.dma_calibration
        else:
            calibration = self.dma_calibration
        if partitions > 0 and free_bytes > 0 and calibration is not None:
            bandwidth = calibration.bandwidth_gbps(partitions, free_bytes)
            return nbytes / bandwidth  # GB/s is numerically bytes/ns
        if self.dma_bytes_per_ns is not None:
            bandwidth = self.dma_bytes_per_ns
        elif partitions > 0:
            # Inf2 parquet shows contiguous partitions striped by engine index:
            # p<=16 activates p engines; p=128 gives eight partitions/engine.
            engines = min(self.dma_max_engines, partitions)
            bandwidth = min(
                engines * self.dma_engine_bytes_per_ns,
                self.hbm_bandwidth_bytes_per_ns,
            )
        else:
            bandwidth = min(
                self.dma_max_engines * self.dma_engine_bytes_per_ns,
                self.hbm_bandwidth_bytes_per_ns,
            )
        return self.dma_startup_ns + nbytes / bandwidth

    def cost_ns(self, event: dict[str, Any]) -> float:
        """Return the estimated duration in ns for one trace event."""
        op = event.get("op")
        engine = _canonical_engine(event.get("engine", ""), op or "")
        if op == "dot":
            flops = event.get("flops") or 0
            return self.tensor_startup_ns + flops / self.tensor_flops_per_ns
        if op in ("binary", "compute", "reduce_sum"):
            free_dim = _free_dim(event)
            # Level-B calibrated path: cost one lowered instruction from the
            # measured per-instruction surface, times the Level-A expansion
            # factor (how many hardware instructions this source op lowers to on
            # this engine). Default expansion is 1 (one source op -> one
            # instruction) until a richer expansion table is supplied.
            if free_dim is not None and self.compute_calibration is not None:
                dtype = str(event.get("output_dtype") or "float32")
                streams = _input_stream_count(event)
                per_instr = self.compute_calibration.instruction_ns(
                    engine, dtype, streams, free_dim
                )
                if per_instr is not None:
                    expansion = float(event.get("lowering_expansion") or 1.0)
                    fixed_ns = float(event.get("lowering_fixed_ns") or 0.0)
                    return per_instr * max(1.0, expansion) + fixed_ns
            if free_dim is not None:
                # Compute events may run on VectorE or ScalarE. For now we apply
                # the same fit (since activation latency bounds are likely
                # similar on the scalar unit).
                # TODO: add scalar-specific calibration surface.
                return self.vector_free_startup_ns + free_dim / self.vector_free_per_ns
            # Fallback: no shape info to derive the free dimension.
            elements = int(event.get("elements", 0))
            return self.vector_startup_ns + elements / self.vector_elements_per_ns
        if op in ("transfer", "load", "store"):
            nbytes = int(event.get("bytes", 0))
            if engine == ENGINE_DMA:
                return self._dma_cost_ns(event, nbytes)
            if engine == ENGINE_STATIC_DMA and self.static_dma_calibration is not None:
                copies = int(event.get("static_dma_group_copies") or 0)
                x = int(event.get("static_dma_group_x") or 0)
                y = int(event.get("static_dma_group_y") or 0)
                partitions = int(event.get("partition_count") or 0)
                if copies > 0 and x > 0 and y > 0:
                    return (
                        self.static_dma_calibration.latency_ns(partitions, x, y)
                        / copies
                    )
            return self.onchip_startup_ns + nbytes / self.onchip_bytes_per_ns
        # grid markers and unknown ops take no engine time.
        return 0.0


@dataclass
class TimelineEntry:
    seq: int
    op: str
    engine: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "op": self.op,
            "engine": self.engine,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.end - self.start, 3),
        }


@dataclass
class SimulationResult:
    predicted_latency_ns: float
    timeline: dict[str, list[TimelineEntry]] = field(default_factory=dict)
    engine_busy_ns: dict[str, float] = field(default_factory=dict)
    components_ns: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_latency_ns": round(self.predicted_latency_ns, 3),
            "engine_busy_ns": {k: round(v, 3) for k, v in self.engine_busy_ns.items()},
            "components_ns": {k: round(v, 3) for k, v in self.components_ns.items()},
            "engine_utilization": {
                k: (
                    round(v / self.predicted_latency_ns, 4)
                    if self.predicted_latency_ns
                    else 0.0
                )
                for k, v in self.engine_busy_ns.items()
            },
            "timeline": {
                engine: [entry.as_dict() for entry in entries]
                for engine, entries in self.timeline.items()
            },
        }


def _memory_key(event: dict[str, Any], side: str) -> int | None:
    """Return the base pointer for the given side of a memory event."""
    return event.get(f"{side}_ptr")


def _read_ptrs(event: dict[str, Any]) -> tuple[int, ...]:
    """Base pointers this op reads from (its input buffers)."""
    ptrs: list[int] = []
    src = event.get("src_ptr")
    if src is not None:
        ptrs.append(int(src))
    for ptr in event.get("input_ptrs", ()) or ():
        if ptr is not None:
            ptrs.append(int(ptr))
    return tuple(ptrs)


def _write_ptrs(event: dict[str, Any]) -> tuple[int, ...]:
    """Base pointers this op writes to (its output buffers)."""
    ptrs: list[int] = []
    dst = event.get("dst_ptr")
    if dst is not None:
        ptrs.append(int(dst))
    out = event.get("output_ptr")
    if out is not None:
        ptrs.append(int(out))
    return tuple(ptrs)


_RANGE_INF = float("inf")


def _access_range(event: dict[str, Any], range_key: str) -> tuple[float, float]:
    """Return the ``[lo, hi)`` byte range for one side of a memory access.

    When the event carries no explicit range we fall back to the whole buffer
    ``[0, inf)``. Two whole-buffer accesses to the same storage therefore always
    overlap, which preserves the original base-pointer-only hazard behavior for
    events (and synthetic tests) that do not supply ranges.
    """
    rng = event.get(range_key)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        if hi > lo:
            return lo, hi
    return 0.0, _RANGE_INF


def _read_accesses(
    event: dict[str, Any],
) -> list[tuple[int, int | None, int, float, float]]:
    """Return ``(storage, version, ptr, lo, hi)`` read accesses.

    ``storage_key`` identifies the underlying allocation for hazard matching. It
    defaults to the base pointer, but an explicit ``storage`` field lets a view
    that records a distinct pointer still alias its parent allocation.
    """
    accesses: list[tuple[int, int | None, int, float, float]] = []
    src = event.get("src_ptr")
    src_storage = event.get("src_storage")
    if src is not None or src_storage is not None:
        key = int(src_storage if src_storage is not None else src)
        ptr = int(src if src is not None else key)
        lo, hi = _access_range(event, "src_range")
        version = event.get("src_version")
        accesses.append((key, int(version) if version is not None else None, ptr, lo, hi))
    storages = event.get("input_storages") or ()
    pointers = event.get("input_ptrs") or ()
    ranges = event.get("input_ranges") or ()
    versions = event.get("input_versions") or ()
    if storages:
        for index, storage in enumerate(storages):
            ptr = pointers[index] if index < len(pointers) else storage
            rng = ranges[index] if index < len(ranges) else None
            lo, hi = _access_range({"range": rng}, "range")
            version = versions[index] if index < len(versions) else None
            accesses.append((int(storage), int(version) if version is not None else None, int(ptr), lo, hi))
        return accesses
    for ptr in pointers:
        if ptr is not None:
            accesses.append((int(ptr), None, int(ptr), 0.0, _RANGE_INF))
    return accesses


def _write_accesses(
    event: dict[str, Any],
) -> list[tuple[int, int | None, int, float, float]]:
    """Return ``(storage, version, ptr, lo, hi)`` write accesses."""
    accesses: list[tuple[int, int | None, int, float, float]] = []
    dst = event.get("dst_ptr")
    dst_storage = event.get("dst_storage")
    if dst is not None or dst_storage is not None:
        key = int(dst_storage if dst_storage is not None else dst)
        ptr = int(dst if dst is not None else key)
        lo, hi = _access_range(event, "dst_range")
        version = event.get("dst_version")
        accesses.append((key, int(version) if version is not None else None, ptr, lo, hi))
    storages = event.get("output_storages") or ()
    pointers = event.get("output_ptrs") or ()
    ranges = event.get("output_ranges") or ()
    versions = event.get("output_versions") or ()
    if storages:
        for index, output_storage in enumerate(storages):
            ptr = pointers[index] if index < len(pointers) else output_storage
            rng = ranges[index] if index < len(ranges) else None
            lo, hi = _access_range({"range": rng}, "range")
            version = versions[index] if index < len(versions) else None
            accesses.append(
                (
                    int(output_storage),
                    int(version) if version is not None else None,
                    int(ptr),
                    lo,
                    hi,
                )
            )
        return accesses
    storage = event.get("output_storage")
    if storage is not None:
        ptr = event.get("output_ptr") or storage
        lo, hi = _access_range(event, "output_range")
        version = event.get("output_version")
        accesses.append((int(storage), int(version) if version is not None else None, int(ptr), lo, hi))
        return accesses
    out = event.get("output_ptr")
    if out is not None:
        accesses.append((int(out), None, int(out), 0.0, _RANGE_INF))
    return accesses


def _ranges_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """True when two half-open byte ranges intersect."""
    return a_lo < b_hi and b_lo < a_hi


def eliminate_redundant_hbm_loads(
    events: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Model exact compiler load CSE without using operator identities.

    A later HBM load is redundant only when it reads the same storage, exact
    byte range, shape, and byte count within the same grid program. Any HBM
    store overlapping the cached range invalidates it. This intentionally does
    not guess at partial overlap, eviction, or pointer aliasing.
    """
    retained: list[dict[str, Any]] = []
    cached: dict[tuple[Any, ...], tuple[float, float]] = {}
    eliminated_count = 0
    eliminated_bytes = 0
    for event in events:
        op = event.get("op")
        if op == "store" and str(event.get("mem_dst", "")).lower() == "hbm":
            storage = event.get("dst_storage", event.get("dst_ptr"))
            span = event.get("dst_range") or ()
            if storage is not None and len(span) == 2:
                lo, hi = float(span[0]), float(span[1])
                cached = {
                    key: cached_span
                    for key, cached_span in cached.items()
                    if key[1] != storage
                    or not _ranges_overlap(lo, hi, cached_span[0], cached_span[1])
                }
        if op == "load" and str(event.get("mem_src", "")).lower() == "hbm":
            storage = event.get("src_storage", event.get("src_ptr"))
            span = event.get("src_range") or ()
            if storage is not None and len(span) == 2:
                key = (
                    tuple(event.get("grid_idx") or ()),
                    storage,
                    tuple(span),
                    tuple(event.get("offsets_shape") or ()),
                    int(event.get("bytes") or 0),
                )
                if key in cached:
                    eliminated_count += 1
                    eliminated_bytes += int(event.get("bytes") or 0)
                    continue
                cached[key] = (float(span[0]), float(span[1]))
        retained.append(event)
    return retained, {
        "eliminated_load_count": eliminated_count,
        "eliminated_load_bytes": eliminated_bytes,
    }


def _expand_lowering_groups(
    events: Iterable[dict[str, Any]], model: CostModel
) -> list[dict[str, Any]]:
    """Replace calibrated source fusion groups with per-engine Level-A work."""
    source = list(events)
    if (
        model.lowering_calibration is None
        and model.compositional_lowering is None
        and model.structured_control_lowering is None
    ) or model.compute_calibration is None:
        return source
    expanded: list[dict[str, Any]] = []
    index = 0
    while index < len(source):
        event = source[index]
        group = event.get("fusion_group")
        signature = event.get("fusion_signature")
        if group is None or not signature:
            expanded.append(event)
            index += 1
            continue
        end = index + 1
        while end < len(source) and source[end].get("fusion_group") == group:
            end += 1
        members = source[index:end]
        free_dim = max((_free_dim(member) or 0) for member in members)
        dtype = str(
            next(
                (
                    member.get("output_dtype")
                    for member in members
                    if member.get("output_dtype")
                ),
                "float32",
            )
        )
        targets = (
            model.lowering_calibration.expansions(
                signature, dtype, free_dim, str(event.get("fusion_pattern") or "")
            )
            if model.lowering_calibration is not None
            else {}
        )
        structured = {}
        lowering_free_dim = free_dim
        if (
            not targets
            and model.compositional_lowering is not None
            and event.get("region_ir")
        ):
            structured = model.compositional_lowering.predict(event["region_ir"])
            targets = {
                engine: (value[0], value[1]) for engine, value in structured.items()
            }
        if (
            not targets
            and model.structured_control_lowering is not None
            and event.get("region_ir")
        ):
            region_ir = dict(event["region_ir"])
            # Normalize the lowering input through the shared, operator-agnostic
            # schema. This keeps partition/free geometry independent of an
            # operator name or structural calibration key.
            compute_region = ComputeRegion.from_event(event)
            if compute_region is not None:
                region_ir["partition_count"] = compute_region.partition_count
                region_ir["logical_free_dim"] = compute_region.logical_free_dim
            if str(region_ir.get("dtype", "")).lower() in {"bool", "boolean"}:
                value_dtypes = [
                    str(value)
                    for member in members
                    for value in (
                        [member.get("output_dtype")]
                        + list(member.get("input_dtypes") or [])
                    )
                    if value and str(value).lower() not in {"bool", "boolean"}
                ]
                if value_dtypes:
                    region_ir["dtype"] = Counter(value_dtypes).most_common(1)[0][0]
            structured = model.structured_control_lowering.predict(region_ir)
            lowering_free_dim = int(
                region_ir.get("logical_free_dim")
                or region_ir.get("free_dim")
                or free_dim
            )
            targets = {
                engine: (value[0], value[1]) for engine, value in structured.items()
            }
        if not targets:
            expanded.extend(members)
            index = end
            continue

        first_inputs = list(members[0].get("input_ptrs") or ())
        first_input_storages = list(members[0].get("input_storages") or ())
        first_input_ranges = list(members[0].get("input_ranges") or ())
        first_input_versions = list(members[0].get("input_versions") or ())
        if any(member.get("input_storages") for member in members):
            external: dict[tuple[int, int | None], tuple[int, Any, int | None]] = {}
            produced: set[tuple[int, int | None]] = set()
            for member in members:
                storages = member.get("input_storages") or ()
                ptrs = member.get("input_ptrs") or ()
                ranges = member.get("input_ranges") or ()
                versions = member.get("input_versions") or ()
                for item_index, storage in enumerate(storages):
                    version = versions[item_index] if item_index < len(versions) else None
                    identity = (int(storage), version)
                    if identity not in produced:
                        ptr = ptrs[item_index] if item_index < len(ptrs) else storage
                        rng = ranges[item_index] if item_index < len(ranges) else None
                        external.setdefault(identity, (int(ptr), rng, version))
                output_storage = member.get("output_storage")
                if output_storage is not None:
                    produced.add((int(output_storage), member.get("output_version")))
            first_input_storages = [identity[0] for identity in external]
            first_inputs = [value[0] for value in external.values()]
            first_input_ranges = [value[1] for value in external.values()]
            first_input_versions = [value[2] for value in external.values()]
        final_output = members[-1].get("output_ptr")
        final_output_storage = members[-1].get("output_storage")
        final_output_range = members[-1].get("output_range")
        final_output_version = members[-1].get("output_version")
        for target_index, (engine, (count, streams)) in enumerate(
            sorted(targets.items())
        ):
            lowered = dict(members[0])
            lowered.update(
                {
                    "op": "compute",
                    "api_op": "lowered_fusion",
                    "engine": engine,
                    "input_shape": [1, lowering_free_dim],
                    "output_shape": [1, lowering_free_dim],
                    "free_dim": lowering_free_dim,
                    "output_dtype": dtype,
                    "input_ptrs": first_inputs,
                    "input_storages": first_input_storages,
                    "input_ranges": first_input_ranges,
                    "input_versions": first_input_versions,
                    "input_stream_count": streams,
                    "lowering_expansion": count,
                    "lowering_fixed_ns": (
                        model.lowering_calibration.fixed_ns(
                            signature, dtype, engine, free_dim
                        )
                        if model.lowering_calibration is not None and not structured
                        else (
                            structured.get(engine, (0.0, streams, 0.0))[2]
                            if int(group) == 0
                            else 0.0
                        )
                    ),
                    "lowered_from_signature": signature,
                    # Publish the logical result only once; otherwise parallel
                    # target engines would create artificial WAW hazards.
                    "output_ptr": final_output
                    if target_index == len(targets) - 1
                    else None,
                    "output_storage": final_output_storage
                    if target_index == len(targets) - 1
                    else None,
                    "output_range": final_output_range
                    if target_index == len(targets) - 1
                    else None,
                    "output_version": final_output_version
                    if target_index == len(targets) - 1
                    else None,
                }
            )
            expanded.append(lowered)
        index = end
    return expanded


def simulate(
    events: Iterable[dict[str, Any]], cost_model: CostModel | None = None
) -> SimulationResult:
    """Schedule trace events onto per-engine timelines with data dependencies.

    Scheduling policy (models both parallelism and true data hazards):
      * Each engine is a serial resource; its next op cannot start until its
        previous op finishes (program order within an engine).
      * Full memory dependencies through base pointers:
          - RAW (read-after-write): an op waits for the last writer of every
            buffer it reads;
          - WAW (write-after-write): an op that writes a buffer waits for the
            previous writer of that buffer;
          - WAR (write-after-read): an op that overwrites a buffer waits for all
            prior readers of that buffer to finish (buffer-reuse anti-hazard).
        WAR/WAW matter on NeuronCore because SBUF tiles are reused across DMA
        and compute engines; ignoring them lets an overwrite float before the
        readers/writers it must follow, under-counting latency.
      * Cross-engine sync: when a resolved dependency crosses an engine boundary
        the consumer additionally waits ``cross_engine_sync_ns`` (semaphore
        handoff). Same-engine dependencies pay nothing beyond queueing.
      * Parallel DMA queues: the DMA engine is modeled as ``dma_queue_count``
        independent slots, so transfers with no data hazard between them overlap
        (each transfer runs on the slot that frees earliest). Compute engines
        are single serial slots. Data-dependent transfers still serialize
        through the hazard logic below.
      * ``grid`` markers are ignored for timing.

    Conservative fallback: a compute op that carries *no* resolvable input
    pointers (older traces) still waits for every memory transfer issued before
    it, so the timeline stays physically plausible. With pointer linkage present
    (see ``Dot.input_ptrs``) the precise RAW edge is used instead.

    The result is a predicted end-to-end latency plus per-engine timelines.
    """
    model = cost_model or CostModel()
    source_events = list(events)
    if model.dma_affine_calibration is not None:
        startup_pending = True
        annotated_events = []
        for source_event in source_events:
            event = dict(source_event)
            if startup_pending and event.get("op") in {"load", "store", "transfer"}:
                event["dma_kernel_startup_ns"] = model.dma_affine_calibration.startup_ns
                startup_pending = False
            annotated_events.append(event)
        source_events = annotated_events
    structural_static_ns = (
        model.structural_static_dma.predict_ns(source_events)
        if model.structural_static_dma is not None
        else 0.0
    )
    strided_dma_prediction = (
        model.strided_dma_calibration.predict(source_events)
        if model.strided_dma_calibration is not None
        else None
    )
    if strided_dma_prediction is not None:
        dma_indices = [
            index
            for index, event in enumerate(source_events)
            if event.get("op") in {"load", "store", "transfer"}
        ]
        raw_costs = [model.cost_ns(source_events[index]) for index in dma_indices]
        raw_total = sum(raw_costs)
        if raw_total > 0:
            scale = strided_dma_prediction[0] / raw_total
            source_events = [dict(event) for event in source_events]
            for index, raw_cost in zip(dma_indices, raw_costs):
                source_events[index]["scheduler_duration_override_ns"] = (
                    raw_cost * scale
                )
    events = _expand_lowering_groups(source_events, model)
    sync_ns = max(0.0, model.cross_engine_sync_ns)
    dma_queue_count = max(1, int(model.dma_queue_count))
    dma_resource_count = max(0, int(model.dma_resource_count))
    # Each engine owns a list of per-slot free times. Most engines are a single
    # serial slot; the DMA engine owns ``dma_queue_count`` parallel slots.
    engine_slots: dict[str, list[float]] = {}
    # Per-storage hazard tracking keyed by storage id. Each writer/reader entry
    # is (lo, hi, end_time, engine) so we can test *address-range* overlap, not
    # just base-pointer equality: disjoint tiles of one allocation run in
    # parallel, while overlapping ranges (even via different view pointers that
    # share a storage id) still serialize with the correct RAW/WAR/WAW edge.
    writers: dict[int, list[tuple[float, float, float, str, int | None]]] = {}
    readers: dict[int, list[tuple[float, float, float, str, int | None]]] = {}
    timeline: dict[str, list[TimelineEntry]] = {}
    engine_busy: dict[str, float] = {}
    makespan = 0.0
    cross_engine_edges = 0
    # Running high-water mark of all memory-transfer completions, used as a
    # conservative dependency floor for compute ops that lack pointer linkage.
    prior_transfer_end = 0.0

    def _dep(
        earliest: float, producer_end: float, producer_engine: str, consumer_engine: str
    ) -> float:
        """Fold one dependency edge in, adding sync cost if it crosses engines."""
        nonlocal cross_engine_edges
        ready = producer_end
        if producer_engine != consumer_engine:
            cross_engine_edges += 1
            ready += sync_ns
        return max(earliest, ready)

    def _slots_for(engine: str) -> list[float]:
        if engine not in engine_slots:
            width = (
                dma_resource_count or dma_queue_count if engine == ENGINE_DMA else 1
            )
            engine_slots[engine] = [0.0] * width
        return engine_slots[engine]

    for event in events:
        op = event.get("op")
        if op in (None, "grid", "unknown"):
            continue
        engine = _canonical_engine(event.get("engine", ""), op)
        duration = float(
            event.get("scheduler_duration_override_ns", model.cost_ns(event))
        )

        reads = _read_accesses(event)
        writes = _write_accesses(event)

        # Earliest start from engine availability. For a multi-slot engine the
        # op takes the slot that frees earliest, letting independent transfers
        # overlap; a single-slot engine keeps strict program order.
        slots = _slots_for(engine)
        if engine == ENGINE_DMA and dma_resource_count:
            partitions = max(1, int(event.get("partition_count") or 1))
            demand = min(dma_resource_count, partitions)
            slot_indices = sorted(range(len(slots)), key=lambda i: slots[i])[:demand]
            earliest = max(slots[index] for index in slot_indices)
        else:
            slot_indices = [min(range(len(slots)), key=lambda i: slots[i])]
            earliest = slots[slot_indices[0]]

        # RAW: wait for prior writers whose range overlaps a buffer we read.
        for key, version, _ptr, lo, hi in reads:
            for w_lo, w_hi, w_end, w_eng, w_version in writers.get(key, ()):
                version_matches = (
                    version is None or w_version is None or version == w_version
                )
                if version_matches and _ranges_overlap(lo, hi, w_lo, w_hi):
                    earliest = _dep(earliest, w_end, w_eng, engine)

        # WAW + WAR: a write waits for prior overlapping writers and readers of
        # that storage (buffer-reuse hazards across engines).
        for key, _version, _ptr, lo, hi in writes:
            for w_lo, w_hi, w_end, w_eng, _w_version in writers.get(key, ()):
                if _ranges_overlap(lo, hi, w_lo, w_hi):
                    earliest = _dep(earliest, w_end, w_eng, engine)
            for r_lo, r_hi, r_end, r_eng, _r_version in readers.get(key, ()):
                if _ranges_overlap(lo, hi, r_lo, r_hi):
                    earliest = _dep(earliest, r_end, r_eng, engine)

        if op == "dot" and not reads:
            # Compute op without pointer linkage: conservatively depend on all
            # memory transfers issued so far (older traces without Dot pointers).
            earliest = max(earliest, prior_transfer_end + (sync_ns if sync_ns else 0.0))

        start = earliest
        end = start + duration

        for slot_index in slot_indices:
            slots[slot_index] = end

        # Publish this op's effects for downstream dependency resolution.
        for key, version, _ptr, lo, hi in reads:
            readers.setdefault(key, []).append((lo, hi, end, engine, version))
        for key, version, _ptr, lo, hi in writes:
            # A new write to a range supersedes prior writers/readers that it
            # fully covers; keep only the non-overlapping history so later
            # disjoint accesses are not spuriously serialized against stale
            # entries, while overlapping history is replaced by this write.
            writers[key] = [
                entry
                for entry in writers.get(key, ())
                if not _ranges_overlap(lo, hi, entry[0], entry[1])
            ]
            writers[key].append((lo, hi, end, engine, version))
            readers[key] = [
                entry
                for entry in readers.get(key, ())
                if not _ranges_overlap(lo, hi, entry[0], entry[1])
            ]
        if op in ("transfer", "load", "store"):
            prior_transfer_end = max(prior_transfer_end, end)

        timeline.setdefault(engine, []).append(
            TimelineEntry(
                seq=int(event.get("seq", -1)),
                op=op,
                engine=engine,
                start=start,
                end=end,
            )
        )
        engine_busy[engine] = engine_busy.get(engine, 0.0) + duration
        makespan = max(makespan, end)

    if structural_static_ns > 0:
        engine_busy[ENGINE_STATIC_DMA] = structural_static_ns
    calibrated_nc_ns = None
    if strided_dma_prediction is not None:
        calibrated_nc_ns = makespan + strided_dma_prediction[1]
    if model.runtime_overhead_calibration is not None:
        max_partitions = max(
            (int(event.get("partition_count") or 1) for event in source_events),
            default=1,
        )
        max_free_access = max(
            (
                max(1, pattern.active_access_count // pattern.partition_count)
                for event in source_events
                if (pattern := AccessPattern.from_event(event)) is not None
            ),
            default=1,
        )
        calibrated_nc_ns = max(
            makespan,
            model.runtime_overhead_calibration.predict_ns(
                engine_busy, cross_engine_edges, max_partitions, max_free_access
            ),
        )
        runtime_control_in_domain = model.runtime_overhead_calibration.in_domain(
            max_partitions, max_free_access
        )
    else:
        runtime_control_in_domain = None
    if model.runtime_overhead_calibration is None and model.nc_latency_calibration is not None:
        region = next(
            (event.get("region_ir") for event in source_events if event.get("region_ir")),
            None,
        )
        load_dtypes = [
            str(event["src_dtype"])
            for event in source_events
            if event.get("op") == "load" and event.get("src_dtype")
        ]
        store_dtypes = [
            str(event["src_dtype"])
            for event in source_events
            if event.get("op") == "store" and event.get("src_dtype")
        ]
        kernel_dtypes = load_dtypes or store_dtypes
        if region and kernel_dtypes:
            region = dict(region)
            region["dtype"] = Counter(kernel_dtypes).most_common(1)[0][0]
        elif region and str(region.get("dtype", "")).lower() in {"bool", "boolean"}:
            region = dict(region)
            value_dtypes = [
                str(value)
                for event in source_events
                if event.get("region_ir_key") == region.get("structural_key")
                for value in (
                    [event.get("output_dtype")] + list(event.get("input_dtypes") or [])
                )
                if value and str(value).lower() not in {"bool", "boolean"}
            ]
            if value_dtypes:
                region["dtype"] = Counter(value_dtypes).most_common(1)[0][0]
        residual_ns = model.nc_latency_calibration.predict_ns(region) if region else None
        if residual_ns is not None and calibrated_nc_ns is None:
            calibrated_nc_ns = makespan + residual_ns
    compute_critical_ns = max(
        engine_busy.get(ENGINE_VECTOR, 0.0),
        engine_busy.get(ENGINE_SCALAR, 0.0),
        engine_busy.get(ENGINE_TENSOR, 0.0),
    )
    dma_busy_ns = engine_busy.get(ENGINE_DMA, 0.0) + engine_busy.get(
        ENGINE_STATIC_DMA, 0.0
    )
    phase_critical_ns = dma_busy_ns + compute_critical_ns
    final_ns = (
        calibrated_nc_ns
        if calibrated_nc_ns is not None
        else makespan + max(0.0, model.kernel_overhead_ns)
    )
    return SimulationResult(
        predicted_latency_ns=final_ns,
        timeline=timeline,
        engine_busy_ns=engine_busy,
        components_ns={
            "compute_only": compute_critical_ns,
            "compute_plus_dma": phase_critical_ns,
            "resource_overlap_makespan": makespan,
            "structural_fixed_completion": final_ns - makespan,
            "runtime_control_in_domain": (
                float(runtime_control_in_domain)
                if runtime_control_in_domain is not None
                else -1.0
            ),
            "final": final_ns,
        },
    )


def simulate_jsonl(
    path: str | Path, cost_model: CostModel | None = None
) -> SimulationResult:
    """Load a JSONL trace file and simulate it."""
    events = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return simulate(events, cost_model=cost_model)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", help="Path to a JSONL trace produced by nki_trace_dump"
    )
    parser.add_argument("--dma-calibration-csv", type=Path, default=None)
    parser.add_argument("--dma-write-calibration-csv", type=Path, default=None)
    parser.add_argument("--dma-transpose-calibration-csv", type=Path, default=None)
    parser.add_argument("--static-dma-calibration-csv", type=Path, default=None)
    parser.add_argument("--compute-calibration-csv", type=Path, default=None)
    parser.add_argument("--lowering-calibration-csv", type=Path, default=None)
    parser.add_argument(
        "--kernel-overhead-us",
        type=float,
        default=0.0,
        help="Optional fixed end-to-end kernel overhead; engine busy times are unchanged.",
    )
    args = parser.parse_args()
    calibration = (
        DmaCalibrationSurface.from_csv(args.dma_calibration_csv)
        if args.dma_calibration_csv
        else None
    )
    write_calibration = (
        DmaCalibrationSurface.from_csv(
            args.dma_write_calibration_csv,
            "dma_write_partition_surface",
            "derived.write_gbps_dma_active",
        )
        if args.dma_write_calibration_csv
        else None
    )
    transpose_calibration = (
        DmaCalibrationSurface.from_csv(
            args.dma_transpose_calibration_csv, "dma_transpose_surface"
        )
        if args.dma_transpose_calibration_csv
        else None
    )
    static_dma_calibration = (
        StaticDmaCalibrationSurface.from_csv(args.static_dma_calibration_csv)
        if args.static_dma_calibration_csv
        else None
    )
    compute_calibration = (
        ComputeCalibration.from_csv(args.compute_calibration_csv)
        if args.compute_calibration_csv
        else None
    )
    lowering_calibration = (
        LoweringExpansionCalibration.from_csv(args.lowering_calibration_csv)
        if args.lowering_calibration_csv
        else None
    )
    result = simulate_jsonl(
        args.trace,
        cost_model=CostModel(
            dma_calibration=calibration,
            dma_write_calibration=write_calibration,
            dma_transpose_calibration=transpose_calibration,
            static_dma_calibration=static_dma_calibration,
            compute_calibration=compute_calibration,
            lowering_calibration=lowering_calibration,
            kernel_overhead_ns=max(0.0, args.kernel_overhead_us * 1000.0),
        ),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
