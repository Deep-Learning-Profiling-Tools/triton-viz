"""Declarative architecture parameters, so calibrations can be re-expressed.

Every calibration this project freezes is measured on one part (Inferentia2 /
NeuronCore-v2).  Most of the fitted numbers are *not* properties of the silicon
family; they are properties of that part's clock, lane count and engine count.
Stored in absolute nanoseconds they cannot move to another generation at all.
Stored in **normalised** units -- cycles, fractions of peak bandwidth, per-lane
costs -- they can be re-derived for a different spec without re-measurement.

This module holds the spec and the conversions.  It deliberately does *not*
claim that a normalised coefficient is architecture-invariant: that is a
hypothesis which can only be tested against a second part.  What it does
guarantee, and what the tests check, is

1. **Losslessness.**  Re-expressing an Inferentia2 calibration in normalised
   units and evaluating it against the same spec reproduces the original
   absolute prediction exactly.  Normalisation is a change of units, never a
   change of model.
2. **Spec response.**  Changing a spec field moves predictions by the physically
   correct factor, so a projected part behaves as the arithmetic says it should.

Transferability is classified per calibration in ``TRANSFER_CLASS`` below and
is reported alongside any projected prediction, never silently assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HardwareSpec:
    """Architecture parameters needed to turn normalised costs into times.

    ``clock_ghz`` for Inferentia2 is not a datasheet quote: it is derived from
    the profiler's own counters as ``neuroncore_cycle_count / total_time``,
    which returns 1.4000 GHz with p10 == p90 across 295 measured runs.
    """

    name: str
    clock_ghz: float
    sbuf_partitions: int
    dma_engines_per_queue: int
    pe_rows: int
    pe_cols: int
    hbm_peak_gbps: float

    def ns_from_cycles(self, cycles: float) -> float:
        return cycles / self.clock_ghz

    def cycles_from_ns(self, ns: float) -> float:
        return ns * self.clock_ghz

    @property
    def pe_macs_per_cycle(self) -> int:
        return self.pe_rows * self.pe_cols

    def scaled(self, **fields) -> "HardwareSpec":
        """Return a projected spec, e.g. ``spec.scaled(clock_ghz=2.8)``."""
        return replace(self, **fields)


# Measured on the part every frozen calibration in this repository comes from.
INF2 = HardwareSpec(
    name="inf2/NeuronCore-v2",
    clock_ghz=1.4,
    sbuf_partitions=128,
    dma_engines_per_queue=16,
    pe_rows=128,
    pe_cols=128,
    hbm_peak_gbps=820.0,
)


# How each production calibration behaves under a spec change.
#   "cycles"      -- a pure latency in cycles; scales with clock alone.
#   "per_lane"    -- a cost per lane of work; scales with clock and lane count.
#   "spec_term"   -- already written in terms of spec fields; needs nothing.
#   "remeasure"   -- an empirical surface over geometry.  Its *shape* may carry
#                    over but its level cannot be derived from spec alone.
TRANSFER_CLASS = {
    "global_completion.completion_offset_ns": "cycles",
    "global_completion.overlap_fraction": "dimensionless",
    "global_completion.overlap_imbalance_slope": "dimensionless",
    "global_completion.completion_offset_ns_per_log2_partition": "cycles",
    "global_completion.partition_term": "spec_term",
    "dma_elapsed.ns_per_descriptor": "cycles",
    "onchip_transfer.startup_ns": "cycles",
    "onchip_transfer.ns_per_free_elem": "cycles",
    "compute.startup_ns": "cycles",
    "compute.ns_per_free_elem": "cycles",
    "dma_read_surface": "remeasure",
    "dma_write_surface": "remeasure",
    "static_dma": "remeasure",
    "strided_dma": "remeasure",
    "structured_compute": "remeasure",
    "tensor_source_geometry": "remeasure",
    "attention_pipeline": "remeasure",
}


def project_ns(value_ns: float, source: HardwareSpec, target: HardwareSpec,
               transfer_class: str) -> float:
    """Re-express a nanosecond coefficient measured on ``source`` for ``target``.

    Raises for classes that cannot be projected from spec alone, rather than
    silently returning a number that has no basis.
    """
    if transfer_class == "dimensionless":
        return value_ns
    if transfer_class == "cycles":
        return source.cycles_from_ns(value_ns) / target.clock_ghz
    if transfer_class == "per_lane":
        cycles = source.cycles_from_ns(value_ns) * source.sbuf_partitions
        return cycles / target.clock_ghz / target.sbuf_partitions
    if transfer_class == "spec_term":
        return value_ns
    raise ValueError(
        f"{transfer_class!r} cannot be projected from spec alone; the surface "
        "must be re-measured on the target part and reported out-of-distribution"
    )


def projection_report(source: HardwareSpec, target: HardwareSpec) -> dict:
    """What a spec change does and does not license, for OOD reporting."""
    projectable = sorted(
        key for key, cls in TRANSFER_CLASS.items() if cls != "remeasure"
    )
    remeasure = sorted(key for key, cls in TRANSFER_CLASS.items() if cls == "remeasure")
    return {
        "source_spec": source.name,
        "target_spec": target.name,
        "clock_ratio": target.clock_ghz / source.clock_ghz,
        "partition_ratio": target.sbuf_partitions / source.sbuf_partitions,
        "dma_engine_ratio": (
            target.dma_engines_per_queue / source.dma_engines_per_queue
        ),
        "pe_mac_ratio": target.pe_macs_per_cycle / source.pe_macs_per_cycle,
        "hbm_bandwidth_ratio": target.hbm_peak_gbps / source.hbm_peak_gbps,
        "projectable_coefficients": projectable,
        "requires_remeasurement": remeasure,
        # A projected prediction is by construction out of distribution: no
        # measurement on the target part backs it.
        "out_of_distribution": target.name != source.name,
        "log2_partition_term_changes": math.log2(target.sbuf_partitions)
        != math.log2(source.sbuf_partitions),
    }
