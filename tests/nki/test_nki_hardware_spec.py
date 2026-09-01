"""Architecture projection must be lossless in units and honest about limits."""

import math

import pytest

from triton_viz.tools.nki_cost_model import GlobalCompletionCalibration
from triton_viz.tools.nki_hardware_spec import (
    INF2,
    TRANSFER_CLASS,
    HardwareSpec,
    project_ns,
    projection_report,
)

# A projected part: 2x clock, 2x lanes, 2x DMA engines, 2x PE dimension.
TRN_NEXT = INF2.scaled(
    name="projected/2x",
    clock_ghz=2.8,
    sbuf_partitions=256,
    dma_engines_per_queue=32,
    pe_rows=256,
    pe_cols=256,
    hbm_peak_gbps=1640.0,
)


def test_clock_conversion_round_trips():
    assert INF2.ns_from_cycles(INF2.cycles_from_ns(123.456)) == pytest.approx(123.456)
    # 1.4 GHz: one cycle is 1/1.4 ns.
    assert INF2.ns_from_cycles(1.0) == pytest.approx(1.0 / 1.4)


def test_projection_to_the_same_spec_is_the_identity():
    """Normalisation is a change of units, never a change of model."""
    for cls in ("cycles", "per_lane", "dimensionless", "spec_term"):
        assert project_ns(5867.825325, INF2, INF2, cls) == pytest.approx(5867.825325)


def test_cycle_costs_scale_inversely_with_clock():
    projected = project_ns(5867.825325, INF2, TRN_NEXT, "cycles")
    assert projected == pytest.approx(5867.825325 * 1.4 / 2.8)


def test_per_lane_costs_scale_with_clock_and_lane_count():
    projected = project_ns(100.0, INF2, TRN_NEXT, "per_lane")
    # twice the lanes and twice the clock -> a quarter of the time
    assert projected == pytest.approx(100.0 / 4.0)


def test_empirical_surfaces_refuse_to_be_projected():
    with pytest.raises(ValueError, match="must be re-measured"):
        project_ns(1.0, INF2, TRN_NEXT, "remeasure")


def test_every_production_calibration_is_classified():
    for key, cls in TRANSFER_CLASS.items():
        assert cls in {"cycles", "per_lane", "dimensionless", "spec_term", "remeasure"}, key


def test_projection_report_states_ratios_and_what_needs_remeasuring():
    report = projection_report(INF2, TRN_NEXT)
    assert report["clock_ratio"] == pytest.approx(2.0)
    assert report["partition_ratio"] == pytest.approx(2.0)
    assert report["out_of_distribution"] is True
    assert "dma_read_surface" in report["requires_remeasurement"]
    assert "global_completion.completion_offset_ns" in report["projectable_coefficients"]
    # A same-spec report is not out of distribution.
    assert projection_report(INF2, INF2)["out_of_distribution"] is False


def test_partition_launch_term_follows_the_spec_without_refitting():
    """The log2(partition) launch cost is already written in spec terms."""
    calibration = GlobalCompletionCalibration(
        overlap_fraction=0.35,
        completion_offset_ns=5102.842179584541,
        overlap_imbalance_slope=0.22,
        completion_offset_ns_per_log2_partition=100.0,
    )
    inf2_offset = calibration.offset_ns(INF2.sbuf_partitions)
    next_offset = calibration.offset_ns(TRN_NEXT.sbuf_partitions)
    # 128 -> 256 partitions adds exactly one log2 step of launch cost.
    assert next_offset - inf2_offset == pytest.approx(100.0)
    assert inf2_offset == pytest.approx(5102.842179584541 + 100.0 * math.log2(128))


def test_completion_prediction_is_unchanged_when_the_spec_is_unchanged():
    """Guard against the projection layer perturbing the frozen inf2 result."""
    calibration = GlobalCompletionCalibration(
        overlap_fraction=0.35,
        completion_offset_ns=5102.842179584541,
        overlap_imbalance_slope=0.22,
        completion_offset_ns_per_log2_partition=100.0,
    )
    busy = {"vector": 9000.0, "scalar": 3500.0, "dma": 2300.0}
    before = calibration.predict_ns(busy, INF2.sbuf_partitions)
    projected = GlobalCompletionCalibration(
        overlap_fraction=calibration.overlap_fraction,
        completion_offset_ns=project_ns(
            calibration.completion_offset_ns, INF2, INF2, "cycles"
        ),
        overlap_imbalance_slope=calibration.overlap_imbalance_slope,
        completion_offset_ns_per_log2_partition=project_ns(
            calibration.completion_offset_ns_per_log2_partition, INF2, INF2, "cycles"
        ),
    )
    assert projected.predict_ns(busy, INF2.sbuf_partitions) == pytest.approx(before)


def test_a_spec_field_must_be_positive_to_be_meaningful():
    spec = HardwareSpec("bad", 0.0, 128, 16, 128, 128, 820.0)
    with pytest.raises(ZeroDivisionError):
        spec.ns_from_cycles(1.0)
