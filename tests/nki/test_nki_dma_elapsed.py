"""Descriptor-issue bound on DMA elapsed time (physical, geometry-keyed)."""

import json

import pytest

from triton_viz.tools.nki_cost_model import (
    CostModel,
    DmaElapsedCalibration,
    simulate,
)
from triton_viz.tools.nki_features import AccessPattern


def _store(free_stride_items, partitions, elements, item_bytes=4, seq=0):
    return {
        "seq": seq,
        "op": "store",
        "engine": "dma",
        "mem_src": "sbuf",
        "mem_dst": "hbm",
        "bytes": elements * item_bytes,
        "partition_count": partitions,
        "active_access_count": elements,
        "free_stride_items": free_stride_items,
        "item_bytes": item_bytes,
        "access_span_bytes": elements * item_bytes * max(1, free_stride_items),
        "src_storage": 1,
        "src_range": [0, elements * item_bytes],
        "src_version": 0,
        "dst_storage": 2,
        "dst_range": [0, elements * item_bytes],
        "dst_version": 1,
    }


def test_contiguous_descriptors_are_striped_across_engines():
    """A contiguous run is one bulk descriptor per partition, issued in parallel."""
    pattern = AccessPattern.from_event(_store(1, 128, 128 * 1024))
    assert DmaElapsedCalibration.serial_descriptor_count(pattern, 16) == 8
    assert DmaElapsedCalibration.serial_descriptor_count(pattern, 1) == 128


def test_fragmented_descriptors_are_one_per_element():
    pattern = AccessPattern.from_event(_store(2, 16, 16 * 512))
    assert DmaElapsedCalibration.serial_descriptor_count(pattern, 16) == 16 * 512


def test_queue_floor_totals_descriptors_across_transfers():
    calibration = DmaElapsedCalibration(ns_per_descriptor=6.0)
    events = [_store(2, 16, 1000, seq=0), _store(2, 16, 500, seq=1)]
    floor, total, fragmented = calibration.queue_floor_ns(events, 16)
    assert total == fragmented == 1500
    assert floor == pytest.approx(9000.0)


def test_queue_floor_raises_dma_busy_but_never_lowers_it():
    events = [_store(2, 16, 20000)]
    model_args = dict(
        dma_bytes_per_ns=1000.0,
        dma_startup_ns=0.0,
        cross_engine_sync_ns=0.0,
    )
    plain = simulate(events, CostModel(**model_args))
    bounded = simulate(
        events,
        CostModel(
            **model_args,
            dma_elapsed_calibration=DmaElapsedCalibration(ns_per_descriptor=6.0),
        ),
    )
    assert bounded.engine_busy_ns["dma"] == pytest.approx(120_000.0)
    assert bounded.engine_busy_ns["dma"] > plain.engine_busy_ns["dma"]
    assert bounded.predicted_latency_ns >= plain.predicted_latency_ns

    # A generous bandwidth model already above the floor is left untouched.
    slow = simulate(
        events,
        CostModel(
            dma_bytes_per_ns=0.01,
            dma_startup_ns=0.0,
            cross_engine_sync_ns=0.0,
            dma_elapsed_calibration=DmaElapsedCalibration(ns_per_descriptor=6.0),
        ),
    )
    assert slow.engine_busy_ns["dma"] > 120_000.0


def test_contiguous_only_program_is_unaffected():
    events = [_store(1, 128, 128 * 1024, seq=0)]
    model_args = dict(
        dma_bytes_per_ns=100.0, dma_startup_ns=0.0, cross_engine_sync_ns=0.0
    )
    plain = simulate(events, CostModel(**model_args))
    bounded = simulate(
        events,
        CostModel(
            **model_args,
            dma_elapsed_calibration=DmaElapsedCalibration(ns_per_descriptor=6.0),
        ),
    )
    assert bounded.predicted_latency_ns == pytest.approx(plain.predicted_latency_ns)
    assert bounded.components_ns["dma_fragmented_descriptor_count"] == 0
    assert bounded.components_ns["dma_queue_floor_ood"] == 0.0


def test_fragmented_descriptor_domain_is_reported_as_ood():
    calibration = DmaElapsedCalibration(
        ns_per_descriptor=6.0,
        measured_min_descriptors=256,
        measured_max_descriptors=1024,
    )
    assert calibration.fragmented_out_of_domain(0) is False
    assert calibration.fragmented_out_of_domain(512) is False
    assert calibration.fragmented_out_of_domain(4096) is True
    result = simulate(
        [_store(2, 16, 4096)],
        CostModel(
            dma_bytes_per_ns=1000.0,
            dma_startup_ns=0.0,
            cross_engine_sync_ns=0.0,
            dma_elapsed_calibration=calibration,
        ),
    )
    assert result.components_ns["dma_queue_floor_ood"] == 1.0


def test_dma_elapsed_fit_refuses_target(tmp_path):
    from triton_viz.tools.nki_fit_dma_elapsed import main

    completion = tmp_path / "completion.csv"
    completion.write_text("overlap_fraction,completion_offset_ns\n0.5,1000.0\n")
    with pytest.raises(SystemExit, match="Refusing target artifacts"):
        main(
            [
                str(tmp_path / "results.jsonl"),
                "--artifact-role",
                "target",
                "--global-completion-csv",
                str(completion),
                "--output",
                str(tmp_path / "out.csv"),
                "--cv-output",
                str(tmp_path / "cv.json"),
            ]
        )


def test_dma_elapsed_fit_recovers_the_issue_interval(tmp_path):
    from triton_viz.tools.nki_fit_dma_elapsed import main

    completion = tmp_path / "completion.csv"
    completion.write_text("overlap_fraction,completion_offset_ns\n0.5,1000.0\n")
    results = tmp_path / "results.jsonl"
    lines = []
    for free_dim in (128, 256, 512, 1024):
        for partitions in (1, 16):
            case = tmp_path / f"case_p{partitions}_f{free_dim}"
            case.mkdir()
            (case / "explorer_summary.json").write_text(
                json.dumps({"c": {"dma_active_time": 0.0}})
            )
            descriptors = 2 * partitions * free_dim
            lines.append(
                json.dumps(
                    {
                        "status": "ok",
                        "dir": str(case),
                        "spec": {
                            "kind": "dma_strided_store",
                            "dtype": "float32",
                            "p": partitions,
                            "f": free_dim,
                        },
                        "latency_percentiles": {
                            "nc_latency": {
                                "p50_us": (descriptors * 7.0 + 1000.0) / 1000.0
                            }
                        },
                    }
                )
            )
    results.write_text("\n".join(lines) + "\n")
    output, cv = tmp_path / "frozen.csv", tmp_path / "cv.json"
    assert (
        main(
            [
                str(results),
                "--artifact-role",
                "control",
                "--global-completion-csv",
                str(completion),
                "--output",
                str(output),
                "--cv-output",
                str(cv),
            ]
        )
        == 0
    )
    calibration = DmaElapsedCalibration.from_csv(output)
    assert calibration.ns_per_descriptor == pytest.approx(7.0, abs=0.05)
    report = json.loads(cv.read_text())
    assert report["passed"] is True
    assert report["target_postcompile_prediction_reads"] is False
    assert len(report["folds"]) == 4
