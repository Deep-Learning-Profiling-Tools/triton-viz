"""The single global completion term that replaced per-structure floors."""

import csv
import json

import pytest

from triton_viz.tools.nki_cost_model import (
    CostModel,
    GlobalCompletionCalibration,
    simulate,
)
from triton_viz.tools.nki_fit_global_completion import main as fit_main


def _events(bytes_moved: int) -> list[dict]:
    return [
        {
            "seq": 0,
            "op": "load",
            "engine": "dma",
            "bytes": bytes_moved,
            "src_storage": 1,
            "src_range": [0, bytes_moved],
            "src_version": 0,
            "dst_storage": 100,
            "dst_range": [0, bytes_moved],
            "dst_version": 0,
        },
        {
            "seq": 1,
            "op": "compute",
            "engine": "vector",
            "elements": bytes_moved,
            "input_storages": [100],
            "input_ranges": [[0, bytes_moved]],
            "input_versions": [0],
            "output_storages": [200],
            "output_ranges": [[0, bytes_moved]],
            "output_versions": [0],
        },
    ]


def test_global_completion_is_critical_plus_unoverlapped_work_plus_offset():
    calibration = GlobalCompletionCalibration(
        overlap_fraction=0.5, completion_offset_ns=1000.0
    )
    assert calibration.predict_ns({"dma": 400.0, "vector": 200.0}) == pytest.approx(
        400.0 + 0.5 * 200.0 + 1000.0
    )
    # An empty program still pays the fixed launch/drain cost.
    assert calibration.predict_ns({}) == pytest.approx(1000.0)


def test_global_completion_raises_latency_and_is_reported():
    calibration = GlobalCompletionCalibration(
        overlap_fraction=1.0, completion_offset_ns=5000.0
    )
    model = CostModel(
        dma_bytes_per_ns=1,
        dma_startup_ns=0,
        vector_elements_per_ns=1,
        cross_engine_sync_ns=0,
        global_completion_calibration=calibration,
    )
    result = simulate(_events(100), model)
    components = result.components_ns
    total_busy = sum(result.engine_busy_ns.values())
    assert components["global_completion_ns"] == pytest.approx(total_busy + 5000.0)
    assert components["global_completion_activated"] == 1.0
    assert result.predicted_latency_ns == pytest.approx(
        max(components["makespan_only_ns"], total_busy + 5000.0)
    )
    assert result.predicted_latency_ns > components["makespan_only_ns"]


def test_global_completion_has_no_structural_key():
    """Two programs with identical engine busy time get identical completion."""
    calibration = GlobalCompletionCalibration(
        overlap_fraction=0.4, completion_offset_ns=800.0
    )
    busy = {"dma": 900.0, "vector": 300.0, "scalar": 100.0}
    assert calibration.predict_ns(busy) == calibration.predict_ns(dict(busy))
    assert calibration.predict_ns(busy) == pytest.approx(
        900.0 + 0.4 * 400.0 + 800.0
    )


def _write_control_root(root, free_dims, offset_us):
    fields = ["op", "rows", "cols", "dtype", "status", "hardware_nc_p50_us"]
    rows = []
    for free_dim in free_dims:
        for op in ("mul2", "softmax"):
            case = f"{op}__r128__c{free_dim}__float32"
            busy_us = free_dim / 1000.0
            (root / case / "hardware").mkdir(parents=True, exist_ok=True)
            (root / case / "hardware" / "explorer_summary.json").write_text(
                json.dumps(
                    {
                        case: {
                            "vector_engine_active_time": busy_us / 1e6,
                            "scalar_engine_active_time": 0.0,
                            "gpsimd_engine_active_time": 0.0,
                            "tensor_engine_active_time": 0.0,
                            "dma_active_time": 0.0,
                        }
                    }
                )
            )
            rows.append(
                {
                    "op": op,
                    "rows": 128,
                    "cols": free_dim,
                    "dtype": "float32",
                    "status": "ok",
                    "hardware_nc_p50_us": busy_us + offset_us,
                }
            )
    with (root / "operator_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_global_completion_fit_passes_cv_and_recovers_the_measured_offset(tmp_path):
    root = tmp_path / "controls"
    _write_control_root(root, (512, 1024, 2048, 4096), offset_us=6.0)
    output, cv = tmp_path / "frozen.csv", tmp_path / "cv.json"
    assert (
        fit_main(
            [
                str(root),
                "--artifact-role",
                "control",
                "--output",
                str(output),
                "--cv-output",
                str(cv),
            ]
        )
        == 0
    )
    calibration = GlobalCompletionCalibration.from_csv(output)
    assert calibration.completion_offset_ns == pytest.approx(6000.0)
    report = json.loads(cv.read_text())
    assert report["passed"] is True
    assert report["target_postcompile_prediction_reads"] is False
    assert len(report["folds"]) == 4


def test_global_completion_fit_refuses_target(tmp_path):
    with pytest.raises(SystemExit, match="Refusing target artifacts"):
        fit_main(
            [
                str(tmp_path),
                "--artifact-role",
                "target",
                "--output",
                str(tmp_path / "out.csv"),
                "--cv-output",
                str(tmp_path / "cv.json"),
            ]
        )


def test_whole_program_routing_interpolates_and_flags_clamping():
    """Engine occupancy is read by interpolation, not nearest neighbour."""
    from triton_viz.tools.nki_cost_model import WholeProgramRoutingCalibration

    def sample(distance, vector):
        return {
            "key": ("float32", 128, (), 1, 0, 0),
            "distance_feature": float(distance),
            "actual": {"vector": vector, "scalar": 0.0, "gpsimd": 0.0},
        }

    calibration = WholeProgramRoutingCalibration(
        [sample(1000.0, 10.0), sample(3000.0, 30.0)]
    )

    class _Descriptor:
        pass

    def predict(distance):
        original = calibration.predict_with_provenance.__globals__
        # drive the lookup directly through a stubbed descriptor
        import triton_viz.tools.nki_evaluate_whole_program_regime as regime

        saved = regime.source_descriptor_from_events
        regime.source_descriptor_from_events = lambda events, dtype, case="": {
            "case": "",
            "key": ("float32", 128, (), 1, 0, 0),
            "distance_feature": float(distance),
        }
        try:
            return calibration.predict_with_provenance([], "float32")
        finally:
            regime.source_descriptor_from_events = saved
        assert original

    busy, match = predict(2000.0)
    assert match == "interpolated"
    assert busy["vector"] == pytest.approx(20_000.0)

    busy, match = predict(1000.0)
    assert match == "exact"
    assert busy["vector"] == pytest.approx(10_000.0)

    busy, match = predict(5000.0)
    assert match == "clamped"
    assert busy["vector"] == pytest.approx(30_000.0)


# --- imbalance-dependent overlap fraction -------------------------------------


def test_imbalance_slope_defaults_to_the_single_constant_form(tmp_path):
    """A calibration CSV without the column must behave exactly as before."""
    path = tmp_path / "gc.csv"
    path.write_text("overlap_fraction,completion_offset_ns\n0.55,5867.825325\n")
    calibration = GlobalCompletionCalibration.from_csv(path)
    assert calibration.overlap_imbalance_slope == 0.0
    busy = {"vector": 100.0, "scalar": 50.0, "dma": 25.0}
    assert calibration.predict_ns(busy) == pytest.approx(
        100.0 + 0.55 * 75.0 + 5867.825325
    )


def test_overlap_grows_with_engine_load_imbalance():
    """More balanced engines contend, so a larger share of the residue is serial."""
    calibration = GlobalCompletionCalibration(
        overlap_fraction=0.30, completion_offset_ns=0.0, overlap_imbalance_slope=0.30
    )
    # One dominant engine: residue/critical is small, so overlap is good.
    dominant = calibration.effective_overlap_fraction(1000.0, 100.0)
    # Comparable engines: residue/critical is large, so overlap is poor.
    balanced = calibration.effective_overlap_fraction(1000.0, 1000.0)
    assert dominant == pytest.approx(0.33)
    assert balanced == pytest.approx(0.60)
    assert balanced > dominant


def test_effective_overlap_fraction_is_clamped_to_a_physical_range():
    calibration = GlobalCompletionCalibration(
        overlap_fraction=0.30, completion_offset_ns=0.0, overlap_imbalance_slope=5.0
    )
    assert calibration.effective_overlap_fraction(1.0, 100.0) == 1.0
    negative = GlobalCompletionCalibration(
        overlap_fraction=0.30, completion_offset_ns=0.0, overlap_imbalance_slope=-5.0
    )
    assert negative.effective_overlap_fraction(1.0, 100.0) == 0.0
    # A zero critical engine falls back to the base fraction rather than dividing.
    assert calibration.effective_overlap_fraction(0.0, 10.0) == pytest.approx(0.30)


def test_imbalance_slope_is_read_from_csv_and_used(tmp_path):
    path = tmp_path / "gc.csv"
    path.write_text(
        "overlap_fraction,completion_offset_ns,overlap_imbalance_slope\n"
        "0.31,5080.455899717484,0.3\n"
    )
    calibration = GlobalCompletionCalibration.from_csv(path)
    assert calibration.overlap_imbalance_slope == pytest.approx(0.3)
    busy = {"vector": 200.0, "scalar": 200.0}
    # residue/critical = 1.0 -> beta = 0.31 + 0.3 = 0.61
    assert calibration.predict_ns(busy) == pytest.approx(
        200.0 + 0.61 * 200.0 + 5080.455899717484
    )
