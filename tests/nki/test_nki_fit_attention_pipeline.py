import csv

import pytest

from triton_viz.tools.nki_fit_attention_pipeline import main
from triton_viz.tools.nki_cost_model import AttentionPipelineCalibration


def _write_suite(path, widths):
    fields = [
        "status",
        "spec.kind",
        "spec.dv",
        "profile.tensor_engine_active_time",
        "latency.nc_latency.p50_us",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for width in widths:
            writer.writerow(
                {
                    "status": "ok",
                    "spec.kind": "tensor_attention_pipeline",
                    "spec.dv": width,
                    "profile.tensor_engine_active_time": (2000 + width) / 1e9,
                    "latency.nc_latency.p50_us": (10000 + 2 * width) / 1000,
                }
            )


def test_attention_pipeline_strict_control_cv(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_suite(first, [48, 96, 160])
    _write_suite(second, [64, 112, 144])
    output, cv = tmp_path / "frozen.csv", tmp_path / "cv.json"
    assert main(
        [
            str(first),
            str(second),
            "--artifact-role",
            "control",
            "--output",
            str(output),
            "--cv-output",
            str(cv),
        ]
    ) == 0
    assert output.is_file()
    assert '"target_postcompile_prediction_reads": false' in cv.read_text()
    calibration = AttentionPipelineCalibration.from_csv(output)
    tensor_ns, match = calibration.predict_ns("float32", 104)
    assert tensor_ns == pytest.approx(2104.0)
    assert match == "interpolated"
    # The frozen surface is engine occupancy only: no per-structure completion
    # column survives the removal of the category completion floors.
    assert "nc_completion_ns" not in output.read_text().splitlines()[0]


def test_attention_pipeline_fit_refuses_target(tmp_path):
    with pytest.raises(SystemExit, match="Refusing target artifacts"):
        main(
            [
                str(tmp_path / "target.csv"),
                "--artifact-role",
                "target",
                "--output",
                str(tmp_path / "out.csv"),
                "--cv-output",
                str(tmp_path / "cv.json"),
            ]
        )


# --- median over independent compilations -------------------------------------

HEADER = (
    "row_type,status,spec.kind,spec.dtype,spec.dv,spec.trial,"
    "profile.tensor_engine_active_time\n"
)


def _control_csv(path, points):
    """points: {dv: [tensor_ns per trial]}"""
    lines = [HEADER]
    for width, values in sorted(points.items()):
        for index, value in enumerate(values, start=1):
            lines.append(
                f"benchmark,ok,tensor_attention_pipeline,float32,{width},"
                f"{index},{value / 1e9!r}\n"
            )
    path.write_text("".join(lines))
    return path


def test_median_over_trials_rejects_a_single_slow_compilation(tmp_path):
    """One bimodal outlier per width must not move the frozen surface."""
    from triton_viz.tools.nki_fit_attention_pipeline import _load

    clean = {48: [2600.0, 2610.0, 2605.0], 96: [2700.0, 2705.0, 2702.0]}
    # dv=48 trial 2 lands in the ~2.4x slow allocation.
    dirty = {48: [2600.0, 6240.0, 2605.0], 96: [2700.0, 2705.0, 2702.0]}
    assert _load(_control_csv(tmp_path / "a.csv", clean)) == [
        (48, 2605.0),
        (96, 2702.0),
    ]
    assert _load(_control_csv(tmp_path / "b.csv", dirty)) == [
        (48, 2605.0),
        (96, 2702.0),
    ]


def test_a_single_trial_still_works(tmp_path):
    """Backward compatible with a control set collected before trials existed."""
    from triton_viz.tools.nki_fit_attention_pipeline import _load

    path = _control_csv(tmp_path / "one.csv", {48: [2600.0], 96: [2700.0]})
    assert _load(path) == [(48, 2600.0), (96, 2700.0)]


def test_trial_spread_keeps_the_bimodality_visible(tmp_path):
    """The median must not hide that a width compiled bimodally."""
    from triton_viz.tools.nki_fit_attention_pipeline import trial_spread

    path = _control_csv(tmp_path / "s.csv", {48: [2600.0, 6240.0, 2605.0]})
    spread = trial_spread(path)[48]
    assert spread["trials"] == 3
    assert spread["median_ns"] == pytest.approx(2605.0)
    assert spread["min_ns"] == pytest.approx(2600.0)
    assert spread["max_ns"] == pytest.approx(6240.0)
    assert spread["spread_ratio"] == pytest.approx(2.4, abs=0.05)
