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
    tensor_ns, completion_ns, match = calibration.predict_ns("float32", 104)
    assert tensor_ns == pytest.approx(2104.0)
    assert completion_ns == pytest.approx(10208.0)
    assert match == "interpolated"


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
