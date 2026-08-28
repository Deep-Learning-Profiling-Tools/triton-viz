import csv

import pytest

from triton_viz.tools.nki_cost_model import NormPipelineCalibration
from triton_viz.tools.nki_fit_norm_pipeline import main


def _write_suite(path, free_dims):
    path.parent.mkdir()
    fields = ("kind", "dtype", "p", "f", "hardware_nc_p50_us")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for kind in ("two_pass_reduce_multiply", "two_pass_reduce_affine"):
            for dtype in ("float32", "bfloat16"):
                for partition in (1, 16, 128):
                    for free_dim in free_dims:
                        writer.writerow(
                            {
                                "kind": kind,
                                "dtype": dtype,
                                "p": partition,
                                "f": free_dim,
                                "hardware_nc_p50_us": 10 + partition / 16 + free_dim / 128,
                            }
                        )


def test_norm_pipeline_control_cv_and_structure_key(tmp_path):
    first = tmp_path / "a" / "control_results.csv"
    second = tmp_path / "b" / "control_results.csv"
    _write_suite(first, (96, 384, 3072, 3584))
    _write_suite(second, (160, 512, 2304, 3328))
    output, cv = tmp_path / "model.csv", tmp_path / "cv.json"
    assert main(
        [
            str(first), str(second), "--artifact-role", "control",
            "--output", str(output), "--cv-output", str(cv),
        ]
    ) == 0
    model = NormPipelineCalibration.from_csv(output)
    value, match = model.predict_ns(
        "float32", "one_reduce_rsqrt_broadcast_multiply", 16, 1, 256
    )
    assert value == pytest.approx((10 + 1 + 2) * 1000)
    assert match == "interpolated"
    missing, match = model.predict_ns(
        "float32", "one_reduce_rsqrt_broadcast_multiply", 16, 1, 3072
    )
    assert missing == 0
    assert match == "missing"


def test_norm_pipeline_fit_refuses_target(tmp_path):
    with pytest.raises(SystemExit, match="Refusing target artifacts"):
        main(
            [
                str(tmp_path / "target.csv"), "--artifact-role", "target",
                "--output", str(tmp_path / "out.csv"),
                "--cv-output", str(tmp_path / "cv.json"),
            ]
        )
