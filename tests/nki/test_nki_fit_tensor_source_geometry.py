import csv

import pytest

from triton_viz.tools.nki_fit_tensor_source_geometry import main


def _write_controls(path, scale):
    fieldnames = [
        "status",
        "kind",
        "spec.m",
        "spec.n",
        "spec.k",
        "spec.dtype",
        "profile.tensor_engine_active_time",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for dtype in ("float32", "bfloat16"):
            for m, n, k in ((128, 512, 128), (256, 512, 256), (128, 1024, 384)):
                mt, nt, kt = m // 128, n // 512, k // 128
                active_ns = scale * (
                    100 + 7 * mt * nt * kt + 3 * mt * kt
                    + 2 * kt * nt + mt * nt
                )
                writer.writerow(
                    {
                        "status": "ok",
                        "kind": "tensor_matmul_tiled",
                        "spec.m": m,
                        "spec.n": n,
                        "spec.k": k,
                        "spec.dtype": dtype,
                        "profile.tensor_engine_active_time": active_ns / 1e9,
                    }
                )


def test_control_only_fit_writes_frozen_model_and_cv(tmp_path):
    first = tmp_path / "suite_a.csv"
    second = tmp_path / "suite_b.csv"
    _write_controls(first, 1.0)
    _write_controls(second, 1.0)
    output = tmp_path / "model.csv"
    cv_output = tmp_path / "cv.json"

    assert main(
        [
            str(first),
            str(second),
            "--artifact-role",
            "control",
            "--output",
            str(output),
            "--cv-output",
            str(cv_output),
        ]
    ) == 0
    assert output.is_file()
    assert '"target_postcompile_prediction_reads": false' in cv_output.read_text()


def test_fit_refuses_target_artifacts(tmp_path):
    with pytest.raises(SystemExit, match="Refusing target artifacts"):
        main(
            [
                str(tmp_path / "target.csv"),
                "--artifact-role",
                "target",
                "--output",
                str(tmp_path / "model.csv"),
                "--cv-output",
                str(tmp_path / "cv.json"),
            ]
        )
