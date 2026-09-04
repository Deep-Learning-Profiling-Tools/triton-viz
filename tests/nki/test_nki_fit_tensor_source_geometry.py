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


def _write_trials(path, shapes_to_values):
    """shapes_to_values: {(m,n,k): [active_ns per compilation]}"""
    fieldnames = [
        "status", "kind", "spec.m", "spec.n", "spec.k", "spec.dtype",
        "spec.trial", "profile.tensor_engine_active_time",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for dtype in ("float32", "bfloat16"):
            for (m, n, k), values in shapes_to_values.items():
                for index, active_ns in enumerate(values, start=1):
                    writer.writerow({
                        "status": "ok", "kind": "tensor_matmul_tiled",
                        "spec.m": m, "spec.n": n, "spec.k": k, "spec.dtype": dtype,
                        "spec.trial": index,
                        "profile.tensor_engine_active_time": active_ns / 1e9,
                    })
    return path


def test_median_over_compilations_rejects_a_bimodal_outlier(tmp_path):
    """A ~4x slow compilation must not reach the NNLS design."""
    from triton_viz.tools.nki_fit_tensor_source_geometry import _samples

    clean = {(128, 512, 128): [1000.0, 1010.0, 1005.0]}
    dirty = {(128, 512, 128): [1000.0, 4020.0, 1005.0]}
    a = _samples([_write_trials(tmp_path / "clean.csv", clean)])
    b = _samples([_write_trials(tmp_path / "dirty.csv", dirty)])
    assert a["float32"][0][1] == pytest.approx(1005.0)
    assert b["float32"][0][1] == pytest.approx(1005.0)


def test_shapes_are_never_merged_across_suites(tmp_path):
    """Leave-one-suite-out CV requires suites to stay independent."""
    from triton_viz.tools.nki_fit_tensor_source_geometry import _samples

    shapes = {(128, 512, 128): [1000.0], (256, 512, 256): [2000.0]}
    first = _write_trials(tmp_path / "suite_a.csv", shapes)
    second = _write_trials(tmp_path / "suite_b.csv", shapes)
    samples = _samples([first, second])
    suites = [suite for _, _, suite in samples["float32"]]
    # The same shape measured by two suites stays two samples, one per suite.
    assert sorted(suites) == ["suite_a.csv", "suite_a.csv", "suite_b.csv", "suite_b.csv"]
