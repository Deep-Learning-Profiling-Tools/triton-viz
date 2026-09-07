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


def test_legacy_csv_preserves_predictions_exactly(tmp_path):
    from triton_viz.tools.nki_cost_model import TensorDotCountCalibration

    path = tmp_path / "inf2.csv"
    path.write_text("dtype,startup_ns,dot_ns,lhs_tile_ns,rhs_tile_ns,output_tile_ns\n"
                    "bfloat16,123.25,4.75,1.5,2.25,3.125\n")
    model = TensorDotCountCalibration.from_csv(path)
    for dots, lhs, rhs, outputs in [(1, 1, 1, 1), (96, 48, 24, 8), (1024, 256, 64, 16)]:
        expected = 123.25 + 4.75 * dots + 1.5 * lhs + 2.25 * rhs + 3.125 * outputs
        assert model.active_ns("bfloat16", dots, lhs, rhs, outputs) == (expected, "source_geometry")


def test_revisit_artifact_roundtrip_and_ood(tmp_path):
    import json
    from triton_viz.tools.nki_cost_model import TensorDotCountCalibration

    inputs = []
    for suite, ks in [("a", [2, 5, 9]), ("b", [3, 6, 10]), ("c", [4, 7, 11])]:
        shapes = {}
        for mt in [1, 2, 5]:
            for nt in [1, 2, 3]:
                for kt in ks:
                    shapes[(128 * mt, 512 * nt, 128 * kt)] = [
                        200 + 50 * mt * nt * kt + 20 * mt * kt + 10 * nt * kt
                        + 5 * mt * nt + 3 * mt * (nt - 1) * kt**2
                    ]
        inputs.append(_write_trials(tmp_path / f"{suite}.csv", shapes))
    output, cv = tmp_path / "model.csv", tmp_path / "cv.json"
    main([*map(str, inputs), "--artifact-role", "control", "--output", str(output),
          "--cv-output", str(cv), "--bf16-revisit-term"])
    assert json.loads(cv.read_text())["mean_wape_pct"]["bfloat16"] < 1e-8
    model = TensorDotCountCalibration.from_csv(output)
    value, match = model.active_ns("bfloat16", 24, 12, 8, 6)
    assert value == pytest.approx(200 + 50*24 + 20*12 + 10*8 + 5*6 + 3*3*1*4**2)
    assert match == "source_geometry_revisit"
    assert model.active_ns("bfloat16", 10000, 1000, 1000, 100)[1].endswith("_ood")


def test_revisit_coefficient_requires_explicit_version(tmp_path):
    from triton_viz.tools.nki_cost_model import TensorDotCountCalibration

    path = tmp_path / "unversioned.csv"
    path.write_text("dtype,startup_ns,dot_ns,revisit_ns\nbfloat16,100,10,2\n")
    with pytest.raises(ValueError, match="requires a model version"):
        TensorDotCountCalibration.from_csv(path)


@pytest.mark.parametrize("whole_surface", [False, True])
def test_revisit_ood_propagates_to_simulation(whole_surface):
    from triton_viz.tools.nki_cost_model import (
        CostModel, TensorDotCountCalibration, TensorCalibrationSurface, simulate,
    )

    calibration = TensorDotCountCalibration(
        {"bfloat16": (100, 10, 1, 1, 1)},
        {"bfloat16": 2},
        {"bfloat16": [[2, 20]] * 5},
    )
    event = {"op": "dot", "engine": "tensor", "seq": 0, "flops": 1024,
             "input_dtypes": ["bfloat16", "bfloat16"],
             "tensor_source_dot_count": 1, "tensor_source_lhs_tile_count": 1,
             "tensor_source_rhs_tile_count": 1, "tensor_source_output_tile_count": 1}
    model = CostModel(tensor_dot_count_calibration=calibration)
    if whole_surface:
        model.tensor_calibration = TensorCalibrationSurface(
            {"bfloat16": (100, 10)}, {"bfloat16": (1, 1e6)})
    result = simulate([event], model)
    assert result.components_ns["tensor_source_geometry_ood_count"] == 1


def test_revisit_option_leaves_fp32_fit_and_predictions_unchanged(tmp_path):
    from triton_viz.tools.nki_cost_model import TensorDotCountCalibration

    controls = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for path in controls:
        _write_controls(path, 1.0)
    models = []
    fitted_rows = []
    for name, flags in [("legacy", []), ("revisit", ["--bf16-revisit-term"])]:
        output = tmp_path / f"{name}.csv"
        assert main([*map(str, controls), "--artifact-role", "control", "--output",
                     str(output), "--cv-output", str(tmp_path / f"{name}.json"),
                     *flags]) == 0
        models.append(TensorDotCountCalibration.from_csv(output))
        with output.open() as file:
            fitted_rows.append(next(row for row in csv.DictReader(file)
                                    if row["dtype"] == "float32"))
    for key in fitted_rows[0]:
        assert fitted_rows[0][key] == fitted_rows[1][key]
    assert models[0].active_ns("float32", 24, 12, 8, 6) == models[1].active_ns("float32", 24, 12, 8, 6)
