"""The on-chip PSUM/SBUF copy surface must stay reproducible from repo code.

This suite guards the chain that produced the production
``onchip_transfer_frozen_v1.csv``: the microbench benchmark must stay
registered, and the fitter must keep differencing the fixed per-kernel cost out
of the repeat sweep before it fits the free-width line.
"""

import csv
import json

import pytest

from triton_viz.tools.nki_fit_onchip_copy import (
    cross_validate,
    fit,
    main,
    width_slopes,
)

# startup 60 ns + 1.5 ns per free element, plus a fixed 900 ns per-kernel cost
# that the repeat difference has to remove.
FIXED_NS = 900.0
STARTUP_NS = 60.0
PER_ELEM_NS = 1.5
WIDTHS = (48, 160, 320)
REPEATS = (1, 4, 8, 16)


def _row(dtype, free, repeat, active_ns):
    return {
        "row_type": "benchmark",
        "status": "ok",
        "run_id": "onchip_copy_disjoint_v2",
        "spec.kind": "onchip_copy",
        "spec.dtype": dtype,
        "spec.f": str(free),
        "spec.p": "64",
        "spec.repeat": str(repeat),
        "work.logical_instructions": str(repeat),
        "profile.vector_engine_active_time": repr(active_ns * 1e-9),
    }


def _rows(dtype="float32", startup=STARTUP_NS, per_elem=PER_ELEM_NS):
    rows = []
    for free in WIDTHS:
        per_copy = startup + per_elem * free
        for repeat in REPEATS:
            rows.append(_row(dtype, free, repeat, FIXED_NS + per_copy * repeat))
    return rows


def _write(tmp_path, rows, name="controls.csv"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_repeat_difference_removes_the_fixed_per_kernel_cost():
    slopes = width_slopes(_rows())["float32"]
    for free in WIDTHS:
        assert slopes[free] == pytest.approx(STARTUP_NS + PER_ELEM_NS * free)


def test_a_larger_fixed_cost_does_not_move_the_slopes():
    baseline = width_slopes(_rows())["float32"]
    shifted = _rows()
    for row in shifted:
        row["profile.vector_engine_active_time"] = repr(
            float(row["profile.vector_engine_active_time"]) + 5e-6
        )
    assert width_slopes(shifted)["float32"] == pytest.approx(baseline)


def test_fit_recovers_the_free_width_line(tmp_path):
    path = _write(tmp_path, _rows())
    records = fit(_rows(), path)
    assert len(records) == 1
    record = records[0]
    assert record["engine"] == "vector"
    assert record["dtype"] == "float32"
    assert record["startup_ns"] == pytest.approx(STARTUP_NS)
    assert record["ns_per_free_elem"] == pytest.approx(PER_ELEM_NS)
    assert (record["domain_min_free"], record["domain_max_free"]) == (48, 320)


def test_leave_one_width_out_is_exact_on_a_linear_surface():
    report = cross_validate(width_slopes(_rows()), 20.0)
    assert report["protocol"] == "repeat-difference then leave-one-width-out"
    assert len(report["folds"]) == len(WIDTHS)
    assert report["mean_wape_pct"] == pytest.approx(0.0, abs=1e-9)
    assert report["pass"] is True
    assert report["target_postcompile_prediction_reads"] is False


def test_a_non_linear_surface_fails_the_gate():
    rows = []
    for free in WIDTHS:
        # Quadratic in the free width: the held-out width is badly extrapolated.
        per_copy = STARTUP_NS + 0.004 * free * free
        for repeat in REPEATS:
            rows.append(_row("float32", free, repeat, FIXED_NS + per_copy * repeat))
    report = cross_validate(width_slopes(rows), 20.0)
    assert report["pass"] is False
    assert report["mean_wape_pct"] > 20.0


def test_three_widths_are_required_for_the_gate():
    rows = [r for r in _rows() if int(r["spec.f"]) != 160]
    with pytest.raises(SystemExit, match="needs >=3 widths"):
        cross_validate(width_slopes(rows), 20.0)


def test_target_artifacts_are_mechanically_refused(tmp_path):
    path = _write(tmp_path, _rows())
    with pytest.raises(SystemExit, match="Refusing target artifacts"):
        main([
            str(path), "--artifact-role", "target",
            "--output", str(tmp_path / "o.csv"),
            "--cv-output", str(tmp_path / "cv.json"),
        ])
    assert not (tmp_path / "o.csv").exists()


def test_a_failing_gate_emits_no_production_surface(tmp_path):
    path = _write(tmp_path, _rows())
    out = tmp_path / "surface.csv"
    with pytest.raises(SystemExit, match="refusing to emit"):
        main([
            str(path), "--artifact-role", "control", "--max-mean-wape", "0.0",
            "--output", str(out), "--cv-output", str(tmp_path / "cv.json"),
        ])
    assert not out.exists()
    # The CV report is still written so the negative result is auditable.
    assert json.loads((tmp_path / "cv.json").read_text())["pass"] is False


def test_cli_writes_both_dtypes_in_the_production_schema(tmp_path):
    rows = _rows("float32") + _rows("bfloat16", startup=66.0, per_elem=1.49)
    path = _write(tmp_path, rows)
    out = tmp_path / "onchip_transfer.csv"
    assert main([
        str(path), "--artifact-role", "control",
        "--output", str(out), "--cv-output", str(tmp_path / "cv.json"),
    ]) == 0
    written = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["dtype"] for r in written] == ["bfloat16", "float32"]
    assert float(written[0]["startup_ns"]) == pytest.approx(66.0)
    assert float(written[1]["ns_per_free_elem"]) == pytest.approx(PER_ELEM_NS)
    # The schema the cost model's OnChipTransferCalibration.from_csv expects.
    from triton_viz.tools.nki_cost_model import OnChipTransferCalibration
    calibration = OnChipTransferCalibration.from_csv(out)
    value, match = calibration.latency_ns("vector", "float32", 160)
    assert match == "in_domain"
    assert value == pytest.approx(STARTUP_NS + PER_ELEM_NS * 160)


def test_the_control_benchmark_stays_registered():
    from microbench.inf2_nki.harness.run_microbench import BENCHMARKS

    entry = BENCHMARKS["onchip_copy"]
    assert entry["folder"] == "engine_ops"
    work = entry["work"](p=64, f=160, repeat=8, mode="psum_to_sbuf",
                         dtype_name="bfloat16")
    assert work["partition_count"] == 64
    assert work["free_dimension_elements"] == 160
    # PSUM is float32 regardless of the SBUF dtype.
    assert work["free_bytes_per_partition"] == 160 * 4
    assert work["logical_instructions"] == 8


def test_the_control_config_is_present_and_matches_the_frozen_domain():
    from pathlib import Path

    config = json.loads(
        (Path(__file__).resolve().parents[2]
         / "microbench/inf2_nki/configs/onchip_copy_disjoint_v2.json").read_text()
    )
    bench = config["benchmarks"][0]
    assert bench["kind"] == "onchip_copy"
    assert bench["modes"] == ["psum_to_sbuf"]
    assert bench["matrix"]["f"] == [48, 160, 320]
    assert bench["matrix"]["repeat"] == [1, 4, 8, 16]
    assert sorted(bench["matrix"]["dtype"]) == ["bfloat16", "float32"]
