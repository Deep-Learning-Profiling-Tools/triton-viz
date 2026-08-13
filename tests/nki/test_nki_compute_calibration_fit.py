import csv

import pytest

from microbench.inf2_nki.profile_parser.fit_compute_calibration import fit_rows


def _row(run_id, free_dim, count, active_ns):
    return {
        "run_id": run_id,
        "kind": "vector_add",
        "status": "ok",
        "spec.dtype": "float32",
        "spec.f": free_dim,
        "work.free_dimension_elements": free_dim,
        "work.input_stream_count": 2,
        "profile.vector_engine_active_time": active_ns / 1e9,
        "profile.vector_engine_instruction_count": count,
    }


def test_compute_fit_does_not_mix_same_kind_from_non_calibration_suites(tmp_path):
    path = tmp_path / "all_results.csv"
    rows = [
        _row("engine_lowering_sweep", 128, 36, 36 * (50 + 0.5 * 128)),
        _row("engine_lowering_sweep", 512, 36, 36 * (50 + 0.5 * 512)),
        _row("additional_coverage", 2048, 21, 21 * 900),
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    fitted = fit_rows(path, {"engine_lowering_sweep"})
    assert len(fitted) == 1
    assert fitted[0]["instruction_count_min"] == 36
    assert fitted[0]["instruction_count_max"] == 36
    assert fitted[0]["startup_ns"] == pytest.approx(50)
    assert fitted[0]["ns_per_free_elem"] == pytest.approx(0.5)
    assert fitted[0]["run_ids"] == "engine_lowering_sweep"

    assert fit_rows(path) == []


def test_compute_fit_excludes_minority_lowering_branch_and_reports_it(tmp_path):
    path = tmp_path / "all_results.csv"
    rows = [
        _row("engine_lowering_sweep", 128, 36, 36 * (50 + 0.5 * 128)),
        _row("engine_lowering_sweep", 512, 33, 33 * (90 + 0.2 * 512)),
        _row("engine_lowering_sweep", 1024, 36, 36 * (50 + 0.5 * 1024)),
        _row("engine_lowering_sweep", 2048, 36, 36 * (50 + 0.5 * 2048)),
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    fitted = fit_rows(path, {"engine_lowering_sweep"})
    assert len(fitted) == 1
    assert fitted[0]["points"] == 3
    assert fitted[0]["excluded_branch_points"] == 1
    assert fitted[0]["startup_ns"] == pytest.approx(50)
    assert fitted[0]["ns_per_free_elem"] == pytest.approx(0.5)
