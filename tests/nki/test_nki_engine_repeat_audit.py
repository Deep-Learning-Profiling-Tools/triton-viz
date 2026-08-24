import csv

import pytest

from triton_viz.tools.nki_audit_engine_repeats import audit_repeats


def _write_root(root, cases):
    root.mkdir()
    fields = (
        "case",
        "dtype",
        "p",
        "vector_active_ns",
        "scalar_active_ns",
        "gpsimd_active_ns",
    )
    with (root / "control_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)


def test_repeat_audit_reports_stability_and_mad(tmp_path):
    base = {
        "case": "control_a",
        "dtype": "float32",
        "p": 64,
        "vector_active_ns": 100,
        "scalar_active_ns": 80,
        "gpsimd_active_ns": 5,
    }
    roots = []
    for index, vector in enumerate((100, 102, 98)):
        root = tmp_path / f"rep{index}"
        _write_root(root, [{**base, "vector_active_ns": vector}])
        roots.append(root)

    report = audit_repeats(
        roots,
        {},
        payload_resolution_ns=10,
        stable_coverage_min=0.95,
        relative_mad_max=0.10,
    )

    assert report["summary"]["vector"]["stable_activation_coverage"] == 1
    assert report["summary"]["vector"]["mean_positive_relative_mad"] == 0.02
    assert report["summary"]["vector"]["pass"] is True
    assert report["summary"]["gpsimd"]["stable_activation_coverage"] == 1
    assert report["summary"]["gpsimd"]["pass"] is False


def test_repeat_audit_rejects_mismatched_case_sets(tmp_path):
    roots = []
    for index, case in enumerate(("a", "a", "b")):
        root = tmp_path / f"rep{index}"
        _write_root(
            root,
            [
                {
                    "case": case,
                    "dtype": "float32",
                    "p": 1,
                    "vector_active_ns": 100,
                    "scalar_active_ns": 100,
                    "gpsimd_active_ns": 100,
                }
            ],
        )
        roots.append(root)

    with pytest.raises(ValueError, match="identical case sets"):
        audit_repeats(
            roots,
            {},
            payload_resolution_ns=10,
            stable_coverage_min=0.95,
            relative_mad_max=0.10,
        )
