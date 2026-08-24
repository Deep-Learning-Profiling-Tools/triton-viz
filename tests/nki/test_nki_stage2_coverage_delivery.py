import json

import pytest

from triton_viz.tools.nki_build_stage2_coverage_delivery import build_delivery


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _root(tmp_path):
    _write(
        tmp_path / "diagnostics/stage2_final_acceptance_v1.json",
        {
            "status": "latency_pass_mechanisms_incomplete",
            "gates": {},
            "full_suite": {
                "fp32": {"nc_mape_pct": 10, "dynamic_dma_mape_pct": 11},
                "bf16": {"nc_mape_pct": 12, "dynamic_dma_mape_pct": 13},
            },
        },
    )
    _write(
        tmp_path / "diagnostics/stage2_source_coverage_final_v1.json",
        {
            "cases": [
                {
                    "case": "b",
                    "split": "s1",
                    "dtype": "bfloat16",
                    "lookup_path": "none",
                    "source_region_keys": [],
                },
                {
                    "case": "a",
                    "split": "s0",
                    "dtype": "float32",
                    "lookup_path": "structural_key",
                    "source_region_keys": ["key"],
                },
            ]
        },
    )
    _write(
        tmp_path / "diagnostics/vector_scalar_transfer_stopping_evidence_v1.json",
        {
            "positive_boundary": {
                "vector_mape_pct": 2,
                "scalar_mape_pct": 3,
            },
            "negative_boundary": {"latest_scalar_mape_pct": 25},
            "stop_decision": "diagnostic only",
        },
    )
    _write(
        tmp_path / "diagnostics/gpsimd_deterministic_label_decision_v2.json",
        {
            "decision": "reject_deterministic_gpsimd_payload",
            "summary": {
                "gpsimd": {
                    "stable_activation_coverage": 0.5,
                    "mean_positive_relative_mad": 0.8,
                    "worst_positive_relative_mad": 1.0,
                }
            },
        },
    )
    _write(
        tmp_path / "diagnostics/static_dma_identifiability_decision_v2.json",
        {
            "decision": "reject_full_domain_static_dma_prediction",
            "control_summary": {"worst_activation_f1": 0},
        },
    )
    _write(
        tmp_path / "diagnostics/tensor_source_geometry_v4_cv.json",
        {
            "suite_mean_mape_pct": 8,
            "max_fold_mape_pct": 12,
        },
    )
    return tmp_path


def test_build_delivery_is_sorted_coverage_qualified_and_hashed(tmp_path):
    report = build_delivery(_root(tmp_path))

    assert report["negative_result_delivery_complete"] is True
    assert report["status"].endswith("coverage_qualified")
    assert [row["case"] for row in report["target_cases"]] == ["a", "b"]
    assert report["target_cases"][0]["static_dma"] == (
        "exact_lookup_but_full_gate_failed"
    )
    assert report["target_cases"][1]["static_dma"] == "ood_or_unmodeled"
    assert all(
        len(item["sha256"]) == 64 for item in report["evidence"].values()
    )


def test_build_delivery_rejects_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing Stage-2 evidence"):
        build_delivery(tmp_path)


def test_build_delivery_rejects_duplicate_case_rows(tmp_path):
    root = _root(tmp_path)
    path = root / "diagnostics/stage2_source_coverage_final_v1.json"
    document = json.loads(path.read_text())
    document["cases"].append(dict(document["cases"][0]))
    _write(path, document)

    with pytest.raises(ValueError, match="duplicate target coverage row"):
        build_delivery(root)
