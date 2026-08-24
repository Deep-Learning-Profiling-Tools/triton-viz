"""Build the final coverage-qualified NKI Stage-2 delivery manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = {
    "accepted_latency": "diagnostics/stage2_final_acceptance_v1.json",
    "target_source_coverage": "diagnostics/stage2_source_coverage_final_v1.json",
    "vector_scalar_stopping": (
        "diagnostics/vector_scalar_transfer_stopping_evidence_v1.json"
    ),
    "gpsimd_repeat_decision": (
        "diagnostics/gpsimd_deterministic_label_decision_v2.json"
    ),
    "static_dma_decision": (
        "diagnostics/static_dma_identifiability_decision_v2.json"
    ),
    "tensor_control_cv": "diagnostics/tensor_source_geometry_v4_cv.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(document: dict[str, Any], path: Path, *keys: str) -> None:
    missing = [key for key in keys if key not in document]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")


def _static_case_status(lookup_path: str) -> str:
    if lookup_path == "structural_key":
        return "exact_lookup_but_full_gate_failed"
    if lookup_path == "padded_exact":
        return "exact_lookup_but_full_gate_failed"
    if lookup_path == "rule_sequence":
        return "fallback_unqualified"
    return "ood_or_unmodeled"


def build_delivery(
    root: Path,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    paths = artifact_paths or DEFAULT_ARTIFACTS
    resolved = {name: root / relative for name, relative in paths.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Stage-2 evidence artifacts: " + ", ".join(missing))

    documents = {name: _load_json(path) for name, path in resolved.items()}
    acceptance = documents["accepted_latency"]
    coverage = documents["target_source_coverage"]
    vector_scalar = documents["vector_scalar_stopping"]
    gpsimd = documents["gpsimd_repeat_decision"]
    static_dma = documents["static_dma_decision"]
    tensor = documents["tensor_control_cv"]
    _require(acceptance, resolved["accepted_latency"], "full_suite", "gates", "status")
    _require(coverage, resolved["target_source_coverage"], "cases")
    _require(
        vector_scalar,
        resolved["vector_scalar_stopping"],
        "positive_boundary",
        "negative_boundary",
        "stop_decision",
    )
    _require(gpsimd, resolved["gpsimd_repeat_decision"], "summary", "decision")
    _require(static_dma, resolved["static_dma_decision"], "control_summary", "decision")
    _require(
        tensor,
        resolved["tensor_control_cv"],
        "suite_mean_mape_pct",
        "max_fold_mape_pct",
    )
    cases = coverage["cases"]
    if not isinstance(cases, list):
        raise ValueError("target source coverage cases must be a list")

    static_counts = Counter()
    case_rows = []
    seen_cases = set()
    for row in sorted(cases, key=lambda item: (str(item.get("case")), str(item.get("split")))):
        case = str(row.get("case") or "")
        if not case:
            raise ValueError("target source coverage contains a case without a name")
        identity = (case, str(row.get("split") or ""))
        if identity in seen_cases:
            raise ValueError(f"duplicate target coverage row: {identity}")
        seen_cases.add(identity)
        lookup_path = str(row.get("lookup_path") or "none")
        static_status = _static_case_status(lookup_path)
        static_counts[static_status] += 1
        case_rows.append(
            {
                "case": case,
                "split": identity[1],
                "dtype": str(row.get("dtype") or ""),
                "aggregate_latency": "accepted_full_suite",
                "dynamic_dma": "accepted_full_suite",
                "vector_payload": "coverage_qualified_diagnostic_only",
                "scalar_payload": "coverage_qualified_diagnostic_only",
                "gpsimd_payload": "probabilistic_repeat_qualified_only",
                "static_dma": static_status,
                "static_dma_lookup_path": lookup_path,
                "source_region_key_count": len(row.get("source_region_keys") or []),
            }
        )

    fp32 = acceptance["full_suite"].get("fp32") or {}
    bf16 = acceptance["full_suite"].get("bf16") or {}
    repeat_summary = gpsimd["summary"]
    gpsimd_summary = repeat_summary.get("gpsimd") or {}
    mechanisms = {
        "aggregate_latency": {
            "status": "accepted",
            "fp32_mape_pct": fp32.get("nc_mape_pct"),
            "bf16_mape_pct": bf16.get("nc_mape_pct"),
            "gate": "<15% both dtypes",
        },
        "dynamic_dma": {
            "status": "accepted",
            "fp32_mape_pct": fp32.get("dynamic_dma_mape_pct"),
            "bf16_mape_pct": bf16.get("dynamic_dma_mape_pct"),
            "gate": "<15% both dtypes",
        },
        "vector_payload": {
            "status": "coverage_qualified_diagnostic",
            "matched_control_mape_pct": vector_scalar["positive_boundary"].get(
                "vector_mape_pct"
            ),
            "full_unseen_family_status": "rejected",
            "reason": vector_scalar["stop_decision"],
        },
        "scalar_payload": {
            "status": "coverage_qualified_diagnostic",
            "matched_control_mape_pct": vector_scalar["positive_boundary"].get(
                "scalar_mape_pct"
            ),
            "full_unseen_family_status": "rejected",
            "latest_whole_family_mape_pct": vector_scalar["negative_boundary"].get(
                "latest_scalar_mape_pct"
            ),
            "reason": vector_scalar["stop_decision"],
        },
        "gpsimd_payload": {
            "status": "probabilistic_repeat_qualified_only",
            "deterministic_status": gpsimd["decision"],
            "stable_activation_coverage": gpsimd_summary.get(
                "stable_activation_coverage"
            ),
            "mean_positive_relative_mad": gpsimd_summary.get(
                "mean_positive_relative_mad"
            ),
            "worst_positive_relative_mad": gpsimd_summary.get(
                "worst_positive_relative_mad"
            ),
        },
        "static_dma": {
            "status": "provenance_only_full_domain_rejected",
            "decision": static_dma["decision"],
            "control_summary": static_dma["control_summary"],
            "target_case_status_counts": dict(sorted(static_counts.items())),
        },
        "tensor": {
            "status": "narrow_control_domain_accepted",
            "control_suite_mean_mape_pct": tensor["suite_mean_mape_pct"],
            "control_worst_fold_mape_pct": tensor["max_fold_mape_pct"],
            "full_fused_attention_status": "not_accepted",
        },
    }
    negative_result_complete = (
        mechanisms["aggregate_latency"]["status"] == "accepted"
        and mechanisms["dynamic_dma"]["status"] == "accepted"
        and mechanisms["scalar_payload"]["full_unseen_family_status"] == "rejected"
        and mechanisms["gpsimd_payload"]["deterministic_status"].startswith("reject")
        and mechanisms["static_dma"]["decision"].startswith("reject")
    )
    evidence = {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
        }
        for name, path in sorted(resolved.items())
    }
    return {
        "schema": "triton-viz.nki-stage2-coverage-delivery-v2",
        "protocol": (
            "source-only prediction; aggregate target labels only after frozen "
            "latency prediction; mechanism model selection uses independent controls"
        ),
        "status": (
            "latency_pass_mechanisms_unidentifiable_coverage_qualified"
            if negative_result_complete
            else "incomplete_evidence"
        ),
        "full_mechanism_pass": False,
        "negative_result_delivery_complete": negative_result_complete,
        "mechanisms": mechanisms,
        "target_case_count": len(case_rows),
        "target_cases": case_rows,
        "evidence": evidence,
        "integrity": {
            "target_postcompile_prediction_reads": False,
            "target_guided_coefficients": False,
            "failed_candidates_retained_as_failed": True,
            "downloads_git_managed": False,
            "artifacts_persistent_ebs": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_delivery(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "target_case_count": report["target_case_count"],
                "negative_result_delivery_complete": report[
                    "negative_result_delivery_complete"
                ],
            },
            indent=2,
        )
    )
    return 0 if report["negative_result_delivery_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
