"""Freeze shape-independent controls for the three Stage-2 source structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTROL_DESIGN = {
    "schema": "triton-viz.nki-structural-controls-v1",
    "artifact_role": "control",
    "target_postcompile_allowed": False,
    "shape_protocol": {
        "development_p": [3, 31, 96],
        "development_f": [257, 769, 1281],
        "audit_p": [7, 63, 113],
        "audit_f": [385, 897, 1409],
        "rule": "offset geometries fixed independently of target benchmark shapes",
    },
    "structures": {
        "no_join_linear": {
            "development_kinds": [
                "elementwise_multiply2",
                "elementwise_maximum_masked",
                "elementwise_sigmoid_masked",
                "softmax_reduction",
            ],
            "required_source_properties": {
                "root_count": 1,
                "join_count": 0,
                "memory_only": False,
            },
            "holdout_axis": "offset geometry and independently coded equivalent motif",
        },
        "no_join_memory_only": {
            "development_kinds": ["memory_interleave_offset"],
            "required_source_properties": {
                "root_count": 2,
                "join_count": 0,
                "memory_only": True,
            },
            "holdout_axis": "layout permutation and offset geometry",
        },
        "multi_root_join": {
            "development_kinds": [
                "masked_log_reduction",
                "two_reductions",
                "two_reductions_rsqrt_masked",
            ],
            "required_source_properties": {
                "minimum_root_count": 2,
                "minimum_join_count": 1,
                "memory_only": False,
            },
            "holdout_axis": "join orientation, reduction order, and offset geometry",
        },
    },
    "fit_evidence": [
        "control Instruction.parquet",
        "control instruction_mapping.csv",
        "control Flow.parquet",
        "control DmaPacket.parquet",
        "control aggregate active-time labels",
    ],
    "evaluation": {
        "headline": "target aggregate NC-p50 MAPE",
        "per_engine": "WAPE=sum(abs(predicted-actual))/sum(actual)",
        "required_auxiliary": ["coverage", "activation", "OOD"],
    },
}


def validate_design(design: dict) -> None:
    shapes = design["shape_protocol"]
    development = set(shapes["development_p"]) | set(shapes["development_f"])
    audit = set(shapes["audit_p"]) | set(shapes["audit_f"])
    if development & audit:
        raise ValueError("development and audit geometries must be disjoint")
    if design.get("artifact_role") != "control":
        raise ValueError("structured post-compile calibration must be control-only")
    if design.get("target_postcompile_allowed") is not False:
        raise ValueError("target post-compile access must remain disabled")
    if set(design["structures"]) != {
        "no_join_linear",
        "no_join_memory_only",
        "multi_root_join",
    }:
        raise ValueError("the design must cover exactly the three source structures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_design(CONTROL_DESIGN)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(CONTROL_DESIGN, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote structural control design to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
