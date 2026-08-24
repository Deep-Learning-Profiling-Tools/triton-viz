"""Audit source-state-conditional conserved Vector/Scalar payload routing."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triton_viz.tools.nki_evaluate_program_context_routing import (
    _ROUTING_OP_FEATURES,
    _ROUTING_REGION_FEATURES,
    _samples,
)
from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
)
from triton_viz.tools.nki_program_context import (
    PROGRAM_CONTEXT_FEATURE_NAMES,
    source_complete_routing_regime,
    source_full_routing_regime,
    source_routing_regime,
    source_routing_state,
)

ENGINES = ("vector", "scalar")
TOTAL_RIDGE_ALPHA = 0.1
ROUTING_STATES = ("canonical", "reversed", "interleaved", "blocked")
GEOMETRY_FEATURE_TERMS = (
    "free",
    "partition",
    "allocation",
    "bytes",
    "mask",
    "physical",
    "logical",
    "hbm",
    "transfer",
)


def _feature_names(mode: str = "full") -> list[str]:
    names = sorted(
        PROGRAM_CONTEXT_FEATURE_NAMES
        | _ROUTING_REGION_FEATURES
        | _ROUTING_OP_FEATURES
    )
    if mode == "full":
        return names
    if mode == "no_geometry":
        return [
            name
            for name in names
            if not any(term in name for term in GEOMETRY_FEATURE_TERMS)
        ]
    raise ValueError(f"unknown feature mode: {mode}")


def _matrix(samples: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(sample["features"].get(name, 0.0))
                for name in feature_names
            ]
            for sample in samples
        ],
        dtype=float,
    )


def _annotate_states(
    samples: list[dict],
    *,
    include_join_ownership: bool = False,
    include_local_topology: bool = False,
    include_root_orientation: bool = False,
) -> None:
    for sample in samples:
        sample["routing_state"] = (
            source_complete_routing_regime(sample["features"])
            if include_root_orientation
            else (
                source_full_routing_regime(sample["features"])
                if include_local_topology
                else (
                    source_routing_regime(sample["features"])
                    if include_join_ownership
                    else source_routing_state(sample["features"])
                )
            )
        )


def _fit_state_models(
    samples: list[dict],
    feature_names: list[str],
    *,
    include_join_ownership: bool = False,
    include_local_topology: bool = False,
    include_root_orientation: bool = False,
    ridge_alpha: float = TOTAL_RIDGE_ALPHA,
) -> dict[str, dict]:
    _annotate_states(
        samples,
        include_join_ownership=include_join_ownership,
        include_local_topology=include_local_topology,
        include_root_orientation=include_root_orientation,
    )
    models = {}
    for state in sorted({sample["routing_state"] for sample in samples}):
        selected = [
            sample for sample in samples if sample["routing_state"] == state
        ]
        if len(selected) < 2:
            continue
        x = _matrix(selected, feature_names)
        payload = np.asarray(
            [
                [sample["payloads"][engine] for engine in ENGINES]
                for sample in selected
            ],
            dtype=float,
        )
        total_model = make_pipeline(
            StandardScaler(), Ridge(alpha=ridge_alpha)
        )
        total_model.fit(x, payload.sum(axis=1))
        ratio_model = make_pipeline(
            StandardScaler(),
            (
                LinearRegression()
                if ridge_alpha == 0
                else Ridge(alpha=ridge_alpha)
            ),
        )
        ratio_model.fit(
            x,
            np.log(
                np.maximum(payload[:, 1], 1.0)
                / np.maximum(payload[:, 0], 1.0)
            ),
        )
        models[state] = {
            "total": total_model,
            "ratio": ratio_model,
            "samples": len(selected),
        }
    return models


def _evaluate(
    models: dict[str, dict],
    samples: list[dict],
    feature_names: list[str],
    label: str,
    *,
    include_join_ownership: bool = False,
    include_local_topology: bool = False,
    include_root_orientation: bool = False,
) -> dict:
    _annotate_states(
        samples,
        include_join_ownership=include_join_ownership,
        include_local_topology=include_local_topology,
        include_root_orientation=include_root_orientation,
    )
    rows = []
    ood_states = {}
    for state in sorted({sample["routing_state"] for sample in samples}):
        selected = [
            sample for sample in samples if sample["routing_state"] == state
        ]
        if state not in models:
            ood_states[state] = len(selected)
            continue
        x = _matrix(selected, feature_names)
        total = np.maximum(0.0, models[state]["total"].predict(x))
        log_ratio = np.clip(models[state]["ratio"].predict(x), -20.0, 20.0)
        scalar_share = 1.0 / (1.0 + np.exp(-log_ratio))
        predicted = np.column_stack(
            (total * (1.0 - scalar_share), total * scalar_share)
        )
        for sample, prediction in zip(selected, predicted):
            rows.append(
                {
                    "case": sample["case"],
                    "routing_state": state,
                    "actual": [
                        float(sample["payloads"][engine]) for engine in ENGINES
                    ],
                    "predicted": prediction.tolist(),
                }
            )

    reports = {}
    for index, engine in enumerate(ENGINES):
        errors = [
            abs(row["predicted"][index] - row["actual"][index])
            / max(row["actual"][index], 1.0)
            * 100.0
            for row in rows
        ]
        reports[engine] = {
            "evaluable_cases": len(errors),
            "mape_pct": statistics.mean(errors) if errors else None,
            "max_case_ape_pct": max(errors, default=None),
        }
    total_errors = [
        abs(sum(row["predicted"]) - sum(row["actual"]))
        / max(sum(row["actual"]), 1.0)
        * 100.0
        for row in rows
    ]
    return {
        "audit": label,
        "cases": len(samples),
        "evaluable_cases": len(rows),
        "coverage": len(rows) / max(1, len(samples)),
        "ood_states": ood_states,
        "total_payload_mape_pct": (
            statistics.mean(total_errors) if total_errors else None
        ),
        "engines": reports,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", nargs="+", required=True, type=Path)
    parser.add_argument("--audit-root", action="append", required=True, type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument(
        "--include-join-ownership",
        action="store_true",
        help="Condition on source phase × oriented join-ownership regime.",
    )
    parser.add_argument(
        "--include-local-topology",
        action="store_true",
        help=(
            "Condition on source phase × join ownership × branch-local "
            "fanout topology."
        ),
    )
    parser.add_argument(
        "--include-root-orientation",
        action="store_true",
        help=(
            "Condition on phase × ownership × local topology × ordered "
            "source-root primitive orientation."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("full", "no_geometry"),
        default="full",
        help=(
            "Select coefficient features. no_geometry keeps geometry for "
            "coverage but excludes it from per-regime payload regression."
        ),
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=TOTAL_RIDGE_ALPHA,
        help="Non-negative Ridge alpha used for total and log-ratio fits.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    train = _samples(args.train_root, baselines, required_engines=ENGINES)
    if not train:
        raise ValueError("no complete Vector/Scalar training controls found")
    if args.ridge_alpha < 0:
        raise ValueError("--ridge-alpha must be non-negative")
    feature_names = _feature_names(args.feature_mode)
    models = _fit_state_models(
        train,
        feature_names,
        include_join_ownership=args.include_join_ownership,
        include_local_topology=args.include_local_topology,
        include_root_orientation=args.include_root_orientation,
        ridge_alpha=args.ridge_alpha,
    )
    if (
        not args.include_join_ownership
        and not args.include_local_topology
        and not args.include_root_orientation
    ):
        missing_train_states = sorted(set(ROUTING_STATES) - set(models))
        if missing_train_states:
            raise ValueError(
                "training controls do not cover routing states: "
                f"{missing_train_states}"
            )

    folds = []
    for root in args.audit_root:
        samples = _samples([root], baselines, required_engines=ENGINES)
        if not samples:
            raise ValueError(
                f"no complete Vector/Scalar audit controls found in {root}"
            )
        folds.append(
            _evaluate(
                models,
                samples,
                feature_names,
                root.name,
                include_join_ownership=args.include_join_ownership,
                include_local_topology=args.include_local_topology,
                include_root_orientation=args.include_root_orientation,
            )
        )

    summary = {}
    for engine in ENGINES:
        values = [
            fold["engines"][engine]["mape_pct"]
            for fold in folds
            if fold["engines"][engine]["mape_pct"] is not None
        ]
        summary[engine] = {
            "mean_mape_pct": statistics.mean(values) if values else None,
            "worst_audit_mape_pct": max(values) if values else None,
            "max_case_ape_pct": max(
                (
                    fold["engines"][engine]["max_case_ape_pct"]
                    for fold in folds
                    if fold["engines"][engine]["max_case_ape_pct"] is not None
                ),
                default=None,
            ),
        }
    coverage = min(fold["coverage"] for fold in folds)
    gate = {
        "coverage_min": 1.0,
        "mean_mape_max_pct": 10.0,
        "worst_audit_mape_max_pct": 15.0,
    }
    gate["pass"] = coverage >= gate["coverage_min"] and all(
        item["mean_mape_pct"] is not None
        and item["worst_audit_mape_pct"] is not None
        and item["mean_mape_pct"] < gate["mean_mape_max_pct"]
        and item["worst_audit_mape_pct"] < gate["worst_audit_mape_max_pct"]
        for item in summary.values()
    )
    report = {
        "schema": "triton-viz.state-conditional-routing-audit-v1",
        "protocol": (
            "source-derived routing state; state-conditional conserved "
            "Vector/Scalar work; aggregate control labels only"
        ),
        "model": {
            "states": sorted(models),
            "include_join_ownership": args.include_join_ownership,
            "include_local_topology": args.include_local_topology,
            "include_root_orientation": args.include_root_orientation,
            "feature_mode": args.feature_mode,
            "ridge_alpha": args.ridge_alpha,
            "ratio_model": (
                "standardized linear regression on log Scalar/Vector"
                if args.ridge_alpha == 0
                else "standardized Ridge on log Scalar/Vector"
            ),
        },
        "train_samples": len(train),
        "train_state_samples": {
            state: models[state]["samples"] for state in sorted(models)
        },
        "feature_names": feature_names,
        "folds": folds,
        "summary": summary,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "gate": gate}, indent=2))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
