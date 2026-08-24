"""Frozen control audit for conserved Vector/Scalar routing plus GpSimd hurdle.

The model predicts total compute payload, a log Scalar/Vector payload ratio,
and a separate GpSimd activation/positive-payload hurdle.  The remaining work
is allocated to Vector and Scalar, so predicted engine payload cannot exceed
the predicted total or create duplicate work.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triton_viz.tools.nki_evaluate_program_context_routing import (
    ENGINES,
    _ROUTING_OP_FEATURES,
    _ROUTING_REGION_FEATURES,
    _samples,
)
from triton_viz.tools.nki_fit_source_sequence_lowering import (
    PAYLOAD_RESOLUTION_NS,
)
from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
)
from triton_viz.tools.nki_program_context import PROGRAM_CONTEXT_FEATURE_NAMES

TOTAL_RIDGE_ALPHA = 1.0
LOG_RATIO_RIDGE_ALPHA = 0.1
GPSIMD_LOG_PAYLOAD_RIDGE_ALPHA = 1.0


def _ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


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


def _fit(train: list[dict], feature_names: list[str]) -> dict:
    x = _matrix(train, feature_names)
    payload = np.asarray(
        [[sample["payloads"][engine] for engine in ENGINES] for sample in train],
        dtype=float,
    )
    total = payload.sum(axis=1)
    total_model = _ridge(TOTAL_RIDGE_ALPHA)
    total_model.fit(x, total)

    log_ratio = np.log(
        np.maximum(payload[:, 1], 1.0) / np.maximum(payload[:, 0], 1.0)
    )
    ratio_model = _ridge(LOG_RATIO_RIDGE_ALPHA)
    ratio_model.fit(x, log_ratio)

    gpsimd_active = (payload[:, 2] > PAYLOAD_RESOLUTION_NS).astype(int)
    unique = np.unique(gpsimd_active)
    if len(unique) == 1:
        activation_model = int(unique[0])
    else:
        activation_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260824,
        )
        activation_model.fit(x, gpsimd_active)
    positive = gpsimd_active == 1
    gpsimd_model = None
    if int(np.sum(positive)) >= 2:
        gpsimd_model = _ridge(GPSIMD_LOG_PAYLOAD_RIDGE_ALPHA)
        gpsimd_model.fit(x[positive], np.log(payload[positive, 2]))
    return {
        "total": total_model,
        "ratio": ratio_model,
        "gpsimd_activation": activation_model,
        "gpsimd_payload": gpsimd_model,
    }


def _predict(
    models: dict, samples: list[dict], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(samples, feature_names)
    total = np.maximum(0.0, models["total"].predict(x))
    log_ratio = models["ratio"].predict(x)
    scalar_share = 1.0 / (1.0 + np.exp(-log_ratio))
    activation_model = models["gpsimd_activation"]
    if isinstance(activation_model, int):
        gpsimd_active = np.full(len(samples), activation_model, dtype=int)
    else:
        gpsimd_active = activation_model.predict(x).astype(int)
    gpsimd = np.zeros(len(samples), dtype=float)
    if models["gpsimd_payload"] is not None:
        gpsimd = (
            np.exp(np.clip(models["gpsimd_payload"].predict(x), -20.0, 20.0))
            * gpsimd_active
        )
    gpsimd = np.minimum(gpsimd, total)
    vector_scalar = np.maximum(0.0, total - gpsimd)
    predicted = np.column_stack(
        (
            vector_scalar * (1.0 - scalar_share),
            vector_scalar * scalar_share,
            gpsimd,
        )
    )
    active = (predicted > PAYLOAD_RESOLUTION_NS).astype(int)
    return predicted, active


def _evaluate(
    models: dict,
    samples: list[dict],
    feature_names: list[str],
    label: str,
) -> dict:
    predicted, predicted_active = _predict(models, samples, feature_names)
    actual = np.asarray(
        [[sample["payloads"][engine] for engine in ENGINES] for sample in samples],
        dtype=float,
    )
    reports = {}
    for index, engine in enumerate(ENGINES):
        actual_active = (actual[:, index] > PAYLOAD_RESOLUTION_NS).astype(int)
        positive = actual_active == 1
        errors = (
            np.abs(predicted[positive, index] - actual[positive, index])
            / actual[positive, index]
            * 100.0
        )
        reports[engine] = {
            "activation_f1": float(
                f1_score(
                    actual_active,
                    predicted_active[:, index],
                    average="binary",
                    zero_division=0,
                )
            ),
            "positive_samples": int(np.sum(positive)),
            "conditional_payload_mape_pct": (
                float(np.mean(errors)) if len(errors) else None
            ),
            "max_case_ape_pct": float(np.max(errors)) if len(errors) else None,
            "false_active": int(
                np.sum((predicted_active[:, index] == 1) & ~positive)
            ),
            "false_inactive": int(
                np.sum((predicted_active[:, index] == 0) & positive)
            ),
        }
    actual_total = actual.sum(axis=1)
    predicted_total = predicted.sum(axis=1)
    total_errors = (
        np.abs(predicted_total - actual_total)
        / np.maximum(actual_total, 1.0)
        * 100.0
    )
    return {
        "audit": label,
        "samples": len(samples),
        "total_payload_mape_pct": float(np.mean(total_errors)),
        "dominant_route_accuracy": float(
            np.mean(predicted.argmax(axis=1) == actual.argmax(axis=1))
        ),
        "engines": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", nargs="+", required=True, type=Path)
    parser.add_argument("--audit-root", action="append", required=True, type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    train = _samples(args.train_root, baselines)
    if not train:
        raise ValueError("no complete training controls found")
    feature_names = sorted(
        PROGRAM_CONTEXT_FEATURE_NAMES
        | _ROUTING_REGION_FEATURES
        | _ROUTING_OP_FEATURES
    )
    models = _fit(train, feature_names)
    folds = []
    for root in args.audit_root:
        samples = _samples([root], baselines)
        if not samples:
            raise ValueError(f"no complete audit controls found in {root}")
        folds.append(_evaluate(models, samples, feature_names, root.name))

    summary = {}
    for engine in ENGINES:
        payload_values = [
            fold["engines"][engine]["conditional_payload_mape_pct"]
            for fold in folds
            if fold["engines"][engine]["conditional_payload_mape_pct"] is not None
        ]
        summary[engine] = {
            "mean_activation_f1": statistics.mean(
                fold["engines"][engine]["activation_f1"] for fold in folds
            ),
            "worst_activation_f1": min(
                fold["engines"][engine]["activation_f1"] for fold in folds
            ),
            "mean_conditional_payload_mape_pct": (
                statistics.mean(payload_values) if payload_values else None
            ),
            "worst_audit_payload_mape_pct": (
                max(payload_values) if payload_values else None
            ),
        }
    gate = {
        "activation_f1_min": 0.95,
        "payload_mean_mape_max_pct": 10.0,
        "payload_worst_audit_mape_max_pct": 15.0,
    }
    gate["pass"] = all(
        item["mean_conditional_payload_mape_pct"] is not None
        and item["worst_audit_payload_mape_pct"] is not None
        and item["worst_activation_f1"] >= gate["activation_f1_min"]
        and item["mean_conditional_payload_mape_pct"]
        < gate["payload_mean_mape_max_pct"]
        and item["worst_audit_payload_mape_pct"]
        < gate["payload_worst_audit_mape_max_pct"]
        for item in summary.values()
    )
    report = {
        "schema": "triton-viz.log-ratio-routing-audit-v1",
        "protocol": (
            "frozen train roots to untouched audit roots; declared source "
            "features; aggregate control labels; conserved payload"
        ),
        "model": {
            "total_ridge_alpha": TOTAL_RIDGE_ALPHA,
            "log_ratio_ridge_alpha": LOG_RATIO_RIDGE_ALPHA,
            "gpsimd_log_payload_ridge_alpha": GPSIMD_LOG_PAYLOAD_RIDGE_ALPHA,
        },
        "payload_resolution_ns": PAYLOAD_RESOLUTION_NS,
        "train_samples": len(train),
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
