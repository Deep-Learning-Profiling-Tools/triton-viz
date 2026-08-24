"""Control-only audit of conserved compute payload and engine routing.

The evaluator holds out each independent factorial DAG phase in turn.  It
first predicts total compute payload, then predicts per-engine activation and
non-negative shares whose sum is constrained to one.  All features come from
declared source traces; aggregate control counters are labels only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score

from triton_viz.tools.nki_fit_source_sequence_lowering import (
    PAYLOAD_RESOLUTION_NS,
    _cases,
)
from triton_viz.tools.nki_fit_structured_controls import (
    load_runtime_engine_baselines,
)
from triton_viz.tools.nki_program_context import (
    PROGRAM_CONTEXT_FEATURE_NAMES,
    program_context_features,
)

ENGINES = ("vector", "scalar", "gpsimd")
_ROUTING_REGION_FEATURES = frozenset(
    {
        "reduction_count",
        "broadcast_edge_count",
        "partition_broadcast_input_count",
        "one_input_elementwise_count",
        "two_input_elementwise_count",
        "transcendental_count",
        "compute_mask_count",
        "free_block_count",
        "dag_branch_value_count",
        "dag_join_node_count",
        "dag_max_fanout",
        "dag_max_fanin",
        "dag_max_live_values",
        "dag_critical_path_length",
        "log2_free_dim",
        "free_dim_linear",
        "allocation_free_dim",
        "allocation_to_logical_ratio",
        "mask_or_tail",
        "has_compute_mask",
        "logical_active_partition_count",
        "log2_logical_active_partition_count",
        "token_run_count",
        "token_change_count",
        "first_special_position",
        "last_special_position",
        "special_span",
        "affine_segment_count",
        "affine_segment_total_unique_ops",
        "affine_segment_max_length",
        "affine_segment_internal_changes",
    }
)
_ROUTING_OP_FEATURES = frozenset(
    f"op_{token}"
    for token in (
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "greater",
        "where",
        "broadcast_to",
        "exp",
        "log",
        "rsqrt",
        "sqrt",
        "sigmoid",
        "reduce_sum",
        "max",
        "min",
        "mean",
    )
)


def _routing_features(features: dict[str, float]) -> dict[str, float]:
    """Keep reusable source grammar facts, not positional fingerprints."""
    return {
        name: float(value)
        for name, value in features.items()
        if name.startswith("program_")
        or name in _ROUTING_REGION_FEATURES
        or name in _ROUTING_OP_FEATURES
    }


def _samples(
    roots: list[Path],
    baselines: dict,
    required_engines: tuple[str, ...] = ENGINES,
) -> list[dict]:
    engine_rows = _cases(roots, baselines)
    by_case: dict[str, dict] = {}
    for row in engine_rows:
        sample = by_case.setdefault(
            row["case"],
            {
                "case": row["case"],
                "phase": row["domain"],
                "features": _routing_features(row["features"]),
                "payloads": {},
            },
        )
        sample["payloads"][row["engine"]] = float(row["payload_ns"])
    for root in roots:
        for trace in root.glob("control_*/trace.jsonl"):
            if trace.parent.name not in by_case:
                continue
            events = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_case[trace.parent.name]["features"].update(
                program_context_features(events)
            )
    return [
        sample
        for sample in by_case.values()
        if all(engine in sample["payloads"] for engine in required_engines)
    ]


def _classifier(x: np.ndarray, y: np.ndarray):
    unique = np.unique(y)
    if len(unique) == 1:
        return int(unique[0])
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=20260824,
    )
    model.fit(x, y)
    return model


def _classify(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, int):
        return np.full(len(x), model, dtype=int)
    return model.predict(x).astype(int)


def _regressor(x: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=20260824,
    )
    model.fit(x, y)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--runtime-overhead-results", required=True, type=Path)
    parser.add_argument(
        "--held-phase",
        action="append",
        default=[],
        help=(
            "Evaluate only this phase, training on all other supplied roots. "
            "Repeat for multiple frozen audit phases."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    baselines = load_runtime_engine_baselines(args.runtime_overhead_results)
    samples = _samples(args.roots, baselines)
    if not samples:
        raise ValueError("no complete aggregate control samples found")
    available_phases = sorted({sample["phase"] for sample in samples})
    phases = args.held_phase or available_phases
    unknown_phases = sorted(set(phases) - set(available_phases))
    if unknown_phases:
        raise ValueError(f"held phases not found in roots: {unknown_phases}")
    feature_names = sorted(
        PROGRAM_CONTEXT_FEATURE_NAMES
        | _ROUTING_REGION_FEATURES
        | _ROUTING_OP_FEATURES
    )
    folds = []
    for held_phase in phases:
        train = [sample for sample in samples if sample["phase"] != held_phase]
        test = [sample for sample in samples if sample["phase"] == held_phase]
        if not train or not test:
            raise ValueError(
                f"held phase {held_phase!r} needs non-empty train and test sets"
            )
        train_x = np.asarray(
            [
                [float(sample["features"].get(name, 0.0)) for name in feature_names]
                for sample in train
            ]
        )
        test_x = np.asarray(
            [
                [float(sample["features"].get(name, 0.0)) for name in feature_names]
                for sample in test
            ]
        )
        train_payload = np.asarray(
            [[sample["payloads"][engine] for engine in ENGINES] for sample in train]
        )
        test_payload = np.asarray(
            [[sample["payloads"][engine] for engine in ENGINES] for sample in test]
        )
        train_total = train_payload.sum(axis=1)
        test_total = test_payload.sum(axis=1)
        predicted_total = np.maximum(
            0.0, _regressor(train_x, train_total).predict(test_x)
        )

        activation_predictions = []
        raw_shares = []
        for index, _engine in enumerate(ENGINES):
            train_active = (
                train_payload[:, index] > PAYLOAD_RESOLUTION_NS
            ).astype(int)
            activation_predictions.append(
                _classify(_classifier(train_x, train_active), test_x)
            )
            train_share = np.divide(
                train_payload[:, index],
                np.maximum(train_total, 1e-9),
            )
            raw_shares.append(
                np.maximum(0.0, _regressor(train_x, train_share).predict(test_x))
            )
        predicted_active = np.asarray(activation_predictions).T
        shares = np.asarray(raw_shares).T * predicted_active
        denominators = shares.sum(axis=1)
        for row_index, denominator in enumerate(denominators):
            if denominator <= 0:
                # Preserve work conservation without inventing multi-engine
                # activity: choose the largest unmasked conditional share.
                best = int(np.argmax(np.asarray(raw_shares).T[row_index]))
                shares[row_index, best] = 1.0
            else:
                shares[row_index] /= denominator
        predicted_payload = shares * predicted_total[:, None]

        total_errors = [
            abs(predicted - actual) / actual * 100.0
            for predicted, actual in zip(predicted_total, test_total)
            if actual > PAYLOAD_RESOLUTION_NS
        ]
        dominant_actual = test_payload.argmax(axis=1)
        dominant_predicted = predicted_payload.argmax(axis=1)
        engine_reports = {}
        for index, engine in enumerate(ENGINES):
            actual_active = (
                test_payload[:, index] > PAYLOAD_RESOLUTION_NS
            ).astype(int)
            positive_errors = [
                abs(predicted - actual) / actual * 100.0
                for predicted, actual in zip(
                    predicted_payload[:, index], test_payload[:, index]
                )
                if actual > PAYLOAD_RESOLUTION_NS
            ]
            engine_reports[engine] = {
                "activation_f1": float(
                    f1_score(
                        actual_active,
                        predicted_active[:, index],
                        average="binary",
                        zero_division=0,
                    )
                ),
                "positive_samples": len(positive_errors),
                "conditional_payload_mape_pct": (
                    statistics.mean(positive_errors)
                    if positive_errors
                    else None
                ),
                "max_case_ape_pct": max(positive_errors, default=None),
                "false_active": int(
                    np.sum(
                        (predicted_active[:, index] == 1)
                        & (actual_active == 0)
                    )
                ),
                "false_inactive": int(
                    np.sum(
                        (predicted_active[:, index] == 0)
                        & (actual_active == 1)
                    )
                ),
            }
        folds.append(
            {
                "held_phase": held_phase,
                "samples": len(test),
                "dominant_route_accuracy": float(
                    np.mean(dominant_actual == dominant_predicted)
                ),
                "total_payload_mape_pct": (
                    statistics.mean(total_errors) if total_errors else None
                ),
                "engines": engine_reports,
            }
        )

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
            "worst_phase_payload_mape_pct": (
                max(payload_values) if payload_values else None
            ),
            "false_active": sum(
                fold["engines"][engine]["false_active"] for fold in folds
            ),
            "false_inactive": sum(
                fold["engines"][engine]["false_inactive"] for fold in folds
            ),
        }
    gate = {
        "activation_f1_min": 0.95,
        "payload_mean_mape_max_pct": 10.0,
        "payload_worst_phase_mape_max_pct": 15.0,
    }
    gate["pass"] = all(
        item["mean_conditional_payload_mape_pct"] is not None
        and item["worst_phase_payload_mape_pct"] is not None
        and item["worst_activation_f1"] >= gate["activation_f1_min"]
        and item["mean_conditional_payload_mape_pct"]
        < gate["payload_mean_mape_max_pct"]
        and item["worst_phase_payload_mape_pct"]
        < gate["payload_worst_phase_mape_max_pct"]
        for item in summary.values()
    )
    report = {
        "schema": "triton-viz.program-context-routing-audit-v2",
        "protocol": (
            "frozen held-factorial-phase audit; declared source trace; "
            "conserved total payload; aggregate control labels only"
        ),
        "held_phases": phases,
        "payload_resolution_ns": PAYLOAD_RESOLUTION_NS,
        "samples": len(samples),
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
