"""Backend-independent control-only calibration and provenance primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


def stable_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _nonnegative_fit(x, y):
    """Relative-error least squares, nonnegative coordinate descent.

    Column scaling avoids bytes drowning out startup terms. No SciPy dependency
    is needed on CPU-only modeling hosts.
    """
    weighted = x / y[:, None]
    scale = np.maximum(np.linalg.norm(weighted, axis=0), 1e-30)
    a = weighted / scale
    coefficients = np.zeros(x.shape[1])
    residual = np.ones(len(y))
    for _ in range(10000):
        previous = coefficients.copy()
        for column in range(x.shape[1]):
            vector = a[:, column]
            norm = float(vector @ vector)
            if norm == 0:
                continue
            old = coefficients[column]
            new = max(0.0, old + float(vector @ residual) / norm)
            residual -= vector * (new - old)
            coefficients[column] = new
        if np.max(np.abs(coefficients - previous)) < 1e-10:
            break
    else:
        raise ValueError("Nonnegative calibration did not converge")
    return coefficients / scale


def fit_controls(rows, feature_names, *, fingerprint, gate_pct=20.0):
    """Fit resource service costs with leave-one-geometry-group-out validation.

    Caller supplies only control rows. Artifact role is checked again here so
    accidental mixing cannot silently turn target runtimes into parameters.
    """
    if not rows or any(row.get("role") != "control" for row in rows):
        raise ValueError("Calibration accepts control rows only")
    if any(row.get("contaminated", True) for row in rows):
        raise ValueError("Calibration refuses contaminated or unaudited measurements")
    if any(row.get("fingerprint") != fingerprint for row in rows):
        raise ValueError("Control fingerprint mismatch")
    groups = sorted({str(row["cv_group"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("At least three independent geometry groups are required")
    x = np.array(
        [[row["features"][name] for name in feature_names] for row in rows], dtype=float
    )
    y = np.array([row["latency_us"] for row in rows], dtype=float)
    if (
        not np.all(np.isfinite(x))
        or np.any(x < 0)
        or not np.all(np.isfinite(y))
        or np.any(y <= 0)
    ):
        raise ValueError(
            "Calibration requires finite nonnegative features and positive times"
        )
    predictions = np.zeros(len(rows))
    labels = np.array([str(row["cv_group"]) for row in rows])
    fold_errors = {}
    for group in groups:
        test = labels == group
        predictions[test] = x[test] @ _nonnegative_fit(x[~test], y[~test])
        fold_errors[group] = float(
            np.mean(np.abs(predictions[test] / y[test] - 1)) * 100
        )
    mape = float(np.mean(np.abs(predictions / y - 1)) * 100)
    passed = mape <= gate_pct
    coefficients = _nonnegative_fit(x, y)
    return {
        "schema": "triton-viz.resource-calibration.v1",
        "fingerprint": fingerprint,
        "feature_names": list(feature_names),
        "coefficients_us": dict(zip(feature_names, coefficients.tolist())),
        "domain": {
            name: [float(x[:, i].min()), float(x[:, i].max())]
            for i, name in enumerate(feature_names)
        },
        "cv": {
            "protocol": "leave-one-geometry-group-out",
            "mape_pct": mape,
            "fold_mape_pct": fold_errors,
            "gate_pct": gate_pct,
            "passed": passed,
        },
        "control_count": len(rows),
        "control_digest": stable_digest({"rows": rows}),
    }


def price(features, calibration, *, fingerprint, strict=True):
    if calibration["fingerprint"] != fingerprint:
        raise ValueError("Calibration fingerprint mismatch; recollect controls")
    if not calibration["cv"]["passed"]:
        raise ValueError("Calibration failed its validation gate")
    reasons = []
    contributions = {}
    for name in calibration["feature_names"]:
        value = float(features[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid feature {name}: {value}")
        lo, hi = calibration["domain"][name]
        if value < lo - 1e-9 or value > hi + 1e-9:
            reasons.append(f"outside_control_domain:{name}:{value}:[{lo},{hi}]")
        contributions[name] = value * calibration["coefficients_us"][name]
    if strict and reasons:
        raise ValueError("Out-of-distribution: " + "; ".join(reasons))
    return contributions, reasons
