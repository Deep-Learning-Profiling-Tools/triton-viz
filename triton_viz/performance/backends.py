"""One prediction result contract, with explicitly different backend semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Prediction:
    latency_ns: float
    backend: str
    metric: str
    diagnostics: dict[str, Any]


class Backend(Protocol):
    def predict(self, source: Any) -> Prediction:
        ...


@dataclass
class NkiBackend:
    """Adapt existing NKI calibration and scheduling without changing results."""

    cost_model: Any

    def predict(self, source) -> Prediction:
        from triton_viz.tools.nki_cost_model import simulate

        result = simulate(source, self.cost_model)
        return Prediction(
            result.predicted_latency_ns,
            "nki",
            "neuroncore_latency_ns",
            result.as_dict(),
        )


@dataclass
class GpuBackend:
    calibration: dict[str, Any]
    fingerprint: str
    sm_count: int
    strict: bool = True

    def predict(self, source) -> Prediction:
        from .gpu import predict

        result = predict(
            source,
            self.calibration,
            fingerprint=self.fingerprint,
            sm_count=self.sm_count,
            strict=self.strict,
        )
        return Prediction(result["latency_us"] * 1000, "gpu", result["metric"], result)


def predict_latency(source: Any, backend: Backend) -> Prediction:
    """Predict from observed source using a calibrated, explicitly chosen backend."""
    return backend.predict(source)
