"""Shared, pre-compilation performance modeling infrastructure.

Hardware backends own lowering and scheduling semantics. Importing this package
does not initialize a GPU, import a compiler, or load calibration artifacts.
"""

from .backends import GpuBackend, NkiBackend, Prediction, predict_latency

__all__ = ["GpuBackend", "NkiBackend", "Prediction", "predict_latency"]
