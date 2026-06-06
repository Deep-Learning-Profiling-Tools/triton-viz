from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Profiler": ("triton_viz.clients.profiler.profiler", "Profiler"),
    "LoadStoreBytes": ("triton_viz.clients.profiler.data", "LoadStoreBytes"),
    "OpTypeCounts": ("triton_viz.clients.profiler.data", "OpTypeCounts"),
    "RaceDetector": ("triton_viz.clients.race_detector.race_detector", "RaceDetector"),
    "Sanitizer": ("triton_viz.clients.sanitizer.sanitizer", "Sanitizer"),
    "OutOfBoundsRecord": ("triton_viz.clients.sanitizer.data", "OutOfBoundsRecord"),
    "SymbolicExpr": ("triton_viz.clients.symbolic_engine", "SymbolicExpr"),
    "SymbolicClient": ("triton_viz.clients.symbolic_engine", "SymbolicClient"),
    "RangeWrapper": ("triton_viz.clients.symbolic_engine", "RangeWrapper"),
    "Tracer": ("triton_viz.clients.tracer.tracer", "Tracer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
