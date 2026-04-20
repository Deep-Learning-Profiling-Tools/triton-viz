"""Libdevice function registry.

Centralizes ``LibdeviceSpec`` definitions so that both ``core.patch`` and
``clients.symbolic_engine`` can import them without a circular dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_vectorized_erf = np.vectorize(math.erf)


def _np_erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf that preserves input dtype."""
    return _vectorized_erf(x).astype(x.dtype, copy=False)


@dataclass(frozen=True)
class LibdeviceSpec:
    name: str  # libdevice function name, e.g. "tanh"
    np_func: Callable | None  # numpy equivalent (None for builder ops)
    arity: int  # 1, 2, or 3
    sym_name: str  # symbolic engine op name, e.g. "tanh"
    builder_method: str | None = None  # interpreter_builder method name


_LIBDEVICE_REGISTRY: list[LibdeviceSpec] = [
    # Unary ops (numpy-backed)
    LibdeviceSpec("abs", np.abs, 1, "abs"),
    LibdeviceSpec("ceil", np.ceil, 1, "ceil"),
    LibdeviceSpec("cos", np.cos, 1, "cos"),
    LibdeviceSpec("exp", np.exp, 1, "exp"),
    LibdeviceSpec("exp2", np.exp2, 1, "exp2"),
    LibdeviceSpec("floor", np.floor, 1, "floor"),
    LibdeviceSpec("log", np.log, 1, "log"),
    LibdeviceSpec("log2", np.log2, 1, "log2"),
    LibdeviceSpec("sin", np.sin, 1, "sin"),
    LibdeviceSpec("sqrt", np.sqrt, 1, "sqrt"),
    LibdeviceSpec("tanh", np.tanh, 1, "tanh"),
    LibdeviceSpec("asin", np.arcsin, 1, "asin"),
    LibdeviceSpec("acos", np.arccos, 1, "acos"),
    LibdeviceSpec("erf", _np_erf, 1, "erf"),
    # Builder-backed ops (use interpreter_builder methods directly)
    LibdeviceSpec("rsqrt", None, 1, "rsqrt", builder_method="create_rsqrt"),
]

# Pre-built name → spec lookup
_REGISTERED_SPECS: dict[str, LibdeviceSpec] = {s.name: s for s in _LIBDEVICE_REGISTRY}
