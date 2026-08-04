"""Small NKI-side utility helpers shared by microbench kernels."""

from __future__ import annotations

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


def dtype_for_load(dtype_name: str, fallback):
    return getattr(nl, dtype_name) if dtype_name == "bfloat16" else fallback


def dge_mode(name: str):
    try:
        return getattr(nisa.dge_mode, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown dge_mode {name!r}") from exc
