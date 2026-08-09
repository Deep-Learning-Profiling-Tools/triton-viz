"""Mechanism-level controls for launch, engine activation, and synchronization."""

from __future__ import annotations

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

from microbench.inf2_nki.common.nki_utils import dtype_for_load


def runtime_overhead_factory(
    *, p: int, f: int, mode: str, dtype_name: str = "float32", **_: object
):
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        out = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
        if mode == "empty":
            return out

        loaded = nl.load(x, dtype=kdtype, name="runtime_load")
        value = loaded
        if mode == "dma_only":
            pass
        elif mode == "vector":
            value = nisa.tensor_scalar(
                data=loaded,
                op0=nl.add,
                operand0=1.0,
                dtype=kdtype,
                engine=nisa.engine.vector,
                name="runtime_vector",
            )
        elif mode == "scalar":
            value = nisa.activation(
                np.exp,
                data=loaded,
                scale=0.001,
                dtype=kdtype,
                name="runtime_scalar",
            )
        elif mode == "vector_scalar_chain":
            vector = nisa.tensor_scalar(
                data=loaded,
                op0=nl.add,
                operand0=1.0,
                dtype=kdtype,
                engine=nisa.engine.vector,
                name="runtime_vector_producer",
            )
            value = nisa.activation(
                np.exp,
                data=vector,
                scale=0.001,
                dtype=kdtype,
                name="runtime_scalar_consumer",
            )
        elif mode == "scalar_vector_chain":
            scalar = nisa.activation(
                np.exp,
                data=loaded,
                scale=0.001,
                dtype=kdtype,
                name="runtime_scalar_producer",
            )
            value = nisa.tensor_scalar(
                data=scalar,
                op0=nl.add,
                operand0=1.0,
                dtype=kdtype,
                engine=nisa.engine.vector,
                name="runtime_vector_consumer",
            )
        else:
            raise ValueError(f"unknown runtime overhead mode: {mode}")
        nl.store(out, value, name="runtime_store")
        return out

    return kernel, [(p, f)], (1,)


def work_units(*, p: int, f: int, mode: str, **_: object) -> dict[str, int]:
    return {
        "elements": p * f,
        "program_count": 1,
        "dma_transfers": 0 if mode == "empty" else 2,
        "vector_engines": int("vector" in mode),
        "scalar_engines": int("scalar" in mode),
        "cross_engine_edges": int(mode.endswith("_chain")),
    }
