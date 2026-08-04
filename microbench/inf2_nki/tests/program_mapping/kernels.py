"""Program/core mapping probe kernels."""

from __future__ import annotations

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

from microbench.inf2_nki.common.nki_utils import dtype_for_load


def program_mapping_factory(*, p: int, f: int, repeat: int, mode: str, dtype_name: str = "float32"):
    if not mode.startswith("grid_"):
        raise ValueError("program_mapping mode must be grid_<positive integer>")
    programs = int(mode.removeprefix("grid_"))
    if programs < 1:
        raise ValueError("program_mapping grid must be positive")
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        pid = nl.program_id(0)
        out = nl.ndarray((programs, p, f), dtype=kdtype, buffer=nl.shared_hbm)
        tile = nl.load(x[pid], dtype=kdtype, name="load_program_slice")
        # Keep a named compute event between load/store for source-location join.
        marked = nisa.tensor_scalar(data=tile, op0=nl.add, operand0=1.0, dtype=kdtype, engine=nisa.engine.vector, name="program_marker_compute")
        nl.store(out[pid], marked, name="store_program_slice")
        return out

    return kernel, [(programs, p, f)], (programs,)
