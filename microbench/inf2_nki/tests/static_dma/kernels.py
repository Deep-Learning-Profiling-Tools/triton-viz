"""Static DMA microbenchmarks for fragmented SBUF-to-SBUF transfers."""

from __future__ import annotations

import numpy as np

from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

from microbench.inf2_nki.common.nki_utils import dtype_for_load


def static_dma_scatter_factory(
    *,
    p: int,
    x: int,
    y: int,
    mode: str,
    dtype_name: str = "float32",
):
    """Create a paired baseline or scalar-scatter transpose kernel."""
    f = x * y
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(src):
        kdtype = dtype_for_load(kernel_dtype_name, src.dtype)
        out = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
        src_tile = nl.ndarray((nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=src_tile, src=src, name="seed_load")
        if mode == "hbm_roundtrip_baseline":
            nisa.dma_copy(dst=out, src=src_tile, name="result_store")
        elif mode == "sbuf_transpose_scatter":
            dst_tile = nl.ndarray((nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            for i in nl.affine_range(x):
                for j in nl.affine_range(y):
                    dst_tile[:, nl.ds(j * x + i, 1)] = nisa.tensor_copy(
                        src_tile[:, nl.ds(i * y + j, 1)]
                    )
            nisa.dma_copy(dst=out, src=dst_tile, name="result_store")
        else:
            raise ValueError(f"unknown Static DMA mode {mode!r}")
        return out

    return kernel, [(p, f)], (1,)


def work_units(
    *,
    p: int,
    x: int,
    y: int,
    mode: str,
    dtype_name: str = "float32",
    **_: object,
) -> dict[str, int]:
    itemsize = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }[dtype_name]
    tile_bytes = p * x * y * itemsize
    scatter = mode == "sbuf_transpose_scatter"
    return {
        "partition_count": p,
        "scatter_rows": x,
        "scatter_columns": y,
        "static_dma_transfer_count": x * y if scatter else 0,
        "static_dma_bytes": tile_bytes if scatter else 0,
        "free_bytes_per_partition": itemsize,
        "hbm_read_bytes": tile_bytes,
        "hbm_write_bytes": tile_bytes,
        "total_hbm_bytes": 2 * tile_bytes,
    }
