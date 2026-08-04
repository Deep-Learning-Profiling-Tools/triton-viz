"""DMA bandwidth kernels.

These kernels are bandwidth microbenchmarks: they issue large and/or repeated
bulk transfers to saturate the HBM<->SBUF DMA path. They should be analyzed via
bytes/time and DmaPacket throughput, not as latency tests.
"""

from __future__ import annotations

import numpy as np

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

from microbench.inf2_nki.common.nki_utils import dge_mode as _dge_mode, dtype_for_load


def bulk_copy_factory(*, p: int, f: int, repeat: int, mode: str, dtype_name: str = "float32", dge_mode: str = "unknown", programs: int = 1, placement: str = "serial"):
    """Return a bulk DMA bandwidth kernel.

    Modes:
    - ``hbm_to_sbuf_stream``: repeated independent HBM->SBUF loads plus one final store.
    - ``sbuf_to_hbm_stream``: one seed load then repeated SBUF->HBM stores.
    - ``roundtrip_stream``: repeated HBM->SBUF followed by SBUF->HBM, optimized for total bytes/s.
    """
    dge = _dge_mode(dge_mode)
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        pid = nl.program_id(0)
        if mode == "hbm_to_sbuf_stream":
            # Reduce each loaded tile to one value per partition. This makes
            # every read observable while adding only a tiny HBM write, unlike
            # storing every full tile back to HBM (which measures roundtrip
            # traffic rather than directional read bandwidth).
            out = nl.ndarray((programs, repeat, p, 1), dtype=kdtype, buffer=nl.shared_hbm)
            tiles = nl.ndarray((repeat, nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            summaries = nl.ndarray((repeat, nl.par_dim(p), 1), dtype=kdtype, buffer=nl.sbuf)
            for i in nl.static_range(repeat):
                nisa.dma_copy(dst=tiles[i], src=x[pid, i], dge_mode=dge, name=f"bw_load_{i}")
            for i in nl.static_range(repeat):
                nisa.activation(
                    np.copy,
                    data=tiles[i],
                    reduce_op=nl.add,
                    reduce_res=summaries[i],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    dtype=kdtype,
                    name=f"keepalive_reduce_{i}",
                )
                nisa.dma_copy(dst=out[pid, i], src=summaries[i], dge_mode=dge, name=f"keepalive_store_{i}")
        elif mode == "sbuf_to_hbm_stream":
            out = nl.ndarray((programs, repeat, p, f), dtype=kdtype, buffer=nl.shared_hbm)
            tile = nl.ndarray((nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile, src=x[pid, 0], dge_mode=dge, name="seed_load")
            for i in nl.static_range(repeat):
                nisa.dma_copy(dst=out[pid, i], src=tile, dge_mode=dge, name=f"bw_store_{i}")
        elif mode == "roundtrip_stream":
            out = nl.ndarray((programs, repeat, p, f), dtype=kdtype, buffer=nl.shared_hbm)
            tiles = nl.ndarray((repeat, nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            for i in nl.static_range(repeat):
                nisa.dma_copy(dst=tiles[i], src=x[pid, i], dge_mode=dge, name=f"rt_load_{i}")
                nisa.dma_copy(dst=out[pid, i], src=tiles[i], dge_mode=dge, name=f"rt_store_{i}")
        else:
            raise ValueError("unknown bandwidth_dma mode")
        return out

    if placement == "serial":
        grid_dim = programs
    elif placement == "logical_nc":
        if programs < 1 or programs % 2:
            raise ValueError("logical_nc placement requires a positive even program count")
        grid_dim = nl.nc(2) if programs == 2 else nl.nc(2) * (programs // 2)
    else:
        raise ValueError("placement must be 'serial' or 'logical_nc'")
    return kernel, [(programs, repeat, p, f)], (grid_dim,)


def work_bytes(*, p: int, f: int, repeat: int, mode: str, dtype_name: str = "float32", programs: int = 1, **_: object) -> dict[str, int]:
    bytes_per_elem = {"float32": 4, "int32": 4, "uint32": 4, "float16": 2, "bfloat16": 2, "int8": 1}[dtype_name]
    tile_bytes = p * f * bytes_per_elem
    active_engines = min(16, p)
    geometry = {
        "partition_count": p,
        "free_dimension_elements": f,
        "free_bytes_per_partition": f * bytes_per_elem,
        "dma_engines_expected": active_engines,
        "partitions_per_dma_engine": (p + active_engines - 1) // active_engines,
    }
    if mode == "hbm_to_sbuf_stream":
        keepalive_bytes = programs * repeat * p * bytes_per_elem
        read_bytes = programs * repeat * tile_bytes
        return {**geometry, "hbm_read_bytes": read_bytes, "hbm_write_bytes": keepalive_bytes, "total_hbm_bytes": read_bytes + keepalive_bytes}
    if mode == "sbuf_to_hbm_stream":
        return {**geometry, "hbm_read_bytes": programs * tile_bytes, "hbm_write_bytes": programs * repeat * tile_bytes, "total_hbm_bytes": programs * (repeat + 1) * tile_bytes}
    if mode == "roundtrip_stream":
        return {**geometry, "hbm_read_bytes": programs * repeat * tile_bytes, "hbm_write_bytes": programs * repeat * tile_bytes, "total_hbm_bytes": programs * 2 * repeat * tile_bytes}
    return {**geometry, "total_hbm_bytes": 0}


def transpose_copy_factory(*, p: int, f: int, repeat: int = 1, mode: str,
                           dtype_name: str = "float32"):
    """Return one HBM[f,p] -> SBUF[p,f] DMA-transpose benchmark."""
    if mode != "hbm_to_sbuf_transpose":
        raise ValueError("unknown DMA transpose mode")
    if repeat != 1:
        raise ValueError("DMA transpose surface requires repeat=1 to isolate one transfer")
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        transposed = nisa.dma_transpose(src=x[0])
        summary = nl.ndarray((nl.par_dim(p), 1), dtype=kdtype, buffer=nl.sbuf)
        out = nl.ndarray((p, 1), dtype=kdtype, buffer=nl.shared_hbm)
        nisa.activation(
            np.copy,
            data=transposed,
            reduce_op=nl.add,
            reduce_res=summary,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=kdtype,
            name="transpose_keepalive_reduce",
        )
        nisa.dma_copy(dst=out, src=summary, name="transpose_keepalive_store")
        return out

    return kernel, [(1, f, p)], (1,)


def transpose_work_bytes(*, p: int, f: int, dtype_name: str = "float32", **_: object) -> dict[str, int]:
    itemsize = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype_name]
    read_bytes = p * f * itemsize
    return {
        "partition_count": p,
        "free_dimension_elements": f,
        "free_bytes_per_partition": f * itemsize,
        "hbm_minor_dimension_elements": p,
        "transpose": True,
        "hbm_read_bytes": read_bytes,
        "hbm_write_bytes": p * itemsize,
        "total_hbm_bytes": read_bytes + p * itemsize,
    }


def transpose_pipeline_factory(*, p: int, f: int, repeat: int = 1, mode: str,
                               dtype_name: str = "float32"):
    """Controls for DMA-transpose to dependent-store handoff cost."""
    if repeat != 1:
        raise ValueError("transpose pipeline controls require repeat=1")
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        if mode == "transpose_only":
            tile = nisa.dma_transpose(src=x[0], name="measured_transpose")
            summary = nl.ndarray((nl.par_dim(p), 1), dtype=kdtype, buffer=nl.sbuf)
            out = nl.ndarray((p, 1), dtype=kdtype, buffer=nl.shared_hbm)
            nisa.activation(
                np.copy, data=tile, reduce_op=nl.add, reduce_res=summary,
                reduce_cmd=nisa.reduce_cmd.reset_reduce, dtype=kdtype,
                name="transpose_keepalive_reduce",
            )
            nisa.dma_copy(dst=out, src=summary, name="transpose_keepalive_store")
        elif mode == "store_only":
            # SBUF initialization is not DMA work; the full output store is the
            # only dynamic HBM packet under measurement.
            tile = nl.full((nl.par_dim(p), f), fill_value=1.0,
                           dtype=kdtype, buffer=nl.sbuf)
            out = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=tile, name="measured_store")
        elif mode == "transpose_then_store":
            tile = nisa.dma_transpose(src=x[0], name="measured_transpose")
            out = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=tile, name="dependent_store")
        else:
            raise ValueError("unknown transpose pipeline mode")
        return out

    return kernel, [(1, f, p)], (1,)


def transpose_pipeline_work_bytes(*, p: int, f: int, mode: str,
                                  dtype_name: str = "float32", **_: object) -> dict[str, int]:
    itemsize = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype_name]
    full = p * f * itemsize
    read = full if mode in ("transpose_only", "transpose_then_store") else 0
    write = p * itemsize if mode == "transpose_only" else full
    return {
        "partition_count": p,
        "free_dimension_elements": f,
        "free_bytes_per_partition": f * itemsize,
        "hbm_read_bytes": read,
        "hbm_write_bytes": write,
        "total_hbm_bytes": read + write,
        "dependent_transpose_store": mode == "transpose_then_store",
    }
