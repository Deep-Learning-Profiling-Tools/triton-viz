"""TensorE + DMA overlap microbenchmarks."""

from __future__ import annotations

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

from microbench.inf2_nki.common.nki_utils import dtype_for_load


def tensor_dma_overlap_factory(*, m: int, k: int, n: int, f: int, repeat: int, mode: str, dtype_name: str = "float32"):
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(lhs, rhs, dma_in):
        kdtype = dtype_for_load(kernel_dtype_name, lhs.dtype)
        out = nl.ndarray((repeat, m, n), dtype=kdtype, buffer=nl.shared_hbm)
        dma_out = nl.ndarray((repeat, k, f), dtype=kdtype, buffer=nl.shared_hbm)
        rhs_s = nl.load(rhs, dtype=kdtype)
        if mode in {"independent_overlap", "tensor_only"}:
            lhs_s = nl.load(lhs, dtype=kdtype)
        if mode in {"independent_overlap", "dma_only"}:
            dma_tiles = nl.ndarray((repeat, nl.par_dim(k), f), dtype=kdtype, buffer=nl.sbuf)
        if mode in {"independent_overlap", "tensor_only"}:
            psums = nl.ndarray((repeat, nl.par_dim(m), n), dtype=nl.float32, buffer=nl.psum)

        if mode == "independent_overlap":
            for i in nl.static_range(repeat):
                dma_tiles[i] = nl.load(dma_in[i], dtype=kdtype)
                psums[i] = nisa.nc_matmul(stationary=lhs_s, moving=rhs_s, name=f"ov_mm_{i}")
            for i in nl.static_range(repeat):
                nl.store(dma_out[i], dma_tiles[i])
                sb = nisa.tensor_copy(psums[i], dtype=kdtype, engine=nisa.engine.vector, name=f"ov_evict_{i}")
                nl.store(out[i], sb)
        elif mode == "forced_serial":
            dma_tiles = nl.ndarray((repeat, nl.par_dim(k), f), dtype=kdtype, buffer=nl.sbuf)
            psums = nl.ndarray((repeat, nl.par_dim(m), n), dtype=nl.float32, buffer=nl.psum)
            for i in nl.static_range(repeat):
                dma_tiles[i] = nl.load(dma_in[i], dtype=kdtype)
                # Matmul consumes the just-loaded tile, forcing DMA -> TensorE
                # ordering while preserving the same repeated work as overlap.
                psums[i] = nisa.nc_matmul(stationary=dma_tiles[i][:, 0:m], moving=rhs_s, name=f"serial_mm_{i}")
            for i in nl.static_range(repeat):
                nl.store(dma_out[i], dma_tiles[i])
                sb = nisa.tensor_copy(psums[i], dtype=kdtype, engine=nisa.engine.vector, name=f"serial_evict_{i}")
                nl.store(out[i], sb)
        elif mode == "tensor_only":
            for i in nl.static_range(repeat):
                psums[i] = nisa.nc_matmul(stationary=lhs_s, moving=rhs_s, name=f"tensor_only_mm_{i}")
            for i in nl.static_range(repeat):
                sb = nisa.tensor_copy(psums[i], dtype=kdtype, engine=nisa.engine.vector, name=f"tensor_only_evict_{i}")
                nl.store(out[i], sb)
        elif mode == "dma_only":
            for i in nl.static_range(repeat):
                dma_tiles[i] = nl.load(dma_in[i], dtype=kdtype)
            for i in nl.static_range(repeat):
                nl.store(dma_out[i], dma_tiles[i])
        else:
            raise ValueError("unknown overlap mode")
        if mode == "tensor_only":
            return out
        if mode == "dma_only":
            return dma_out
        return out, dma_out

    return kernel, [(k, m), (k, n), (repeat, k, f)], (1,)
