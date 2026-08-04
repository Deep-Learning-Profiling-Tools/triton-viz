"""HBM latency microbenchmarks.

A true latency microbenchmark creates a loop-carried dependency so the next
memory access cannot be issued before the previous result is known. On CPUs/GPUs
this is usually pointer chasing. The ``pointer_chase_factory`` below does the
same at the NKI level: each ``nl.load(next_idx[idx])`` uses the index returned by
the previous HBM load. The measured slope is therefore an NKI-visible dependent
HBM round-trip, including dynamic indexing/DGE/compiler synchronization, not raw
DRAM cell latency.

The ``dma_roundtrip_factory`` is a complementary serialized bulk-DMA latency
benchmark. It is not pointer chasing, but it is useful for decomposing NKI-visible
DMA trigger/completion latency and minimum packet costs.
"""

from __future__ import annotations

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

from microbench.inf2_nki.common.nki_utils import dge_mode as _dge_mode, dtype_for_load


def pointer_chase_factory(*, ring_length: int, repeat: int, stride: int = 1, mode: str = "hbm_index_chain", dtype_name: str = "uint32"):
    """Return a dependent HBM pointer-chase kernel.

    ``next_idx`` is a uint32 HBM ring of shape ``(1, ring_length)``. Each hop
    loads ``next_idx[0, idx]`` where ``idx`` is the value produced by the
    previous HBM load, so hop ``i+1`` cannot be issued before hop ``i``'s result
    is known. That loop-carried dependency is what makes this a latency (not a
    bandwidth) microbenchmark. Analyze it by fitting latency slope versus the
    number of dependent hops.

    Implementation note: the dependent chain is emitted by compile-time
    *recursion*, not a Python ``for`` loop. The NKI tracer forbids reassigning a
    tile that is used as an index across a traced loop (it raises a scope error),
    and a plain Python loop over ``nl.load`` results collapses/breaks the chain.
    Recursion unrolls the chain into straight-line traced code, which is the only
    form verified to produce the correct walked index under ``nki.simulate_kernel``.
    """
    if mode != "hbm_index_chain":
        raise ValueError("pointer_chase supports only mode='hbm_index_chain'")
    if dtype_name != "uint32":
        raise ValueError("pointer_chase requires dtype='uint32'")
    if repeat < 1:
        raise ValueError("pointer_chase requires repeat >= 1")

    @nki.jit
    def kernel(next_idx):
        out = nl.ndarray((1, 1), dtype=next_idx.dtype, buffer=nl.shared_hbm)

        def hop(idx, remaining):
            # Straight-line (recursively unrolled) dependent gathers. Each load
            # indexes the ring free dimension with the previous load's result.
            if remaining == 0:
                return idx
            nxt = nl.load(next_idx[0:1, idx], name=f"ptr_chase_load_{repeat - remaining}")
            return hop(nxt, remaining - 1)

        seed = nl.load(next_idx[0:1, 0:1], name="seed_index_load")
        final = hop(seed, repeat)
        nl.store(out[0:1, 0:1], final)
        return out

    return kernel, [(1, ring_length)], (1,)


def dma_roundtrip_factory(*, p: int, f: int, repeat: int, mode: str = "serialized_roundtrip", dtype_name: str = "float32", dge_mode: str = "unknown"):
    """Return a serialized HBM<->SBUF round-trip latency kernel."""
    if mode != "serialized_roundtrip":
        raise ValueError("dma_roundtrip supports only mode='serialized_roundtrip'")
    dge = _dge_mode(dge_mode)
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        state = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
        tile_a = nl.ndarray((nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
        tile_b = nl.ndarray((nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile_a, src=x, dge_mode=dge, name="seed_load")
        for i in nl.static_range(repeat):
            # Store then dependent reload serializes completion and wakeup.
            nisa.dma_copy(dst=state, src=tile_a, dge_mode=dge, name=f"lat_store_{i}")
            nisa.dma_copy(dst=tile_b, src=state, dge_mode=dge, name=f"lat_load_{i}")
            tile_a = nisa.tensor_copy(tile_b, dtype=kdtype, engine=nisa.engine.vector, name=f"carry_{i}")
        nisa.dma_copy(dst=state, src=tile_a, dge_mode=dge, name="final_store")
        return state

    return kernel, [(p, f)], (1,)


def work_units(*, repeat: int, mode: str, p: int | None = None, f: int | None = None, dtype_name: str = "float32", **_: object) -> dict[str, int | str]:
    if mode == "hbm_index_chain":
        return {"dependent_hbm_loads": repeat + 1, "latency_metric": "slope_vs_repeat"}
    if mode == "serialized_roundtrip" and p is not None and f is not None:
        bytes_per_elem = {"float32": 4, "int32": 4, "uint32": 4, "float16": 2, "bfloat16": 2, "int8": 1}[dtype_name]
        tile_bytes = p * f * bytes_per_elem
        return {"serialized_roundtrips": repeat, "hbm_bytes_per_roundtrip": 2 * tile_bytes, "latency_metric": "slope_vs_repeat"}
    return {"latency_metric": "slope_vs_repeat"}
