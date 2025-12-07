"""
Thaihoa attention case study (tiled): load K in smaller chunks to stay within SBUF.
We synthesize tracer records so the UI can display SBUF usage.
"""

from __future__ import annotations

import numpy as np
import torch

import triton_viz
from triton_viz.core.data import Grid, Load, Store, Dot
from triton_viz.core.trace import launches


def make_launch(records, tensors):
    launches.clear()

    class Launch:
        def __init__(self, t):
            self.records = records
            self.tensors = t

    launches.append(Launch(tensors))


def emit_tiled_records(M=64, N=1024, D=64, TILE_N=64, max_tiles=4):
    q = torch.randn((M, D), dtype=torch.float32)
    k = torch.randn((N, D), dtype=torch.float32)
    s = torch.empty((M, N), dtype=torch.float32)

    records = [Grid(idx=(0, 0, 0))]
    time_idx = 0

    load_q = Load(
        ptr=q.data_ptr(),
        offsets=np.arange(q.numel(), dtype=np.int64),
        masks=np.ones(q.numel(), dtype=bool),
    )
    load_q.mem_src = "HBM"
    load_q.mem_dst = "SBUF"
    load_q.bytes = q.numel() * 4
    load_q.time_idx = time_idx
    time_idx += 1
    records.append(load_q)

    tiles_emitted = 0
    for tile_start in range(0, N, TILE_N):
        if tiles_emitted >= max_tiles:
            break
        tile_size = min(TILE_N, N - tile_start)
        load_k = Load(
            ptr=k.data_ptr(),
            offsets=np.arange(tile_size * D, dtype=np.int64) + tile_start * D,
            masks=np.ones(tile_size * D, dtype=bool),
        )
        load_k.mem_src = "HBM"
        load_k.mem_dst = "SBUF"
        load_k.bytes = 3 * 1024 * 1024  # small chunk to stay under 24 MiB
        load_k.time_idx = time_idx
        time_idx += 1
        records.append(load_k)

        dot = Dot((M, D), (tile_size, D), (M, tile_size), q.numpy(), k[tile_start : tile_start + tile_size].numpy())
        dot.mem_src = "SBUF"
        dot.mem_dst = "PSUM"
        dot.bytes = 3 * 1024 * 1024
        dot.time_idx = time_idx
        time_idx += 1
        records.append(dot)

        copy = Store(
            ptr=s.data_ptr(),
            offsets=np.arange(M * tile_size, dtype=np.int64) + tile_start * M,
            masks=np.ones(M * tile_size, dtype=bool),
        )
        copy.mem_src = "PSUM"
        copy.mem_dst = "SBUF"
        copy.bytes = M * tile_size * 4
        copy.time_idx = time_idx
        time_idx += 1
        records.append(copy)

        store = Store(
            ptr=s.data_ptr(),
            offsets=np.arange(M * tile_size, dtype=np.int64) + tile_start * M,
            masks=np.ones(M * tile_size, dtype=bool),
        )
        store.mem_src = "SBUF"
        store.mem_dst = "HBM"
        store.bytes = 3 * 1024 * 1024
        store.time_idx = time_idx
        time_idx += 1
        records.append(store)
        tiles_emitted += 1

    return records, [q, k, s]


if __name__ == "__main__":
    records, tensors = emit_tiled_records()
    make_launch(records, tensors)
    triton_viz.launch(share=True)

