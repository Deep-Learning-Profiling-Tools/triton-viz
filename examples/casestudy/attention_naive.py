"""
Thaihoa attention case study (naive): load entire K tile into SBUF at once.
We synthesize tracer records so the UI can display SBUF overflow without
actually running a Triton kernel.
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


def emit_attention_records(M=64, N=4096, D=64):
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

    load_k = Load(
        ptr=k.data_ptr(),
        offsets=np.arange(k.numel(), dtype=np.int64),
        masks=np.ones(k.numel(), dtype=bool),
    )
    load_k.mem_src = "HBM"
    load_k.mem_dst = "SBUF"
    load_k.bytes = 26 * 1024 * 1024  # intentionally exceed 24 MiB (Trn1 NC-v2)
    load_k.time_idx = time_idx
    time_idx += 1
    records.append(load_k)

    dot = Dot((M, D), (N, D), (M, N), q.numpy(), k.numpy())
    dot.mem_src = "SBUF"
    dot.mem_dst = "PSUM"
    dot.bytes = 8 * 1024 * 1024
    dot.time_idx = time_idx
    time_idx += 1
    records.append(dot)

    copy = Store(
        ptr=s.data_ptr(),
        offsets=np.arange(s.numel(), dtype=np.int64),
        masks=np.ones(s.numel(), dtype=bool),
    )
    copy.mem_src = "PSUM"
    copy.mem_dst = "SBUF"
    copy.bytes = s.numel() * 4
    copy.time_idx = time_idx
    time_idx += 1
    records.append(copy)

    store = Store(
        ptr=s.data_ptr(),
        offsets=np.arange(s.numel(), dtype=np.int64),
        masks=np.ones(s.numel(), dtype=bool),
    )
    store.mem_src = "SBUF"
    store.mem_dst = "HBM"
    store.bytes = s.numel() * 4
    store.time_idx = time_idx
    records.append(store)

    return records, [q, k, s]


if __name__ == "__main__":
    records, tensors = emit_attention_records()
    make_launch(records, tensors)
    triton_viz.launch(share=True)
