"""Structural NKI attention holdout: trace fidelity and control-only pricing."""

import importlib.util
import math
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import triton_viz
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.data import Dot, LoadTranspose
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import records_to_events

pytestmark = pytest.mark.nki

IMPL = (
    Path(__file__).resolve().parents[2]
    / "microbench"
    / "inf2_nki"
    / "holdout_operators"
    / "nki_scaled_dot_attention"
    / "impl_nki.py"
)


def _load_attention_kernel():
    spec = importlib.util.spec_from_file_location("nki_scaled_dot_attention", IMPL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nki_scaled_dot_attention_kernel


def _trace_attention(seq: int, d: int, dtype) -> tuple[np.ndarray, list[dict]]:
    triton_viz.clear()
    kernel = _load_attention_kernel()
    traced = triton_viz.trace(client=Tracer(), frontend="nki")(kernel.func)
    rng = np.random.RandomState(0)
    q = rng.randn(seq, d).astype(dtype)
    k = rng.randn(seq, d).astype(dtype)
    v = rng.randn(seq, d).astype(dtype)
    result = traced[(1,)](q, k, v)
    return np.asarray(result.data), records_to_events(launches[-1].records)


def test_attention_trace_is_numerically_correct_for_both_dtypes():
    for dtype in (np.float32, np.dtype(ml_dtypes.bfloat16)):
        seq, d = 16, 32
        result, _ = _trace_attention(seq, d, dtype)
        rng = np.random.RandomState(0)
        q = rng.randn(seq, d).astype(np.float32)
        k = rng.randn(seq, d).astype(np.float32)
        v = rng.randn(seq, d).astype(np.float32)
        scores = q @ k.T / math.sqrt(d)
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        expected = (probs @ v).astype(dtype)
        tolerance = 1e-5 if dtype == np.float32 else 0.05
        assert np.abs(result.astype(np.float32) - expected.astype(np.float32)).max() < tolerance


def test_attention_trace_records_transposed_loads_and_two_dots():
    seq, d = 64, 128
    _, events = _trace_attention(seq, d, np.float32)
    loads = [event for event in events if event["op"] == "load"]
    transpose_loads = [event for event in loads if event["dma_pattern"] == "transpose"]
    dots = [event for event in events if event["op"] == "dot"]

    # Q and K are PF-transposed during the HBM load; V is a plain copy load.
    assert len(transpose_loads) == 2
    assert len(loads) == 3
    for event in transpose_loads:
        assert event["partition_count"] == d
        assert event["free_bytes_per_partition"] == seq * 4
        assert event["dst_shape"] == [d, seq]

    # scores = Q @ K^T (seq x seq) and out = probs @ V (seq x d).
    assert len(dots) == 2
    assert dots[0]["input_shape"] == [seq, d]
    assert dots[0]["other_shape"] == [d, seq]
    assert dots[0]["flops"] == 2 * seq * seq * d
    assert dots[1]["input_shape"] == [seq, seq]
    assert dots[1]["other_shape"] == [seq, d]
    assert dots[1]["flops"] == 2 * seq * d * seq
    assert all(dot["output_dtype"] == "float32" for dot in dots)


def test_attention_dot_inputs_alias_transpose_load_storage():
    """The QK^T Dot must read the exact SBUF tiles produced by the loads."""
    triton_viz.clear()
    kernel = _load_attention_kernel()
    traced = triton_viz.trace(client=Tracer(), frontend="nki")(kernel.func)
    rng = np.random.RandomState(0)
    traced[(1,)](
        rng.randn(32, 32).astype(np.float32),
        rng.randn(32, 32).astype(np.float32),
        rng.randn(32, 32).astype(np.float32),
    )
    records = launches[-1].records
    transpose_loads = [
        record for record in records if isinstance(record, LoadTranspose)
    ]
    dots = [record for record in records if isinstance(record, Dot)]
    load_storages = {record.dst_storage for record in transpose_loads}
    assert set(dots[0].input_storages) <= load_storages
