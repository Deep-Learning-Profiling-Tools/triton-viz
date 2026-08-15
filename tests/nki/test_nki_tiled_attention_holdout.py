"""Structural NKI beta2 tiled-attention holdout: trace coverage and no leakage."""

import importlib.util
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import triton_viz
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import write_jsonl

pytestmark = pytest.mark.nki

EXAMPLE = Path("examples/nki_beta2/tiled_attention.py").resolve()


def _load_example():
    spec = importlib.util.spec_from_file_location("tiled_attention", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(m_size=128, dv_size=64):
    module = _load_example()
    batch = num_heads = num_heads_k = num_heads_v = 1
    d_size = n_size = 128
    rng = np.random.RandomState(0)
    q = rng.randn(batch, num_heads, m_size, d_size).astype(np.float32)
    k = rng.randn(batch, num_heads_k, n_size, d_size).astype(np.float32)
    v = rng.randn(batch, num_heads_v, n_size, dv_size).astype(np.float32)
    out = np.zeros((batch, num_heads, m_size, dv_size), np.float32)
    args = (
        q, k, v, out,
        batch, num_heads, num_heads_k, num_heads_v,
        m_size, n_size, d_size, dv_size,
    )
    triton_viz.clear()
    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(
        module.tiled_attention_kernel
    )
    traced[(1,)](*args)
    events = write_jsonl(
        launches[-1].records,
        Path(tempfile.mkdtemp()) / "tiled_attention_trace.jsonl",
    )
    return module, args, events


def test_tiled_attention_trace_covers_full_attention_dataflow():
    module, args, events = _trace()
    q, k, v, out = args[:4]
    counts = Counter(event.get("op") for event in events)
    assert counts["dot"] == 2
    assert counts["tensor_transpose"] == 3
    assert counts["reduce_sum"] == 2
    assert counts["compute"] >= 6
    assert counts["transfer"] >= 10
    assert np.allclose(
        out,
        module._numpy_tiled_attention(q, k, v),
        rtol=1e-5,
        atol=1e-5,
    )
    transposes = [event for event in events if event["op"] == "tensor_transpose"]
    assert all(event["engine"] == "tensor" for event in transposes)
    assert all(event["flops"] == 2 * 128 * 128 * 128 for event in transposes)

    # The online-softmax region must classify with the existing
    # reduction-with-transcendental grammar once the beta2 primitive names are
    # canonicalized (tensor_scalar->multiply, activation->exp, tensor_reduce->
    # max/reduce_sum). It is one fusion group, not a fragmented softmax.
    from triton_viz.tools.nki_region_ir import structural_family

    softmax_regions = [
        event["region_ir"]
        for event in events
        if event.get("region_ir")
        and int(event["region_ir"].get("reduction_count", 0)) >= 2
    ]
    assert len({id(region) for region in softmax_regions}) == 1
    family = structural_family(softmax_regions[0])
    assert family.startswith("reduction_transcendental")


def test_attention_dot_shapes_have_no_shape_keyed_calibration_lookup(tmp_path):
    """The TensorE calibration must never expose a per-shape lookup table."""
    from triton_viz.tools.nki_cost_model import TensorCalibrationSurface

    calibration = TensorCalibrationSurface(
        points={"float32": (20_000.0, 1000.0)},
        flops_domain={"float32": (5e8, 4e9)},
    )
    assert not hasattr(calibration, "ns_per_dot")
    assert not hasattr(calibration, "shape_points")
    # The two matmul shapes below are exactly the attention Dot geometries;
    # pricing is FLOPs-only and there is no (dtype, m, n) table to hit.
    assert calibration.flops_per_ns("float32") == pytest.approx(20_000.0)
