"""One-time GPU launch capture for the flash-linear-attention corpus.

fla-core is analyzed AS INSTALLED (pip, like liger) — kernels are NOT
vendored; ``runner._fla_provenance()`` pins the exact version + upstream
commit in every results header. This module drives the public ``fla.ops``
entry points (one small-fp32 case per family × variant, forward and —
where supported — backward, plus a varlen twin) under the shared
``JITFunction.run`` hook (capture_common.py) and records every kernel's
FIRST real launch. Autotune is left ON: benchmark launches are real
launches, so the first config in the sweep is captured — the harness
only consumes signature/constexprs/grid/args, never num_warps.

Small int/bool tensors carry exact value snapshots (cu_seqlens must stay
monotone, chunk index tables must stay coupled to it — by-range randint
rebuilds would fabricate invalid inputs).

Cross-case duplicates (families share fla/ops/common and fla/ops/utils
kernels) are dropped at merge time when the full specialization
fingerprint matches; each case records what it dropped.

Usage (GPU machine):
    uv run python -m evaluation.fla_capture                # all cases
    uv run python -m evaluation.fla_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SPECS_PATH = Path(__file__).parent / "kernels" / "fla_specs.json"
PER_CASE_TIMEOUT_S = 600
UPSTREAM = "https://github.com/fla-org/flash-linear-attention"


# ── case table ───────────────────────────────────────────────────
# Each case: (family, bwd, run) where run(torch, device, dtype) builds
# small inputs, calls one public fla.ops entry point, and returns the
# output tensors (summed + .backward()'d by the driver when bwd).
# Shapes default to B=2, T=64, H=2, K=V=64 (varlen: packed [1, 64, ...]
# with cu_seqlens [0, 29, 64]); constructions follow fla's own tests.


def _abc_chunk(torch, device, dtype):
    from fla.ops.abc import chunk_abc

    B, T, H, K, V, M = 2, 64, 2, 64, 64, 32
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    s = torch.randn(B, T, H, M, device=device, dtype=dtype, requires_grad=True)
    o, final_state = chunk_abc(q, k, v, s, initial_state=None, output_final_state=True)
    hkt, hvt = final_state
    return [o, hkt, hvt]


def _attn_parallel(torch, device, dtype):
    from fla.ops.attn import parallel_attn

    B, T, H, HQ, D = 2, 64, 2, 8, 64
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    o = parallel_attn(q=q, k=k, v=v, scale=D**-0.5)
    return [o]


def _attn_parallel_varlen(torch, device, dtype):
    from fla.ops.attn import parallel_attn

    T, H, HQ, D = 64, 2, 8, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = torch.randn(1, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    o = parallel_attn(q=q, k=k, v=v, scale=D**-0.5, cu_seqlens=cu_seqlens)
    return [o]


def _based_fused_chunk(torch, device, dtype):
    from fla.ops.based import fused_chunk_based

    B, T, H, V = 2, 64, 2, 64
    K = 16
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    o = fused_chunk_based(q, k, v, use_norm=True)
    return [o]


def _based_parallel(torch, device, dtype):
    from fla.ops.based import parallel_based

    B, T, H, V = 2, 64, 2, 64
    K = 16
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    o = parallel_based(q, k, v, use_norm=True)
    return [o]


def _comba_chunk(torch, device, dtype):
    from fla.ops.comba import chunk_comba
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    p = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    g = F.logsigmoid(
        torch.rand(B, T, H, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    o, ht = chunk_comba(
        q=q, k=k, v=v, p=p, g=g, beta=beta, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _comba_chunk_varlen(torch, device, dtype):
    from fla.ops.comba import chunk_comba
    import torch.nn.functional as F

    B, T, H, K, V = 1, 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    p = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    g = F.logsigmoid(
        torch.rand(B, T, H, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    o, ht = chunk_comba(
        q=q,
        k=k,
        v=v,
        p=p,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, ht]


def _comba_fused_recurrent(torch, device, dtype):
    from fla.ops.comba import fused_recurrent_comba
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    k = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    p = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, device=device, dtype=torch.float32))
    o, ht = fused_recurrent_comba(
        q=q, k=k, v=v, p=p, g=g, beta=beta, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _delta_rule_chunk(torch, device, dtype):
    from fla.ops.delta_rule import chunk_delta_rule
    import torch.nn.functional as F

    # chunk_delta_rule asserts against float32 inputs; it requires bf16/fp16.
    dtype = torch.bfloat16
    B, T, H, K, V = 2, 64, 2, 64, 64
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    o, ht = chunk_delta_rule(q, k, v, beta, initial_state=None, output_final_state=True)
    return [o, ht]


def _delta_rule_chunk_varlen(torch, device, dtype):
    from fla.ops.delta_rule import chunk_delta_rule
    import torch.nn.functional as F

    # chunk_delta_rule asserts against float32 inputs; it requires bf16/fp16.
    dtype = torch.bfloat16
    B, T, H, K, V = 1, 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    o, ht = chunk_delta_rule(
        q,
        k,
        v,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, ht]


def _delta_rule_fused_recurrent(torch, device, dtype):
    from fla.ops.delta_rule import fused_recurrent_delta_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.randn(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    o, ht = fused_recurrent_delta_rule(
        q, k, v, beta, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _gated_delta_rule_chunk(torch, device, dtype):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    g = F.logsigmoid(
        torch.rand(B, T, H, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    o, ht = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _gated_delta_rule_chunk_varlen(torch, device, dtype):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import torch.nn.functional as F

    B, T, H, K, V = 1, 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    g = F.logsigmoid(
        torch.rand(B, T, H, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    o, ht = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, ht]


def _gated_delta_rule_fused_recurrent(torch, device, dtype):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    k = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, device=device, dtype=torch.float32))
    o, ht = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _gated_oja_rule_chunk(torch, device, dtype):
    from fla.ops.gated_oja_rule import chunk_gated_oja_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = (
        F.normalize(
            torch.randn(B, T, H, V, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .detach()
        .requires_grad_(True)
    )
    beta = (
        torch.rand(B, T, H, device=device, dtype=torch.float32)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    gv = (
        F.logsigmoid(torch.rand(B, T, H, V, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = chunk_gated_oja_rule(
        q=q, k=k, v=v, gv=gv, beta=beta, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _gated_oja_rule_chunk_varlen(torch, device, dtype):
    from fla.ops.gated_oja_rule import chunk_gated_oja_rule
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = (
        F.normalize(
            torch.randn(1, T, H, V, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .detach()
        .requires_grad_(True)
    )
    beta = (
        torch.rand(1, T, H, device=device, dtype=torch.float32)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    gv = (
        F.logsigmoid(torch.rand(1, T, H, V, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.zeros(N, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = chunk_gated_oja_rule(
        q=q,
        k=k,
        v=v,
        gv=gv,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, ht]


def _gated_oja_rule_fused_recurrent(torch, device, dtype):
    from fla.ops.gated_oja_rule import fused_recurrent_gated_oja_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = F.normalize(
        torch.randn(B, T, H, V, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    gv = F.logsigmoid(torch.rand(B, T, H, V, device=device, dtype=torch.float32))
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = fused_recurrent_gated_oja_rule(
        q=q, k=k, v=v, gv=gv, beta=beta, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _gdn2_chunk(torch, device, dtype):
    from fla.ops.gdn2 import chunk_gdn2
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = (torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    g = (
        torch.empty(B, T, H, K, device=device, dtype=torch.float32)
        .uniform_(-5.0, -0.1)
        .requires_grad_(True)
    )
    b = torch.rand(B, T, H, K, device=device, dtype=dtype).requires_grad_(True)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype).requires_grad_(True)
    o, ht = chunk_gdn2(q, k, v, g, b, w, initial_state=None, output_final_state=True)
    return [o, ht]


def _gdn2_chunk_varlen(torch, device, dtype):
    from fla.ops.gdn2 import chunk_gdn2
    import torch.nn.functional as F

    B, T, H, K, V = 1, 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        F.normalize(
            torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = (torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    g = (
        torch.empty(B, T, H, K, device=device, dtype=torch.float32)
        .uniform_(-5.0, -0.1)
        .requires_grad_(True)
    )
    b = torch.rand(B, T, H, K, device=device, dtype=dtype).requires_grad_(True)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype).requires_grad_(True)
    o, ht = chunk_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, ht]


def _gdn2_fused_recurrent(torch, device, dtype):
    from fla.ops.gdn2 import fused_recurrent_gdn2
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    k = F.normalize(
        torch.randn(B, T, H, K, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.5
    g = torch.empty(B, T, H, K, device=device, dtype=torch.float32).uniform_(-5.0, -0.1)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)
    o, ht = fused_recurrent_gdn2(
        q, k, v, g, b, w, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _generalized_delta_rule_dplr_fused_recurrent(torch, device, dtype):
    from fla.ops.generalized_delta_rule import fused_recurrent_dplr_delta_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    a = F.normalize(torch.rand(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32)) / 4
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = fused_recurrent_dplr_delta_rule(
        q, k, v, a, b, gk, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _generalized_delta_rule_iplr_fused_recurrent(torch, device, dtype):
    from fla.ops.generalized_delta_rule import fused_recurrent_iplr_delta_rule
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    a = (
        F.normalize(torch.rand(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1)
        .detach()
        .requires_grad_(True)
    )
    b = (-a).detach().requires_grad_(True)
    h0 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = fused_recurrent_iplr_delta_rule(
        q, k, v, a, b, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _generalized_delta_rule_iplr_fused_recurrent_varlen(torch, device, dtype):
    from fla.ops.generalized_delta_rule import fused_recurrent_iplr_delta_rule
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    a = (
        F.normalize(torch.rand(1, T, H, K, device=device, dtype=dtype), p=2, dim=-1)
        .detach()
        .requires_grad_(True)
    )
    b = (-a).detach().requires_grad_(True)
    h0 = torch.zeros(N, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = fused_recurrent_iplr_delta_rule(
        q, k, v, a, b, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens
    )
    return [o, ht]


def _gla_chunk(torch, device, dtype):
    from fla.ops.gla import chunk_gla
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, ht = chunk_gla(q, k, v, g, initial_state=None, output_final_state=True)
    return [o, ht]


def _gla_chunk_varlen(torch, device, dtype):
    from fla.ops.gla import chunk_gla
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(1, T, H, K, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, ht = chunk_gla(
        q, k, v, g, initial_state=None, output_final_state=True, cu_seqlens=cu_seqlens
    )
    return [o, ht]


def _gla_fused_recurrent(torch, device, dtype):
    from fla.ops.gla import fused_recurrent_gla
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    gk = (
        F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, ht = fused_recurrent_gla(
        q, k, v, gk=gk, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _gsa_chunk(torch, device, dtype):
    from fla.ops.gsa import chunk_gsa
    import torch.nn.functional as F

    B, T, H, K, V, M = 2, 64, 2, 64, 64, 32
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    s = torch.randn(B, T, H, M, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, M, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    hk0 = torch.randn(
        B, H, K, M, device=device, dtype=torch.float32, requires_grad=True
    )
    hv0 = torch.randn(
        B, H, M, V, device=device, dtype=torch.float32, requires_grad=True
    )
    o, (hkt, hvt) = chunk_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=K**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
    )
    return [o, hkt, hvt]


def _gsa_chunk_varlen(torch, device, dtype):
    from fla.ops.gsa import chunk_gsa
    import torch.nn.functional as F

    T, H, K, V, M = 64, 2, 64, 64, 32
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    s = torch.randn(1, T, H, M, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(1, T, H, M, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    hk0 = torch.randn(
        N, H, K, M, device=device, dtype=torch.float32, requires_grad=True
    )
    hv0 = torch.randn(
        N, H, M, V, device=device, dtype=torch.float32, requires_grad=True
    )
    o, (hkt, hvt) = chunk_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=K**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, hkt, hvt]


def _gsa_fused_recurrent(torch, device, dtype):
    from fla.ops.gsa import fused_recurrent_gsa
    import torch.nn.functional as F

    B, T, H, K, V, M = 2, 64, 2, 64, 64, 32
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    s = torch.randn(B, T, H, M, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, M, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    hk0 = torch.randn(
        B, H, K, M, device=device, dtype=torch.float32, requires_grad=True
    )
    hv0 = torch.randn(
        B, H, M, V, device=device, dtype=torch.float32, requires_grad=True
    )
    o, (hkt, hvt) = fused_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=K**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
    )
    return [o, hkt, hvt]


def _hgrn_chunk(torch, device, dtype):
    from fla.ops.hgrn import chunk_hgrn
    import torch.nn.functional as F

    B, T, D = 2, 64, 64
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    g = torch.randn(B, T, D, device=device, dtype=dtype)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x = x.detach().requires_grad_(True)
    g = g.detach().requires_grad_(True)
    h0 = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    o, ht = chunk_hgrn(x, g, initial_state=h0, output_final_state=True)
    return [o, ht]


def _hgrn_fused_recurrent(torch, device, dtype):
    from fla.ops.hgrn import fused_recurrent_hgrn
    import torch.nn.functional as F

    B, T, D = 2, 64, 64
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    g = torch.randn(B, T, D, device=device, dtype=dtype)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x = x.detach().requires_grad_(True)
    g = g.detach().requires_grad_(True)
    h0 = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    o, ht = fused_recurrent_hgrn(x, g, initial_state=h0, output_final_state=True)
    return [o, ht]


def _hgrn_fused_recurrent_varlen(torch, device, dtype):
    from fla.ops.hgrn import fused_recurrent_hgrn
    import torch.nn.functional as F

    T, D = 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    x = torch.randn(1, T, D, device=device, dtype=dtype)
    g = torch.randn(1, T, D, device=device, dtype=dtype)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x = x.detach().requires_grad_(True)
    g = g.detach().requires_grad_(True)
    h0 = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
    o, ht = fused_recurrent_hgrn(
        x, g, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens
    )
    return [o, ht]


def _kda_chunk(torch, device, dtype):
    from fla.ops.kda import chunk_kda
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.rand(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.rand(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.rand(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    beta = (
        torch.randn(B, T, H, device=device, dtype=dtype)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = chunk_kda(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
    )
    return [o, ht]


def _kda_chunk_varlen(torch, device, dtype):
    from fla.ops.kda import chunk_kda
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int64, device=device)
    N = len(cu_seqlens) - 1
    q = torch.rand(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.rand(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.rand(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(1, T, H, K, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    beta = (
        torch.randn(1, T, H, device=device, dtype=dtype)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.randn(N, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = chunk_kda(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.cpu(),
    )
    return [o, ht]


def _kda_fused_recurrent(torch, device, dtype):
    from fla.ops.kda import fused_recurrent_kda
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.rand(B, T, H, K, device=device, dtype=dtype)
    k = torch.rand(B, T, H, K, device=device, dtype=dtype)
    v = torch.rand(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32))
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = fused_recurrent_kda(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
    )
    return [o, ht]


def _log_linear_attn_chunk(torch, device, dtype):
    from fla.ops.log_linear_attn import chunk_log_linear_attn
    import torch.nn.functional as F

    # K fixed at 64 (must be divisible by BLOCK_K=64); V=32 (power of two)
    # keeps the fused fp32 backward kernel within the RTX 4090's 101KB smem.
    B, T, H, K, V = 2, 64, 2, 64, 32
    L = 7  # int(log2(T) + 1) for T=64
    q = torch.randn(B, T, 1, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, 1, K, device=device, dtype=dtype, requires_grad=True)
    dt = F.softplus(torch.randn(B, T, H, device=device, dtype=torch.float32) - 4)
    a = -torch.exp(torch.rand(H, device=device, dtype=torch.float32))
    x = torch.randn(B, T, H, V, device=device, dtype=dtype)
    v = (x * dt.unsqueeze(-1)).to(dtype).detach().requires_grad_(True)
    g = (a * dt).detach().requires_grad_(True)
    level_scales = torch.randn(
        B, T, H, L, device=device, dtype=dtype, requires_grad=True
    )
    o, _ = chunk_log_linear_attn(q, k, v, g, level_scales)
    return [o]


def _log_linear_attn_chunk_varlen(torch, device, dtype):
    from fla.ops.log_linear_attn import chunk_log_linear_attn
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 32
    L = 7  # int(ceil(log2(total_T)) + 1) for total_T=64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int64, device=device)
    q = torch.randn(1, T, 1, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, 1, K, device=device, dtype=dtype, requires_grad=True)
    dt = F.softplus(torch.randn(1, T, H, device=device, dtype=torch.float32) - 4)
    a = -torch.exp(torch.rand(H, device=device, dtype=torch.float32))
    x = torch.randn(1, T, H, V, device=device, dtype=dtype)
    v = (x * dt.unsqueeze(-1)).to(dtype).detach().requires_grad_(True)
    g = (a * dt).detach().requires_grad_(True)
    level_scales = torch.randn(
        1, T, H, L, device=device, dtype=dtype, requires_grad=True
    )
    o, _ = chunk_log_linear_attn(q, k, v, g, level_scales, cu_seqlens=cu_seqlens)
    return [o]


def _mesa_net_chunk(torch, device, dtype):
    from fla.ops.mesa_net import chunk_mesa_net
    import torch.nn.functional as F

    B, T, H, D = 2, 64, 2, 64
    q = (torch.rand(B, T, H, D, device=device, dtype=dtype) / 10).requires_grad_(True)
    k = (
        F.normalize(
            torch.rand(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = (torch.rand(B, T, H, D, device=device, dtype=dtype) / 10).requires_grad_(True)
    beta = (
        torch.rand(B, T, H, device=device, dtype=dtype)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    g = (
        torch.empty(B, T, H, device=device, dtype=torch.float32)
        .uniform_(0.8, 0.99)
        .log()
        .detach()
        .requires_grad_(True)
    )
    lamb = (
        (torch.rand(H, D, device=device, dtype=dtype).sigmoid() * 0.75 + 0.25)
        .detach()
        .requires_grad_(True)
    )
    k_init = F.normalize(torch.rand(B, H, D, device=device, dtype=dtype), p=2, dim=-1)
    h_kk_init = (
        (k_init.unsqueeze(-1) * k_init.unsqueeze(-2))
        .detach()
        .float()
        .requires_grad_(True)
    )
    h_kv_init = torch.rand(
        B, H, D, D, device=device, dtype=torch.float32
    ).requires_grad_(True)
    o, h_kk, h_kv = chunk_mesa_net(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        lamb=lamb,
        h_kk_init=h_kk_init,
        h_kv_init=h_kv_init,
        max_CG_iteration=D,
        output_final_state=True,
    )
    return [o, h_kk, h_kv]


def _mesa_net_chunk_varlen(torch, device, dtype):
    from fla.ops.mesa_net import chunk_mesa_net
    import torch.nn.functional as F

    T, H, D = 64, 2, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.long, device=device)
    N = len(cu_seqlens) - 1
    q = (torch.randn(1, T, H, D, device=device, dtype=dtype) / 10).requires_grad_(True)
    k = (
        F.normalize(
            torch.randn(1, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = (torch.randn(1, T, H, D, device=device, dtype=dtype) / 10).requires_grad_(True)
    beta = (
        torch.rand(1, T, H, device=device, dtype=dtype)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    g = (
        torch.empty(1, T, H, device=device, dtype=torch.float32)
        .uniform_(0.8, 0.99)
        .log()
        .detach()
        .requires_grad_(True)
    )
    lamb = (
        (torch.rand(H, D, device=device, dtype=dtype).sigmoid() * 0.75 + 0.25)
        .detach()
        .requires_grad_(True)
    )
    k_init = F.normalize(torch.rand(N, H, D, device=device, dtype=dtype), p=2, dim=-1)
    h_kk_init = (
        (k_init.unsqueeze(-1) * k_init.unsqueeze(-2))
        .detach()
        .float()
        .requires_grad_(True)
    )
    h_kv_init = torch.rand(
        N, H, D, D, device=device, dtype=torch.float32
    ).requires_grad_(True)
    o, h_kk, h_kv = chunk_mesa_net(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        lamb=lamb,
        h_kk_init=h_kk_init,
        h_kv_init=h_kv_init,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, h_kk, h_kv]


def _mesa_net_decoding_one_step(torch, device, dtype):
    from fla.ops.mesa_net import mesa_net_decoding_one_step
    import torch.nn.functional as F

    B, H, D = 2, 2, 64
    q = torch.rand(B, H, D, device=device, dtype=dtype)
    k = F.normalize(
        torch.randn(B, H, D, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    v = torch.rand(B, H, D, device=device, dtype=dtype)
    g = (
        torch.empty(B, H, device=device, dtype=torch.float32)
        .uniform_(0.95, 0.99)
        .log()
        .to(dtype)
    )
    beta = torch.rand(B, H, device=device, dtype=dtype).sigmoid()
    lamb = torch.rand(H, D, device=device, dtype=dtype).sigmoid() * 0.75 + 0.25
    k_init = F.normalize(torch.rand(B, H, D, device=device, dtype=dtype), p=2, dim=-1)
    prev_h_kk = (k_init.unsqueeze(-1) * k_init.unsqueeze(-2)).float()
    prev_h_kv = torch.rand(B, H, D, D, device=device, dtype=torch.float32)
    o, h_kk, h_kv = mesa_net_decoding_one_step(
        q=q,
        k=k,
        v=v,
        g=g,
        lamb=lamb,
        beta=beta,
        prev_h_kk=prev_h_kk,
        prev_h_kv=prev_h_kv,
        max_CG_iteration=30,
    )
    return [o, h_kk, h_kv]


def _nsa_parallel(torch, device, dtype):
    from fla.ops.nsa import parallel_nsa

    # NSA enforces GQA group size HQ/H to be a power of 2 and >= 16.
    B, T, H, HQ, D, S, block_size = 2, 64, 1, 16, 64, 16, 32
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    block_indices = torch.full((B, T, H, S), -1, dtype=torch.long, device=device)
    for b in range(B):
        for i in range(T):
            for h in range(H):
                i_i = torch.randperm((i + block_size) // block_size)[:S]
                block_indices[b, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    o = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        scale=D**-0.5,
    )
    return [o]


def _nsa_parallel_varlen(torch, device, dtype):
    from fla.ops.nsa import parallel_nsa
    from fla.ops.utils import prepare_token_indices

    T, H, HQ, D, S, block_size = 64, 1, 16, 64, 16, 32
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = torch.randn(1, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    seq_indices = prepare_token_indices(cu_seqlens).tolist()
    block_indices = torch.full((1, T, H, S), -1, dtype=torch.long, device=device)
    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm((t + block_size) // block_size)[:S]
            block_indices[0, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    o = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        cu_seqlens=cu_seqlens,
    )
    return [o]


def _path_attn_parallel(torch, device, dtype):
    import importlib
    import pkgutil
    import torch.nn.functional as F
    from triton.runtime.jit import JITFunction
    import fla.ops.path_attn as pkg
    from fla.ops.path_attn import parallel_path_attn

    # fla-core 0.5.1 marks T as tl.constexpr while also listing it in
    # do_not_specialize; Triton >= 3.3 rejects that combination at compile time
    # (fixed in fla HEAD by dropping the constexpr annotation). Clearing the
    # do_not_specialize flag on constexpr params is semantically a no-op
    # (constexpr args are always specialized) and lets the kernels compile.
    for minfo in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"fla.ops.path_attn.{minfo.name}")
        for obj in vars(mod).values():
            fn = obj
            while not isinstance(fn, JITFunction) and hasattr(fn, "fn"):
                fn = fn.fn
            if isinstance(fn, JITFunction):
                for p in fn.params:
                    if p.is_constexpr and (
                        p.do_not_specialize or p.do_not_specialize_on_alignment
                    ):
                        p.do_not_specialize = False
                        p.do_not_specialize_on_alignment = False
    B, T, H, HQ, D = 2, 64, 2, 8, 64
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    w = (
        F.normalize(
            torch.randn(B, T, H, D, device=device, dtype=torch.float32), dim=-1, p=2
        )
        .detach()
        .requires_grad_(True)
    )
    beta = (
        torch.empty(B, T, H, device=device, dtype=torch.float32)
        .uniform_(1.5, 2.0)
        .requires_grad_(True)
    )
    g = (
        torch.empty(B, T, HQ, device=device, dtype=torch.float32)
        .uniform_(0.95, 1)
        .log()
        .detach()
        .requires_grad_(True)
    )
    o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, scale=D**-0.5)
    return [o]


def _path_attn_parallel_varlen(torch, device, dtype):
    import importlib
    import pkgutil
    import torch.nn.functional as F
    from triton.runtime.jit import JITFunction
    import fla.ops.path_attn as pkg
    from fla.ops.path_attn import parallel_path_attn

    # See path_attn_parallel: work around the constexpr/do_not_specialize
    # conflict in fla-core 0.5.1 on Triton >= 3.3.
    for minfo in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"fla.ops.path_attn.{minfo.name}")
        for obj in vars(mod).values():
            fn = obj
            while not isinstance(fn, JITFunction) and hasattr(fn, "fn"):
                fn = fn.fn
            if isinstance(fn, JITFunction):
                for p in fn.params:
                    if p.is_constexpr and (
                        p.do_not_specialize or p.do_not_specialize_on_alignment
                    ):
                        p.do_not_specialize = False
                        p.do_not_specialize_on_alignment = False
    T, H, HQ, D = 64, 2, 8, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = torch.randn(1, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    w = (
        F.normalize(
            torch.randn(1, T, H, D, device=device, dtype=torch.float32), dim=-1, p=2
        )
        .detach()
        .requires_grad_(True)
    )
    beta = (
        torch.rand(1, T, H, device=device, dtype=torch.float32)
        .sigmoid()
        .detach()
        .requires_grad_(True)
    )
    g = (
        torch.empty(1, T, HQ, device=device, dtype=torch.float32)
        .uniform_(0.95, 1)
        .log()
        .detach()
        .requires_grad_(True)
    )
    o, _ = parallel_path_attn(
        q=q, k=k, v=v, w=w, beta=beta, g=g, scale=D**-0.5, cu_seqlens=cu_seqlens
    )
    return [o]


def _retention_chunk(torch, device, dtype):
    from fla.ops.retention import chunk_retention

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    o, ht = chunk_retention(q, k, v, initial_state=None, output_final_state=True)
    return [o, ht]


def _retention_chunk_varlen(torch, device, dtype):
    from fla.ops.retention import chunk_retention

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    o, ht = chunk_retention(
        q, k, v, initial_state=None, output_final_state=True, cu_seqlens=cu_seqlens
    )
    return [o, ht]


def _rwkv6_chunk(torch, device, dtype):
    from fla.ops.rwkv6 import chunk_rwkv6
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    w = (
        F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    u = torch.randn(H, K, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.randn(B, H, K, V, device=device, dtype=dtype, requires_grad=True)
    o, ht = chunk_rwkv6(q, k, v, w, u, initial_state=h0, output_final_state=True)
    return [o, ht]


def _rwkv6_chunk_varlen(torch, device, dtype):
    from fla.ops.rwkv6 import chunk_rwkv6
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    w = (
        F.logsigmoid(torch.randn(1, T, H, K, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    u = torch.randn(H, K, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.randn(N, H, K, V, device=device, dtype=dtype, requires_grad=True)
    o, ht = chunk_rwkv6(
        q, k, v, w, u, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens
    )
    return [o, ht]


def _rwkv6_fused_recurrent(torch, device, dtype):
    from fla.ops.rwkv6 import fused_recurrent_rwkv6
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    w = (
        F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    u = torch.randn(H, K, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.randn(B, H, K, V, device=device, dtype=dtype, requires_grad=True)
    o, ht = fused_recurrent_rwkv6(
        q, k, v, w, u, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _rwkv7_chunk(torch, device, dtype):
    from fla.ops.rwkv7 import chunk_rwkv7
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    r = (
        torch.empty(B, T, H, K, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        torch.empty(B, T, H, K, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    v = (
        torch.empty(B, T, H, V, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    w = (
        torch.empty(B, T, H, K, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    kk = F.normalize(torch.empty(B, T, H, K, device=device).uniform_(-1, 1), dim=-1).to(
        dtype
    )
    a = (-kk).detach().requires_grad_(True)
    b = (
        (kk * torch.empty(B, T, H, K, device=device).uniform_(0, 0.1))
        .to(dtype)
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = chunk_rwkv7(
        r=r, w=w, k=k, v=v, a=a, b=b, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _rwkv7_chunk_varlen(torch, device, dtype):
    from fla.ops.rwkv7 import chunk_rwkv7
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    r = (
        torch.empty(1, T, H, K, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    k = (
        torch.empty(1, T, H, K, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    v = (
        torch.empty(1, T, H, V, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    w = (
        torch.empty(1, T, H, K, device=device)
        .uniform_(-8, -6)
        .to(dtype)
        .requires_grad_(True)
    )
    kk = F.normalize(torch.empty(1, T, H, K, device=device).uniform_(-1, 1), dim=-1).to(
        dtype
    )
    a = (-kk).detach().requires_grad_(True)
    b = (
        (kk * torch.empty(1, T, H, K, device=device).uniform_(0, 0.1))
        .to(dtype)
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.randn(N, H, K, V, device=device, dtype=torch.float32, requires_grad=True)
    o, ht = chunk_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.cpu(),
    )
    return [o, ht]


def _rwkv7_fused_recurrent(torch, device, dtype):
    from fla.ops.rwkv7 import fused_recurrent_rwkv7
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    r = torch.empty(B, T, H, K, device=device).uniform_(-8, -6).to(dtype)
    k = torch.empty(B, T, H, K, device=device).uniform_(-8, -6).to(dtype)
    v = torch.empty(B, T, H, V, device=device).uniform_(-8, -6).to(dtype)
    w = torch.empty(B, T, H, K, device=device).uniform_(-8, -6).to(dtype)
    kk = F.normalize(torch.empty(B, T, H, K, device=device).uniform_(-1, 1), dim=-1).to(
        dtype
    )
    a = -kk
    b = kk * torch.empty(B, T, H, K, device=device).uniform_(0, 0.1).to(dtype)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = fused_recurrent_rwkv7(
        r=r, w=w, k=k, v=v, a=a, b=b, initial_state=h0, output_final_state=True
    )
    return [o, ht]


def _simple_gla_chunk(torch, device, dtype):
    from fla.ops.simple_gla import chunk_simple_gla
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, ht = chunk_simple_gla(q, k, v, g, initial_state=None, output_final_state=True)
    return [o, ht]


def _simple_gla_chunk_varlen(torch, device, dtype):
    from fla.ops.simple_gla import chunk_simple_gla
    import torch.nn.functional as F

    T, H, K, V = 64, 2, 64, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    q = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(1, T, H, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, ht = chunk_simple_gla(
        q, k, v, g, initial_state=None, output_final_state=True, cu_seqlens=cu_seqlens
    )
    return [o, ht]


def _simple_gla_fused_recurrent(torch, device, dtype):
    from fla.ops.simple_gla import fused_recurrent_simple_gla
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, ht = fused_recurrent_simple_gla(
        q, k, v, g, initial_state=None, output_final_state=True
    )
    return [o, ht]


def _simple_gla_parallel(torch, device, dtype):
    from fla.ops.simple_gla import parallel_simple_gla
    import torch.nn.functional as F

    B, T, H, K, V = 2, 64, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
        .detach()
        .requires_grad_(True)
    )
    o, attn = parallel_simple_gla(q, k, v, g, output_attentions=False)
    return [o]


def _ttt_chunk(torch, device, dtype):
    from fla.ops.ttt import chunk_ttt_linear
    import torch.nn.functional as F

    B, T, H, D = 2, 64, 2, 64
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = (
        F.normalize(
            torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(H, D, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(H, D, device=device, dtype=dtype, requires_grad=True)
    eta = (
        (torch.randn(B, T, H, 1, device=device, dtype=dtype) * 5e-3)
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.randn(B, H, D, D, device=device, dtype=torch.float32, requires_grad=True)
    hb0 = torch.randn(
        B, H, 1, D, device=device, dtype=torch.float32, requires_grad=True
    )
    o, ht, hbt = chunk_ttt_linear(
        q,
        k,
        v,
        w,
        b,
        eta,
        scale=1.0,
        chunk_size=16,
        initial_state=h0,
        initial_state_bias=hb0,
        output_final_state=True,
    )
    return [o, ht, hbt]


def _ttt_chunk_varlen(torch, device, dtype):
    from fla.ops.ttt import chunk_ttt_linear
    import torch.nn.functional as F

    T, H, D = 64, 2, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    q = torch.randn(1, T, H, D, device=device, dtype=dtype)
    k = F.normalize(
        torch.randn(1, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(1, T, H, D, device=device, dtype=dtype)
    w = torch.randn(H, D, device=device, dtype=dtype)
    b = torch.randn(H, D, device=device, dtype=dtype)
    eta = torch.randn(1, T, H, 1, device=device, dtype=dtype) * 5e-3
    h0 = torch.randn(N, H, D, D, device=device, dtype=torch.float32)
    hb0 = torch.randn(N, H, 1, D, device=device, dtype=torch.float32)
    o, ht, hbt = chunk_ttt_linear(
        q,
        k,
        v,
        w,
        b,
        eta,
        scale=1.0,
        chunk_size=16,
        initial_state=h0,
        initial_state_bias=hb0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return [o, ht, hbt]


def _ttt_fused_chunk(torch, device, dtype):
    from fla.ops.ttt import fused_chunk_ttt_linear
    import torch.nn.functional as F

    B, T, H, D = 2, 64, 2, 64
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = (
        F.normalize(
            torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
        )
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(H, D, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(H, D, device=device, dtype=dtype, requires_grad=True)
    eta = (
        (torch.randn(B, T, H, 1, device=device, dtype=dtype) * 5e-3)
        .detach()
        .requires_grad_(True)
    )
    h0 = torch.randn(B, H, D, D, device=device, dtype=torch.float32, requires_grad=True)
    hb0 = torch.randn(
        B, H, 1, D, device=device, dtype=torch.float32, requires_grad=True
    )
    o, ht, hbt = fused_chunk_ttt_linear(
        q,
        k,
        v,
        w,
        b,
        eta,
        scale=1.0,
        chunk_size=16,
        initial_state=h0,
        initial_state_bias=hb0,
        output_final_state=True,
    )
    return [o, ht, hbt]


def _utils_chunk_local_cumsum(torch, device, dtype):
    from fla.ops.utils.cumsum import chunk_local_cumsum

    B, T, H, D = 2, 64, 2, 64
    s = torch.randn(B, T, H, device=device, dtype=dtype)
    x = torch.randn(B, T, H, D, device=device, dtype=dtype)
    o_scalar = chunk_local_cumsum(s, chunk_size=16)
    o_vector = chunk_local_cumsum(x, chunk_size=16)
    return [o_scalar, o_vector]


def _utils_chunk_local_cumsum_varlen(torch, device, dtype):
    from fla.ops.utils.cumsum import chunk_local_cumsum

    T, H, D = 64, 2, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    s = torch.randn(1, T, H, device=device, dtype=dtype)
    x = torch.randn(1, T, H, D, device=device, dtype=dtype)
    o_scalar = chunk_local_cumsum(s, chunk_size=16, cu_seqlens=cu_seqlens)
    o_vector = chunk_local_cumsum(x, chunk_size=16, cu_seqlens=cu_seqlens)
    return [o_scalar, o_vector]


def _utils_mean_pooling(torch, device, dtype):
    from fla.ops.utils.pooling import mean_pooling

    B, T, H, D = 2, 64, 2, 64
    x = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    o = mean_pooling(x, chunk_size=16)
    return [o]


def _utils_mean_pooling_varlen(torch, device, dtype):
    from fla.ops.utils.pooling import mean_pooling

    T, H, D = 64, 2, 64
    cu_seqlens = torch.tensor([0, 29, 64], dtype=torch.int32, device=device)
    x = torch.randn(1, T, H, D, device=device, dtype=dtype, requires_grad=True)
    o = mean_pooling(x, chunk_size=16, cu_seqlens=cu_seqlens)
    return [o]


def _utils_solve_tril(torch, device, dtype):
    from fla.ops.utils.solve_tril import solve_tril
    import torch.nn.functional as F

    B, T, H, C, D = 2, 64, 2, 16, 64
    k = F.normalize(torch.randn(B, H, T, D, device=device, dtype=torch.float32), dim=-1)
    k_blocks = k.reshape(B, H, T // C, C, D)
    A = (k_blocks @ k_blocks.transpose(-1, -2)).tril(-1)
    A = A.reshape(B, H, T, C).transpose(1, 2)
    Ai = solve_tril(A)
    return [Ai]


CASES: dict = {
    "abc_chunk": ("abc", True, _abc_chunk),
    "attn_parallel": ("attn", True, _attn_parallel),
    "attn_parallel_varlen": ("attn", True, _attn_parallel_varlen),
    "based_fused_chunk": ("based", True, _based_fused_chunk),
    "based_parallel": ("based", True, _based_parallel),
    "comba_chunk": ("comba", True, _comba_chunk),
    "comba_chunk_varlen": ("comba", True, _comba_chunk_varlen),
    "comba_fused_recurrent": ("comba", False, _comba_fused_recurrent),
    "delta_rule_chunk": ("delta_rule", True, _delta_rule_chunk),
    "delta_rule_chunk_varlen": ("delta_rule", True, _delta_rule_chunk_varlen),
    "delta_rule_fused_recurrent": ("delta_rule", True, _delta_rule_fused_recurrent),
    "gated_delta_rule_chunk": ("gated_delta_rule", True, _gated_delta_rule_chunk),
    "gated_delta_rule_chunk_varlen": (
        "gated_delta_rule",
        True,
        _gated_delta_rule_chunk_varlen,
    ),
    "gated_delta_rule_fused_recurrent": (
        "gated_delta_rule",
        False,
        _gated_delta_rule_fused_recurrent,
    ),
    "gated_oja_rule_chunk": ("gated_oja_rule", True, _gated_oja_rule_chunk),
    "gated_oja_rule_chunk_varlen": (
        "gated_oja_rule",
        True,
        _gated_oja_rule_chunk_varlen,
    ),
    "gated_oja_rule_fused_recurrent": (
        "gated_oja_rule",
        False,
        _gated_oja_rule_fused_recurrent,
    ),
    "gdn2_chunk": ("gdn2", True, _gdn2_chunk),
    "gdn2_chunk_varlen": ("gdn2", True, _gdn2_chunk_varlen),
    "gdn2_fused_recurrent": ("gdn2", False, _gdn2_fused_recurrent),
    "generalized_delta_rule_dplr_fused_recurrent": (
        "generalized_delta_rule",
        False,
        _generalized_delta_rule_dplr_fused_recurrent,
    ),
    "generalized_delta_rule_iplr_fused_recurrent": (
        "generalized_delta_rule",
        True,
        _generalized_delta_rule_iplr_fused_recurrent,
    ),
    "generalized_delta_rule_iplr_fused_recurrent_varlen": (
        "generalized_delta_rule",
        True,
        _generalized_delta_rule_iplr_fused_recurrent_varlen,
    ),
    "gla_chunk": ("gla", True, _gla_chunk),
    "gla_chunk_varlen": ("gla", True, _gla_chunk_varlen),
    "gla_fused_recurrent": ("gla", True, _gla_fused_recurrent),
    "gsa_chunk": ("gsa", True, _gsa_chunk),
    "gsa_chunk_varlen": ("gsa", True, _gsa_chunk_varlen),
    "gsa_fused_recurrent": ("gsa", True, _gsa_fused_recurrent),
    "hgrn_chunk": ("hgrn", True, _hgrn_chunk),
    "hgrn_fused_recurrent": ("hgrn", True, _hgrn_fused_recurrent),
    "hgrn_fused_recurrent_varlen": ("hgrn", True, _hgrn_fused_recurrent_varlen),
    "kda_chunk": ("kda", True, _kda_chunk),
    "kda_chunk_varlen": ("kda", True, _kda_chunk_varlen),
    "kda_fused_recurrent": ("kda", False, _kda_fused_recurrent),
    "log_linear_attn_chunk": ("log_linear_attn", True, _log_linear_attn_chunk),
    "log_linear_attn_chunk_varlen": (
        "log_linear_attn",
        True,
        _log_linear_attn_chunk_varlen,
    ),
    "mesa_net_chunk": ("mesa_net", True, _mesa_net_chunk),
    "mesa_net_chunk_varlen": ("mesa_net", True, _mesa_net_chunk_varlen),
    "mesa_net_decoding_one_step": ("mesa_net", False, _mesa_net_decoding_one_step),
    "nsa_parallel": ("nsa", True, _nsa_parallel),
    "nsa_parallel_varlen": ("nsa", True, _nsa_parallel_varlen),
    "path_attn_parallel": ("path_attn", True, _path_attn_parallel),
    "path_attn_parallel_varlen": ("path_attn", True, _path_attn_parallel_varlen),
    "retention_chunk": ("retention", True, _retention_chunk),
    "retention_chunk_varlen": ("retention", True, _retention_chunk_varlen),
    "rwkv6_chunk": ("rwkv6", True, _rwkv6_chunk),
    "rwkv6_chunk_varlen": ("rwkv6", True, _rwkv6_chunk_varlen),
    "rwkv6_fused_recurrent": ("rwkv6", True, _rwkv6_fused_recurrent),
    "rwkv7_chunk": ("rwkv7", True, _rwkv7_chunk),
    "rwkv7_chunk_varlen": ("rwkv7", True, _rwkv7_chunk_varlen),
    "rwkv7_fused_recurrent": ("rwkv7", False, _rwkv7_fused_recurrent),
    "simple_gla_chunk": ("simple_gla", True, _simple_gla_chunk),
    "simple_gla_chunk_varlen": ("simple_gla", True, _simple_gla_chunk_varlen),
    "simple_gla_fused_recurrent": ("simple_gla", True, _simple_gla_fused_recurrent),
    "simple_gla_parallel": ("simple_gla", True, _simple_gla_parallel),
    "ttt_chunk": ("ttt", True, _ttt_chunk),
    "ttt_chunk_varlen": ("ttt", False, _ttt_chunk_varlen),
    "ttt_fused_chunk": ("ttt", True, _ttt_fused_chunk),
    "utils_chunk_local_cumsum": ("utils", False, _utils_chunk_local_cumsum),
    "utils_chunk_local_cumsum_varlen": (
        "utils",
        False,
        _utils_chunk_local_cumsum_varlen,
    ),
    "utils_mean_pooling": ("utils", True, _utils_mean_pooling),
    "utils_mean_pooling_varlen": ("utils", True, _utils_mean_pooling_varlen),
    "utils_solve_tril": ("utils", False, _utils_solve_tril),
}


# ── capture driver ───────────────────────────────────────────────


def main() -> None:
    from evaluation.capture_common import capture_one_case, run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        os.environ.setdefault("FLA_USE_TMA", "0")  # sm90-only path, keep off
        result = capture_one_case(CASES, args.one, dtype_name="float32")
        args.out.write_text(json.dumps(result, indent=1))
        return

    from evaluation.runner import _fla_provenance

    prov = _fla_provenance()
    run_case_capture(
        "evaluation.fla_capture",
        CASES,
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "fla_core": prov.get("fla_core"),
            "upstream_commit": prov.get("fla_core_commit"),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
