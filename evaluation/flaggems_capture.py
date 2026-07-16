"""One-time GPU launch capture for the FlagGems corpus.

flag_gems is analyzed AS INSTALLED — git-pinned pip install (PyPI lags
upstream by months; ``runner._flaggems_provenance()`` reads the exact
commit from pip's direct_url.json). This module drives public
``flag_gems.ops.*`` entry points across the race-relevant op families
(atomic scatter/index, histogram, embedding backward, sort/topk/scan,
unique/nonzero cumsum-addressed stores, the stream-K GEMM spinlock,
reductions/norms) under the shared capture layer (capture_common.py).

Kernels are wrapped in @libentry() (+ optional @libtuner/@triton.
autotune/@triton.heuristics); all wrappers expose ``.fn`` chains, so
the shared type-descent unwrap reaches the JITFunction, and the
JITFunction.run hook fires on the first launch per specialization
(LibEntry serves later launches from its own cache — irrelevant to
first-launch capture). Runtime-CODEGEN kernels (pointwise_dynamic
writes modules under ~/.flaggems/code_cache with process-dependent
names) are filtered to ``skipped_kernels`` via module_prefix — they
cannot be re-imported at rebuild time.

Usage (GPU machine):
    uv run python -m evaluation.flaggems_capture               # all cases
    uv run python -m evaluation.flaggems_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECS_PATH = Path(__file__).parent / "kernels" / "flaggems_specs.json"
PER_CASE_TIMEOUT_S = 600
UPSTREAM = "https://github.com/flagos-ai/FlagGems"


# ── case table ───────────────────────────────────────────────────
# Each case: (family, bwd, run) — run(torch, device, dtype) calls one
# public flag_gems.ops entry point at small shapes (index tensors kept
# ≤8192 elements so value snapshots stay exact) and returns its output
# tensors. GPU-validated per case before landing here.


def _addmm(torch, device, dtype):
    import flag_gems

    bias = torch.randn(512, device=device, dtype=dtype)
    mat1 = torch.randn(256, 128, device=device, dtype=dtype)
    mat2 = torch.randn(128, 512, device=device, dtype=dtype)
    return [flag_gems.ops.addmm(bias, mat1, mat2, beta=0.5, alpha=2.0)]


def _amax(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.amax(x, dim=[1])]


def _any_dim(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype) > 2.5
    return [flag_gems.ops.any_dim(x, dim=1)]


def _any_full(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype) > 2.5
    return [flag_gems.ops.any(x)]


def _argmax_dim(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.argmax(x, dim=1)]


def _argmax_full(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(4096, device=device, dtype=dtype)
    return [flag_gems.ops.argmax(x)]


def _argsort(torch, device, dtype):
    import flag_gems

    inp = torch.randn(8192, device=device, dtype=dtype)
    return [flag_gems.ops.argsort(inp, dim=-1, descending=True)]


def _bincount(torch, device, dtype):
    import flag_gems

    inp = torch.randint(0, 512, (8192,), device=device)
    return [flag_gems.ops.bincount(inp, minlength=600)]


def _bincount_weighted(torch, device, dtype):
    import flag_gems

    inp = torch.randint(0, 256, (4096,), device=device)
    weights = torch.rand(4096, device=device, dtype=dtype)
    return [flag_gems.ops.bincount(inp, weights=weights)]


def _bmm(torch, device, dtype):
    import flag_gems

    A = torch.randn(4, 256, 128, device=device, dtype=dtype)
    B = torch.randn(4, 128, 256, device=device, dtype=dtype)
    return [flag_gems.ops.bmm(A, B)]


def _count_nonzero(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = (torch.randn(512, 512, device=device, dtype=dtype) > 0.5).to(dtype)
    return [flag_gems.ops.count_nonzero(x)]


def _count_nonzero_dim(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = (torch.randn(512, 512, device=device, dtype=dtype) > 0.5).to(dtype)
    return [flag_gems.ops.count_nonzero(x, dim=1)]


def _cross_entropy_loss(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 64, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, 64, (512,), device=device)
    return [flag_gems.cross_entropy_loss(inp, target, reduction="mean")]


def _cross_entropy_loss_smooth(torch, device, dtype):
    import flag_gems

    inp = torch.randn(256, 128, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, 128, (256,), device=device)
    weight = torch.rand(128, device=device, dtype=dtype) + 0.1
    return [
        flag_gems.cross_entropy_loss(
            inp,
            target,
            weight=weight,
            reduction="sum",
            ignore_index=7,
            label_smoothing=0.1,
        )
    ]


def _cummax(torch, device, dtype):
    import flag_gems

    inp = torch.randint(0, 64, (4096,), device=device).to(dtype)
    values, indices = flag_gems.ops.cummax(inp, dim=0)
    return [values, indices]


def _cumprod(torch, device, dtype):
    import flag_gems

    inp = torch.rand(4096, device=device, dtype=dtype) * 0.1 + 0.95
    return [flag_gems.ops.cumprod(inp, dim=0)]


def _cumsum(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.cumsum(inp, dim=1)]


def _dropout(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    out, mask = flag_gems.ops.dropout(x, 0.5, True)
    return [out, mask]


def _dropout_bwd(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    out, mask = flag_gems.ops.dropout(x, 0.5, True)
    dy = torch.randn(512, 512, device=device, dtype=dtype)
    dx = flag_gems.ops.dropout_backward(dy, mask, 1.0 / (1.0 - 0.5))
    return [dx]


def _embedding_bwd_dup(torch, device, dtype):
    import flag_gems

    num_weights = 1024
    indices = torch.randint(0, num_weights, (4, 64), device=device)
    indices[:, ::4] = 3  # duplicates -> atomic_add contention in grad_weight
    grad_out = torch.randn(4, 64, 128, device=device, dtype=dtype)
    return [
        flag_gems.ops.embedding_dense_backward(
            grad_out, indices, num_weights, -1, False
        )
    ]


def _embedding_bwd_freq(torch, device, dtype):
    import flag_gems

    num_weights = 512
    indices = torch.randint(0, num_weights, (2048,), device=device)
    indices[::3] = 11  # heavy duplicates for the frequency-count path
    grad_out = torch.randn(2048, 64, device=device, dtype=dtype)
    return [
        flag_gems.ops.embedding_dense_backward(grad_out, indices, num_weights, 2, True)
    ]


def _embedding_dup(torch, device, dtype):
    import flag_gems

    weight = torch.randn(1024, 128, device=device, dtype=dtype)
    indices = torch.randint(0, 1024, (4, 64), device=device)
    indices[:, ::2] = 7  # intentional duplicate rows
    return [flag_gems.ops.embedding(weight, indices)]


def _group_mm(torch, device, dtype):
    import flag_gems

    groups, N, K = 4, 64, 64
    M_list = [32, 48, 16, 64]
    dt = torch.bfloat16
    A = torch.randn(sum(M_list), K, device=device, dtype=dt)
    B = torch.randn(groups, K, N, device=device, dtype=dt)
    offs = torch.tensor(
        [sum(M_list[: i + 1]) for i in range(groups)],
        dtype=torch.int32,
        device=device,
    )
    return [flag_gems.ops.group_mm(A, B, offs)]


def _group_norm(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
    w = torch.randn(16, device=device, dtype=dtype)
    b = torch.randn(16, device=device, dtype=dtype)
    y, mean, rstd = flag_gems.ops.group_norm(x, w, b, 2, 16, 64, 4, 1e-5)
    return [y, mean, rstd]


def _group_norm_bwd(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
    w = torch.randn(16, device=device, dtype=dtype)
    b = torch.randn(16, device=device, dtype=dtype)
    y, mean, rstd = flag_gems.ops.group_norm(x, w, b, 2, 16, 64, 4, 1e-5)
    dy = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
    dx, dw, db = flag_gems.ops.group_norm_backward(
        dy, x, mean, rstd, w, 2, 16, 64, 4, [True, True, True]
    )
    return [dx, dw, db]


def _histc(torch, device, dtype):
    import flag_gems

    inp = torch.randn(32768, device=device, dtype=dtype)
    return [flag_gems.ops.histc(inp, bins=64, min=-3.0, max=3.0)]


def _index_add_dup(torch, device, dtype):
    import flag_gems

    inp = torch.randn(256, 32, device=device, dtype=dtype)
    src = torch.randn(64, 32, device=device, dtype=dtype)
    index = torch.randint(0, 16, (64,), device=device)  # duplicate rows -> atomic adds
    return [flag_gems.ops.index_add(inp, 0, index, src)]


def _index_put_acc_dup(torch, device, dtype):
    import flag_gems

    inp = torch.zeros(512, device=device, dtype=dtype)
    idx = torch.randint(
        0, 32, (2048,), device=device
    )  # heavy duplicates, accumulate=True is legal
    vals = torch.randn(2048, device=device, dtype=dtype)
    return [flag_gems.ops.index_put(inp, [idx], vals, accumulate=True)]


def _index_put_unique(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 32, device=device, dtype=dtype)
    idx = torch.randperm(512, device=device)[
        :128
    ]  # unique rows: duplicates illegal for accumulate=False
    vals = torch.randn(128, 32, device=device, dtype=dtype)
    return [flag_gems.ops.index_put(inp, [idx], vals, accumulate=False)]


def _index_reduce_amax_dup(torch, device, dtype):
    import flag_gems

    inp = torch.randn(256, 32, device=device, dtype=dtype)
    source = torch.randn(64, 32, device=device, dtype=dtype)
    index = torch.randint(
        0, 256, (64,), device=device
    )  # duplicates allowed for amax reduce
    return [flag_gems.ops.index_reduce_(inp, 0, index, source, "amax")]


def _kthvalue(torch, device, dtype):
    import flag_gems

    inp = torch.randn(4, 1024, device=device, dtype=dtype)
    values, indices = flag_gems.ops.kthvalue(inp, 7, dim=-1, keepdim=False)
    return [values, indices]


def _layer_norm(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    w = torch.randn(512, device=device, dtype=dtype)
    b = torch.randn(512, device=device, dtype=dtype)
    y, mean, rstd = flag_gems.ops.layer_norm(x, [512], w, b, 1e-5)
    return [y, mean, rstd]


def _layer_norm_bwd(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    w = torch.randn(512, device=device, dtype=dtype)
    b = torch.randn(512, device=device, dtype=dtype)
    y, mean, rstd = flag_gems.ops.layer_norm(x, [512], w, b, 1e-5)
    dy = torch.randn(512, 512, device=device, dtype=dtype)
    dx, dw, db = flag_gems.ops.layer_norm_backward(
        dy, x, [512], mean, rstd, w, b, [True, True, True]
    )
    return [dx, dw, db]


def _log_softmax(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.log_softmax(x, 1)]


def _logsumexp(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.logsumexp(x, dim=1)]


def _masked_fill(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 64, device=device, dtype=dtype)
    mask = inp < 0
    return [flag_gems.ops.masked_fill(inp, mask, -1.0)]


def _masked_scatter(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 64, device=device, dtype=dtype)
    mask = torch.rand(512, 64, device=device) > 0.5
    source = torch.randn(512 * 64, device=device, dtype=dtype)
    return [flag_gems.ops.masked_scatter(inp, mask, source)]


def _masked_select(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 64, device=device, dtype=dtype)
    mask = inp > 0
    return [flag_gems.ops.masked_select(inp, mask)]


def _mean_dim(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.mean_dim(x, dim=[1])]


def _mean_full(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.mean(x)]


def _mm(torch, device, dtype):
    import flag_gems

    a = torch.randn(512, 512, device=device, dtype=dtype)
    b = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.mm(a, b)]


def _mm_streamk(torch, device, dtype):
    # Stream-K mm with spinlock tile handoff (atomic_cas spin + relaxed atomic_add).
    # flag_gems.ops.mm's own streamk path hard-codes an A100 config
    # (BLOCK 128^3, num_stages=3 -> 128KB smem) that OOMs on sm_89 (99KB limit),
    # so we replicate streamk_mm's host-side launch with a 64^3/2-stage config
    # and launch the real first_wave + classic_mm kernels directly.
    import importlib
    import triton
    import flag_gems  # noqa: F401  (initializes runtime/backends)

    sk = importlib.import_module("flag_gems.ops.mm_streamk")

    M, N, K = 512, 512, 2048
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    BLOCK_M = BLOCK_N = BLOCK_K = 64
    GROUP_M, num_stages, num_warps = 8, 2, 4

    tiles_per_wave = torch.cuda.get_device_properties(device).multi_processor_count
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)  # 64
    iters_per_tile = triton.cdiv(K, BLOCK_K)  # 32
    number_cooperative_tiles = total_tiles // 2  # 32 tiles via spinlock wave
    total_iters_streamk = number_cooperative_tiles * iters_per_tile
    iters_per_pid = total_iters_streamk // tiles_per_wave
    iters_remaining = total_iters_streamk % tiles_per_wave
    even_k = K % BLOCK_K == 0

    locks = torch.zeros((number_cooperative_tiles,), device=device, dtype=torch.int32)
    sk.first_wave[(tiles_per_wave,)](
        a,
        b,
        c,
        M,
        N,
        K,
        locks,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        iters_per_pid=iters_per_pid,
        iters_remaining=iters_remaining,
        iters_per_tile=iters_per_tile,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        EVEN_K=even_k,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    classic_grid = total_tiles - number_cooperative_tiles
    if classic_grid > 0:
        sk.classic_mm[(classic_grid,)](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            total_tiles_streamk=number_cooperative_tiles,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return [c]


def _multinomial_replacement(torch, device, dtype):
    import flag_gems

    prob = torch.rand(512, device=device, dtype=dtype) + 0.01
    return [flag_gems.ops.multinomial(prob, 256, True)]


def _mv(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 512, device=device, dtype=dtype)
    vec = torch.randn(512, device=device, dtype=dtype)
    return [flag_gems.ops.mv(inp, vec)]


def _nll_loss_bwd(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 64, device=device, dtype=dtype).log_softmax(dim=1)
    target = torch.randint(0, 64, (512,), device=device)
    weight = torch.rand(64, device=device, dtype=dtype) + 0.1
    out, total_weight = flag_gems.ops.nll_loss_forward(
        inp, target, weight=weight, reduction=1, ignore_index=5
    )
    grad_output = torch.ones_like(out)
    grad_input = flag_gems.ops.nll_loss_backward(
        grad_output,
        inp,
        target,
        weight=weight,
        reduction=1,
        ignore_index=5,
        total_weight=total_weight,
    )
    return [grad_input]


def _nll_loss_fwd(torch, device, dtype):
    import flag_gems

    inp = torch.randn(512, 64, device=device, dtype=dtype).log_softmax(dim=1)
    target = torch.randint(0, 64, (512,), device=device)
    weight = torch.rand(64, device=device, dtype=dtype) + 0.1
    out, total_weight = flag_gems.ops.nll_loss_forward(
        inp, target, weight=weight, reduction=1, ignore_index=-100
    )
    return [out, total_weight]


def _nonzero(torch, device, dtype):
    import flag_gems

    inp = (torch.rand(512, 512, device=device, dtype=dtype) < 0.1).to(dtype)
    return [flag_gems.ops.nonzero(inp, as_tuple=False)]


def _outer(torch, device, dtype):
    import flag_gems

    x = torch.randn(512, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(256, device=device, dtype=dtype, requires_grad=True)
    return [flag_gems.outer(x, y)]


def _rand(torch, device, dtype):
    import flag_gems

    return [flag_gems.ops.rand((512, 512), dtype=dtype, device=device)]


def _rms_norm(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(512, device=device, dtype=dtype, requires_grad=True)
    return [flag_gems.ops.rms_norm(x, [512], w, 1e-5)]


def _rotary_embedding(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    q = torch.randn(1, 128, 8, 64, device=device, dtype=dtype)
    k = torch.randn(1, 128, 2, 64, device=device, dtype=dtype)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, device=device, dtype=dtype) / 32))
    t = torch.arange(128, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    q_emb, k_emb = flag_gems.fused.apply_rotary_pos_emb(q, k, cos, sin)
    return [q_emb, k_emb]


def _scatter_add_dup(torch, device, dtype):
    import flag_gems

    inp = torch.zeros(256, 32, device=device, dtype=dtype)
    src = torch.randn(64, 32, device=device, dtype=dtype)
    index = torch.randint(
        0, 16, (64, 32), device=device
    )  # duplicate destinations -> atomic adds
    return [flag_gems.ops.scatter_add_(inp, 0, index, src)]


def _scatter_dup_add(torch, device, dtype):
    import flag_gems

    inp = torch.zeros(256, 32, device=device, dtype=dtype)
    src = torch.randn(64, 32, device=device, dtype=dtype)
    # heavy duplicates: many source rows collide on the same destination rows
    index = torch.randint(0, 8, (64, 32), device=device)
    return [flag_gems.ops.scatter(inp, 0, index, src, reduce="add")]


def _scatter_reduce_amax_dup(torch, device, dtype):
    import flag_gems

    inp = torch.randn(256, 32, device=device, dtype=dtype)
    src = torch.randn(64, 32, device=device, dtype=dtype)
    index = torch.randint(
        0, 16, (64, 32), device=device
    )  # duplicate destinations -> atomic max
    return [flag_gems.ops.scatter_reduce(inp, 0, index, src, "amax", include_self=True)]


def _scatter_unique(torch, device, dtype):
    import flag_gems

    inp = torch.randn(256, 32, device=device, dtype=dtype)
    src = torch.randn(64, 32, device=device, dtype=dtype)
    # unique destination rows per column: distinct rows via randperm
    index = (
        torch.randperm(256, device=device)[:64].unsqueeze(1).expand(64, 32).contiguous()
    )
    return [flag_gems.ops.scatter(inp, 0, index, src)]


def _sdpa_causal_fp16(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    q = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    return [flag_gems.ops.scaled_dot_product_attention(q, k, v, is_causal=True)]


def _sdpa_noncausal_fp16(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    q = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    return [flag_gems.ops.scaled_dot_product_attention(q, k, v, is_causal=False)]


def _searchsorted(torch, device, dtype):
    import flag_gems

    sorted_seq = torch.sort(torch.randn(1024, device=device, dtype=dtype)).values
    values = torch.randn(512, device=device, dtype=dtype)
    return [flag_gems.ops.searchsorted(sorted_seq, values, right=False)]


def _softmax(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    return [flag_gems.ops.softmax(x, 1)]


def _sort(torch, device, dtype):
    import flag_gems

    inp = torch.randn(4, 4096, device=device, dtype=dtype)
    values, indices = flag_gems.ops.sort(inp, dim=-1, descending=False)
    return [values, indices]


def _topk(torch, device, dtype):
    import flag_gems

    inp = torch.randn(4, 8192, device=device, dtype=dtype)
    values, indices = flag_gems.ops.topk(inp, 32, dim=-1, largest=True, sorted=True)
    return [values, indices]


def _unique_dup(torch, device, dtype):
    import flag_gems

    inp = torch.randint(0, 128, (4096,), device=device)
    data_out, inverse, counts = flag_gems.ops._unique2(
        inp, sorted=True, return_inverse=True, return_counts=True
    )
    return [data_out, inverse, counts]


def _unique_large(torch, device, dtype):
    import flag_gems

    inp = torch.randint(0, 512, (32768,), device=device)
    data_out, inverse, counts = flag_gems.ops._unique2(
        inp, sorted=True, return_inverse=True, return_counts=True
    )
    return [data_out, inverse, counts]


def _var_mean(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    x = torch.randn(512, 512, device=device, dtype=dtype)
    var, mean = flag_gems.ops.var_mean(x, dim=[1], correction=1)
    return [var, mean]


def _vdot(torch, device, dtype):
    import flag_gems

    a = torch.randn(65536, device=device, dtype=torch.float32)
    b = torch.randn(65536, device=device, dtype=torch.float32)
    return [flag_gems.ops.vdot(a, b)]


def _weight_norm(torch, device, dtype):
    import flag_gems

    torch.manual_seed(0)
    v = torch.randn(64, 128, device=device, dtype=dtype)
    g = torch.randn(64, 1, device=device, dtype=dtype)
    w, norms = flag_gems.ops.weight_norm_interface(v, g, 0)
    return [w, norms]


CASES: dict = {
    "addmm": ("blas", False, _addmm),
    "amax": ("reduction", False, _amax),
    "any_dim": ("reduction", False, _any_dim),
    "any_full": ("reduction", False, _any_full),
    "argmax_dim": ("reduction", False, _argmax_dim),
    "argmax_full": ("reduction", False, _argmax_full),
    "argsort": ("sortscan", False, _argsort),
    "bincount": ("histogram", False, _bincount),
    "bincount_weighted": ("histogram", False, _bincount_weighted),
    "bmm": ("blas", False, _bmm),
    "count_nonzero": ("reduction", False, _count_nonzero),
    "count_nonzero_dim": ("reduction", False, _count_nonzero_dim),
    "cross_entropy_loss": ("loss", True, _cross_entropy_loss),
    "cross_entropy_loss_smooth": ("loss", True, _cross_entropy_loss_smooth),
    "cummax": ("sortscan", False, _cummax),
    "cumprod": ("sortscan", False, _cumprod),
    "cumsum": ("sortscan", False, _cumsum),
    "dropout": ("rand", False, _dropout),
    "dropout_bwd": ("rand", False, _dropout_bwd),
    "embedding_bwd_dup": ("embedding", False, _embedding_bwd_dup),
    "embedding_bwd_freq": ("embedding", False, _embedding_bwd_freq),
    "embedding_dup": ("embedding", False, _embedding_dup),
    "group_mm": ("blas", False, _group_mm),
    "group_norm": ("norm", False, _group_norm),
    "group_norm_bwd": ("norm", False, _group_norm_bwd),
    "histc": ("histogram", False, _histc),
    "index_add_dup": ("scatter", False, _index_add_dup),
    "index_put_acc_dup": ("scatter", False, _index_put_acc_dup),
    "index_put_unique": ("scatter", False, _index_put_unique),
    "index_reduce_amax_dup": ("scatter", False, _index_reduce_amax_dup),
    "kthvalue": ("sortscan", False, _kthvalue),
    "layer_norm": ("norm", False, _layer_norm),
    "layer_norm_bwd": ("norm", False, _layer_norm_bwd),
    "log_softmax": ("reduction", False, _log_softmax),
    "logsumexp": ("reduction", False, _logsumexp),
    "masked_fill": ("scatter", False, _masked_fill),
    "masked_scatter": ("scatter", False, _masked_scatter),
    "masked_select": ("scatter", False, _masked_select),
    "mean_dim": ("reduction", False, _mean_dim),
    "mean_full": ("reduction", False, _mean_full),
    "mm": ("blas", False, _mm),
    "mm_streamk": ("blas", False, _mm_streamk),
    "multinomial_replacement": ("sortscan", False, _multinomial_replacement),
    "mv": ("blas", False, _mv),
    "nll_loss_bwd": ("loss", False, _nll_loss_bwd),
    "nll_loss_fwd": ("loss", False, _nll_loss_fwd),
    "nonzero": ("sortscan", False, _nonzero),
    "outer": ("blas", True, _outer),
    "rand": ("rand", False, _rand),
    "rms_norm": ("norm", True, _rms_norm),
    "rotary_embedding": ("attn", False, _rotary_embedding),
    "scatter_add_dup": ("scatter", False, _scatter_add_dup),
    "scatter_dup_add": ("scatter", False, _scatter_dup_add),
    "scatter_reduce_amax_dup": ("scatter", False, _scatter_reduce_amax_dup),
    "scatter_unique": ("scatter", False, _scatter_unique),
    "sdpa_causal_fp16": ("attn", True, _sdpa_causal_fp16),
    "sdpa_noncausal_fp16": ("attn", True, _sdpa_noncausal_fp16),
    "searchsorted": ("sortscan", False, _searchsorted),
    "softmax": ("reduction", False, _softmax),
    "sort": ("sortscan", False, _sort),
    "topk": ("sortscan", False, _topk),
    "unique_dup": ("sortscan", False, _unique_dup),
    "unique_large": ("sortscan", False, _unique_large),
    "var_mean": ("reduction", False, _var_mean),
    "vdot": ("blas", False, _vdot),
    "weight_norm": ("norm", False, _weight_norm),
}


def main() -> None:
    from evaluation.capture_common import capture_one_case, run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        result = capture_one_case(
            CASES, args.one, dtype_name="float32", module_prefix="flag_gems."
        )
        args.out.write_text(json.dumps(result, indent=1))
        return

    from evaluation.runner import _flaggems_provenance

    prov = _flaggems_provenance()
    run_case_capture(
        "evaluation.flaggems_capture",
        CASES,
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "flag_gems": prov.get("flag_gems"),
            "upstream_commit": prov.get("flag_gems_commit"),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
