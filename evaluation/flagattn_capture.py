"""One-time GPU launch capture for the FlagAttention corpus.

flag_attn is analyzed AS INSTALLED — pinned by a git pip install
(``flag_attn @ git+https://github.com/FlagOpen/FlagAttention@<commit>``,
no PyPI release exists), so ``runner._flagattn_provenance()`` reads the
exact commit from pip's ``direct_url.json``. This module drives the four
public ops (flash / piecewise / split-kv / paged attention) at small
fp16 shapes under the shared capture layer (capture_common.py): flash
covers causal/non-causal, GQA, dropout (philox), non-divisible seqlen
(mask paths on) and the aux-output kernels; paged covers both the
single-split and the num_splits>1 partition+reduce pair.

The repo uses NO autotune — hand-written config tables fall back to
(BLOCK 32x32, 1 stage, 4 warps) on sm89 — so captures are naturally
deterministic.

Usage (GPU machine):
    uv run python -m evaluation.flagattn_capture               # all cases
    uv run python -m evaluation.flagattn_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECS_PATH = Path(__file__).parent / "kernels" / "flagattn_specs.json"
PER_CASE_TIMEOUT_S = 600
UPSTREAM = "https://github.com/FlagOpen/FlagAttention"


# ── case table ───────────────────────────────────────────────────
# Each case: (family, bwd, run) — run(torch, device, dtype) builds small
# fp16 inputs, calls one public flag_attn op, returns output tensors
# (summed + .backward()'d by the shared driver when bwd).


def _flash_causal(torch, device, dtype):
    import flag_attn

    B, H, T, D = 2, 2, 128, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return [flag_attn.flash_attention(q, k, v, causal=True)]


def _flash_noncausal_scaled(torch, device, dtype):
    import flag_attn

    B, H, T, D = 2, 2, 128, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return [flag_attn.flash_attention(q, k, v, causal=False, sm_scale=0.5)]


def _flash_gqa(torch, device, dtype):
    import flag_attn

    B, Hq, Hk, T, D = 2, 4, 2, 128, 64
    q = torch.randn(B, Hq, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, Hk, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, Hk, T, D, device=device, dtype=dtype, requires_grad=True)
    return [flag_attn.flash_attention(q, k, v, causal=True)]


def _flash_dropout(torch, device, dtype):
    import flag_attn

    B, H, T, D = 2, 2, 128, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return [flag_attn.flash_attention(q, k, v, causal=False, dropout_p=0.5)]


def _flash_nondivisible(torch, device, dtype):
    import flag_attn

    # M=N=100 is not a multiple of any block size: DIVISIBLE_M/N are
    # False and every load/store in the kernels runs with masks on
    B, H, T, D = 2, 2, 100, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    return [flag_attn.flash_attention(q, k, v, causal=True)]


def _flash_aux_outputs(torch, device, dtype):
    import flag_attn

    # return_total_attention launches the extra _total_attention_kernel;
    # log_normalizer exposes the L buffer the bwd kernels re-read
    B, H, T, D = 2, 2, 128, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    outs = flag_attn.flash_attention(
        q,
        k,
        v,
        causal=True,
        return_log_normalizer=True,
        return_total_attention=True,
    )
    return list(outs)


def _splitkv_decode(torch, device, dtype):
    import flag_attn

    # decoding shape: M=1 query against a long KV
    B, H, N, D = 2, 2, 512, 64
    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    return [flag_attn.flash_attention_split_kv(q, k, v, causal=False)]


def _paged(torch, device, dtype):
    import flag_attn

    num_seqs, num_kv_heads, qgs, head_size = 2, 2, 1, 64
    kv_block_size, max_seq_len = 16, 128
    max_blocks = max_seq_len // kv_block_size
    num_blocks = num_seqs * max_blocks
    q = torch.randn(num_seqs, num_kv_heads * qgs, head_size, device=device, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, num_kv_heads, kv_block_size, head_size, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    context_lens = torch.tensor([100, 128], device=device, dtype=torch.int32)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        num_seqs, max_blocks
    )
    o = flag_attn.paged_attention(
        q,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        head_size**-0.5,
        max_seq_len,
    )
    return [o]


def _paged_v2(torch, device, dtype):
    import flag_attn

    # num_splits > 1 exercises the partition kernel + the v2 reduce
    # kernel that combines partial results across partitions
    num_seqs, num_kv_heads, qgs, head_size = 2, 2, 1, 64
    kv_block_size, max_seq_len = 16, 512
    max_blocks = max_seq_len // kv_block_size
    num_blocks = num_seqs * max_blocks
    q = torch.randn(num_seqs, num_kv_heads * qgs, head_size, device=device, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, num_kv_heads, kv_block_size, head_size, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    context_lens = torch.tensor([400, 512], device=device, dtype=torch.int32)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        num_seqs, max_blocks
    )
    o = flag_attn.paged_attention(
        q,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        head_size**-0.5,
        max_seq_len,
        num_splits=4,
    )
    return [o]


def _piecewise_causal(torch, device, dtype):
    import flag_attn

    B, H, T, D = 2, 2, 128, 64
    q1 = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k1 = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    q2 = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k2 = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    o = flag_attn.piecewise_attention(
        q1, k1, q2, k2, v, dist_threshold=T // 2, causal=True
    )
    return [o]


CASES: dict = {
    "flash_causal": ("flash", True, _flash_causal),
    "flash_noncausal_scaled": ("flash", True, _flash_noncausal_scaled),
    "flash_gqa": ("flash", True, _flash_gqa),
    "flash_dropout": ("flash", True, _flash_dropout),
    "flash_nondivisible": ("flash", True, _flash_nondivisible),
    "flash_aux_outputs": ("flash", True, _flash_aux_outputs),
    "splitkv_decode": ("split_kv", False, _splitkv_decode),
    "paged": ("paged", False, _paged),
    "paged_v2": ("paged", False, _paged_v2),
    "piecewise_causal": ("piecewise", True, _piecewise_causal),
}


def main() -> None:
    from evaluation.capture_common import capture_one_case, run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        result = capture_one_case(CASES, args.one, dtype_name="float16")
        args.out.write_text(json.dumps(result, indent=1))
        return

    from evaluation.runner import _flagattn_provenance

    prov = _flagattn_provenance()
    run_case_capture(
        "evaluation.flagattn_capture",
        CASES,
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "flag_attn": prov.get("flag_attn"),
            "upstream_commit": prov.get("flag_attn_commit"),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
