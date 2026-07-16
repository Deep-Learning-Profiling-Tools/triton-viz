"""One-time GPU launch capture for the tritonbench_meta corpus
(meta-pytorch/tritonbench — Meta's Triton operator benchmark suite;
distinct from thunlp/TritonBench = our tritonbench_g corpus).

tritonbench is analyzed AS INSTALLED — git-pinned pip install; the dist
version is a constant 0.0.1, so the corpus module hard-checks the
INSTALLED direct_url.json commit against the captured one instead.

Unlike the case tables of fla/flagattn/flaggems/torchao, capture drives
the upstream benchmark harness itself: each case instantiates one
``BenchmarkOperator`` with ``--only <impl> --num-inputs 1 --input-id 0
--test-only`` and runs it once. ``module_prefix="tritonbench."`` keeps
only the suite's OWN kernels — the same operators also benchmark
liger / inductor / vendor backends, which would duplicate surfaces we
already analyze (liger corpus) or are runtime codegen (excluded class).

The PAIRS table lists every impl that launches an own-Triton kernel on
sm89. Registry-DISABLED impls were each tried once under ``--force``
and removed only on a verified structural failure:
- addmm triton_addmm / gemm hstu_triton_matmul / decoding_attention
  triton_splitk / fp8_gemm_{blockwise,rowwise,rowwise_grouped} _triton:
  NameError on hstu / fmha (xformers) / cutlass-ck symbols — optional
  deps not installed, exactly why upstream disables them.
- addmm streamk_addmm + gemm streamk_matmul: the stream-k kernel takes
  host-side TensorDescriptor (TMA) args (capture/rebuild + reader
  support live on the M4 track) behind a ~13-min autotune.
- layer_norm triton_multi_cta_layer_norm: upstream module lacks the
  symbol on this triton (cluster launch support).
- grouped_gemm split_size_grouped_gemm_fprop_triton: runs but launches
  no tritonbench.* kernel (routes to pt2/aten internally).

NOT in the table, recorded here for the sweep report:
- blackwell_attentions(+_mxfp8), nvfp4_gemm: sm100-only families
  (tlx / gluon / autows / warp-spec / TMA persistent).
- gemm/addmm/fp8_gemm TMA + warp-spec + tlx + b200 variants: sm90+ or
  the tlx Triton fork.
- flex_attention, inductor_*, launch_latency, test_op: torch.compile
  codegen, degenerate nop kernels, or harness self-tests.
- cross_entropy, embedding, fused_linear_*, geglu, rope, swiglu, jsd,
  kl_div, jagged_layer_norm: liger/compile-only backends (the liger
  surface is already a corpus).
- gdn_fwd_h, mamba2_*: eager/compile/fla-package backends only (fla is
  already a corpus).
- custom_shape_attentions, decoding_attention (except triton_splitk),
  mixed_gemm, mx4_to_fp32, fp32_to_mx4, fp8_fused_quant_gemm_rowwise,
  ragged_attention: need cutedsl / fa2-fa3 / marlin / fbgemm / mslk /
  generative_recommenders — not installed.

Usage (GPU machine):
    uv run python -m evaluation.tritonbench_meta_capture           # all
    uv run python -m evaluation.tritonbench_meta_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECS_PATH = Path(__file__).parent / "kernels" / "tritonbench_meta_specs.json"
PER_CASE_TIMEOUT_S = 600
UPSTREAM = "https://github.com/meta-pytorch/tritonbench"


# ── case table ───────────────────────────────────────────────────
# (operator, impl, mode) — mode "fwd" or "fwd_bwd" (--fwd-bwd also
# captures the backward kernels, e.g. layer_norm's dw/db lock pattern).

PAIRS: tuple = (
    ("bf16xint16_gemm", "bf16xbf16", "fwd"),
    ("bf16xint16_gemm", "bf16xint16", "fwd"),
    ("bf16xint16_gemm", "bf16xint16_casted", "fwd"),
    ("flash_attention", "triton_tutorial_flash_v2", "fwd"),
    ("flash_attention", "triton_tutorial_flash_v2", "fwd_bwd"),
    ("flash_attention", "triton_tutorial_flash_v2_tma", "fwd"),
    ("fp8_attention", "triton_flash_v2", "fwd"),
    ("fp8_attention", "triton_flash_v2_tma", "fwd"),
    ("fp8_gemm", "triton_fp8_gemm", "fwd"),
    ("fp8_gemm", "triton_persistent_fp8_gemm", "fwd"),
    ("gather_gemv", "triton_gather_gemv", "fwd"),
    ("gdpa", "gdpa", "fwd"),
    ("gdpa", "gdpa_opt", "fwd"),
    ("gdpa", "gdpa_opt_sorted", "fwd"),
    ("gemm", "triton_tutorial_matmul", "fwd"),
    ("gemm", "matmul_partition_k", "fwd"),
    ("gemm", "triton_persistent_matmul", "fwd"),
    ("gemm", "triton_ops_matmul", "fwd"),
    ("grouped_gemm", "triton_grouped_gemm", "fwd"),
    ("int4_gemm", "triton_int4_gemm", "fwd"),
    ("int4_gemm", "preprocessed_triton_int4_gemm", "fwd"),
    ("jagged_mean", "triton_jagged_mean_simple_fused", "fwd"),
    ("jagged_mean", "triton_jagged_mean_variable_length_loop", "fwd"),
    ("jagged_softmax", "triton_jagged_softmax_simple_fused", "fwd"),
    ("jagged_softmax", "triton_jagged_softmax_variable_length_loop", "fwd"),
    ("jagged_sum", "triton_jagged_sum_no_pad_simple_fused", "fwd"),
    ("jagged_sum", "triton_jagged_sum_no_pad_variable_length_loop", "fwd"),
    ("layer_norm", "triton_layer_norm", "fwd"),
    ("layer_norm", "triton_layer_norm", "fwd_bwd"),
    ("layer_norm", "triton_fused_layer_norm", "fwd"),
    ("layer_norm", "triton_fused_layer_norm", "fwd_bwd"),
    ("low_mem_dropout", "triton_dropout", "fwd"),
    ("low_mem_dropout", "seeded_dropout", "fwd"),
    # rms_norm has NO fwd case: upstream RMSNorm.forward is a pure
    # torch reference — only its backward launches a Triton kernel
    ("rms_norm", "triton_fused_rmsnorm", "fwd_bwd"),
    ("softmax", "triton_softmax", "fwd"),
    ("softmax", "triton_softmax", "fwd_bwd"),
    ("sum", "triton_sum", "fwd"),
    ("template_attention", "test_no_exp2", "fwd"),
    ("template_attention", "test_with_exp2", "fwd"),
    ("vector_add", "triton_add", "fwd"),
    ("vector_exp", "triton_exp", "fwd"),
    ("welford", "triton_welford", "fwd"),
    ("welford", "test_no_welford", "fwd"),
)


def _tb_case(op: str, impl: str, fwd_bwd: bool):
    def run(torch, device, dtype):
        from tritonbench.operators import load_opbench_by_name
        from tritonbench.utils.parser import get_parser

        # --force: the registry disables impls for missing optional deps
        # or newer arches, and --only silently runs NOTHING for a
        # disabled impl; --force runs it anyway so sm89-plausible
        # disabled impls get a real try (failures are recorded)
        argv = [
            "--op",
            op,
            "--only",
            impl,
            "--num-inputs",
            "1",
            "--input-id",
            "0",
            "--test-only",
            "--force",
        ]
        if fwd_bwd:
            argv.append("--fwd-bwd")
        tb_args, extra = get_parser().parse_known_args(argv)
        opbench = load_opbench_by_name(op)(tb_args=tb_args, extra_args=extra)
        opbench.run()
        return []

    return run


CASES: dict = {
    f"{op}__{impl}" + ("__bwd" if mode == "fwd_bwd" else ""): (
        op,
        False,
        _tb_case(op, impl, mode == "fwd_bwd"),
    )
    for op, impl, mode in PAIRS
}


def main() -> None:
    from evaluation.capture_common import capture_one_case, run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        result = capture_one_case(
            CASES, args.one, dtype_name="bfloat16", module_prefix="tritonbench."
        )
        args.out.write_text(json.dumps(result, indent=1))
        return

    from evaluation.runner import _tritonbench_meta_provenance

    prov = _tritonbench_meta_provenance()
    run_case_capture(
        "evaluation.tritonbench_meta_capture",
        CASES,
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "tritonbench_meta": prov.get("tritonbench_meta"),
            "upstream_commit": prov.get("tritonbench_meta_commit"),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
