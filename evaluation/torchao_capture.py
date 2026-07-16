"""One-time GPU launch capture for the torchao corpus.

torchao is analyzed AS INSTALLED — pinned by a git pip install
(``USE_CPP=0 pip install "torchao @ git+https://github.com/pytorch/ao@
<commit>"``; the Triton kernels are pure Python so the C++ extension is
skipped), and ``runner._torchao_provenance()`` reads the exact commit
from pip's ``direct_url.json``. This module drives the hand-written
Triton kernel families through their public wrappers at small shapes
under the shared capture layer (capture_common.py).

Families captured (RTX 4090, sm89): attention QKV fp8 quantization
(rope/hadamard variants), MoE-training fp8 rowwise/jagged scaling
(incl. the atomic_min scales kernel) and mxfp8 scale swizzles,
DeepSeek-style 128-blockwise fp8 training quant+GEMM, DeepGEMM-layout
grouped quant, float8nocompile tensorwise casts (incl. the atomic_max
amax kernel), torchao/kernel blockwise fp8 + int8 matmuls + BSR addmm,
hqq int4 mixed mm, quantized-training int8 mm, split-k matmul, and the
sm89-reachable mx_formats kernels.

NOT capturable in this environment, recorded here for the sweep report:
- ``prototype/fp8_sdpa_inference`` (9 kernels): the package __init__
  imports torch-2.11 experimental FA3 symbols — unimportable on torch
  2.10, and the module path must be importable for rebuild.
- ``prototype/moe_training/nvfp4_training`` (11 kernels) and the
  mxfp8 CUDA/CuTeDSL quant ops: hard ``is_sm_at_least_100()`` gates
  (Blackwell TMA / FP4 cvt PTX).
- ``prototype/mx_formats`` dim0/dim1 quant kernels (5): defined only
  inside the sm100+CUDA12.8 gate block — not even created on sm89.
- ``moe_training/kernels/mxfp8/comms.py`` + the 8 ``triton_utils``
  device helpers: need torch.distributed symmetric-memory rendezvous.
- ``float8nocompile`` ``to_fp8_col_major_t``: dead code (no launch
  path anywhere upstream).
- The split-k ``tl.atomic_add`` branch of hqq mixed_mm / common matmul
  is reachable only when autotune benchmarks a SPLIT_K>1 config first;
  which config a capture records is autotune-order-dependent.
- ``common/triton/matmul`` with fp8 inputs: ``_call`` KeyErrors on any
  fp8 dtype (``supported_acc_dtypes`` has no fp8 entry, both branches)
  — the fp8 path is unreachable as installed.

Usage (GPU machine):
    uv run python -m evaluation.torchao_capture               # all cases
    uv run python -m evaluation.torchao_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECS_PATH = Path(__file__).parent / "kernels" / "torchao_specs.json"
PER_CASE_TIMEOUT_S = 600
UPSTREAM = "https://github.com/pytorch/ao"


# ── case table ───────────────────────────────────────────────────
# Each case: (family, bwd, run) — run(torch, device, dtype) builds small
# inputs, calls one public torchao wrapper, returns output tensors.


def _rope_tables(torch, S, D, device):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = torch.outer(torch.arange(S, dtype=torch.float32), freqs)
    cos = torch.cat([torch.cos(ang), torch.cos(ang)], dim=-1).to(device)
    sin = torch.cat([torch.sin(ang), torch.sin(ang)], dim=-1).to(device)
    return cos, sin


# ---- attention QKV fp8 quantization (layouts: rope [B,S,H,D], plain [B,H,S,D])


def _attnq_rope_qkv(torch, device, dtype):
    from torchao.prototype.attention.quantization.triton_rope_qkv_quantization import (
        triton_fp8_rope_sdpa_quantize,
    )

    B, S, H, D = 2, 128, 4, 64
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    cos, sin = _rope_tables(torch, S, D, device)
    return list(triton_fp8_rope_sdpa_quantize(q, k, v, cos, sin))


def _attnq_qkv(torch, device, dtype):
    from torchao.prototype.attention.quantization.triton_qkv_quantization import (
        triton_fp8_sdpa_quantize,
    )

    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return list(triton_fp8_sdpa_quantize(q, k, v))


def _attnq_qkv_gqa(torch, device, dtype):
    from torchao.prototype.attention.quantization.triton_qkv_quantization import (
        triton_fp8_sdpa_quantize,
    )

    B, S, D = 2, 128, 64
    q = torch.randn(B, 8, S, D, device=device, dtype=dtype)
    k = torch.randn(B, 2, S, D, device=device, dtype=dtype)
    v = torch.randn(B, 2, S, D, device=device, dtype=dtype)
    return list(triton_fp8_sdpa_quantize(q, k, v))


def _attnq_hadamard_rope(torch, device, dtype):
    from torchao.prototype.attention.quantization.triton_hadamard_rope_qkv_quantization import (  # noqa: E501
        triton_fp8_hadamard_rope_sdpa_quantize,
    )

    B, S, H, D = 2, 128, 4, 64
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    cos, sin = _rope_tables(torch, S, D, device)
    return list(triton_fp8_hadamard_rope_sdpa_quantize(q, k, v, cos, sin))


def _attnq_hadamard_qkv(torch, device, dtype):
    from torchao.prototype.attention.quantization.triton_hadamard_qkv_quantization import (  # noqa: E501
        triton_fp8_hadamard_sdpa_quantize,
    )

    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return list(triton_fp8_hadamard_sdpa_quantize(q, k, v))


def _attnq_inverse_hadamard(torch, device, dtype):
    from torchao.prototype.attention.quantization.triton_hadamard_utils import (
        inverse_hadamard_transform,
    )

    x = torch.randn(2, 4, 128, 64, device=device, dtype=dtype)
    return [inverse_hadamard_transform(x)]


# ---- MoE-training fp8 rowwise / jagged scaling + mxfp8 scale swizzles


def _moe_rowwise_3d_transpose(torch, device, dtype):
    from torchao.prototype.moe_training.kernels.float8_rowwise import (
        triton_fp8_rowwise_3d_transpose_rhs,
    )

    x = torch.randn(2, 128, 128, dtype=dtype, device=device)  # (E, K, N)
    return list(
        triton_fp8_rowwise_3d_transpose_rhs(
            x, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_rowwise_3d_fused_reduction(torch, device, dtype):
    from torchao.prototype.moe_training.kernels.float8_rowwise import (
        triton_fp8_rowwise_3d_transpose_rhs_fused_reduction,
    )

    x = torch.randn(2, 128, 128, dtype=dtype, device=device)
    return list(
        triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
            x, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_colwise_3d(torch, device, dtype):
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_colwise_3d_scale_and_cast,
    )

    x = torch.randn(2, 128, 128, dtype=dtype, device=device).transpose(-2, -1)
    return list(
        triton_fp8_colwise_3d_scale_and_cast(
            x, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_rowwise_2d(torch, device, dtype):
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_rowwise_2d_scale_and_cast,
    )

    x = torch.randn(128, 128, dtype=dtype, device=device)
    return list(
        triton_fp8_rowwise_2d_scale_and_cast(
            x, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_jagged_rowwise(torch, device, dtype):
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_per_group_rowwise_scales,
    )

    x = torch.randn(128, 256, dtype=dtype, device=device)
    offs = torch.tensor([128, 256], dtype=torch.int32, device=device)
    return list(
        triton_fp8_per_group_rowwise_scales(
            x, offs, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_jagged_colwise(torch, device, dtype):
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_per_group_colwise_scales,
    )

    # col-major (K, N); max group size 64 -> non-fused kernel
    x = torch.randn(128, 64, dtype=dtype, device=device).t().contiguous().t()
    offs = torch.tensor([64, 128], dtype=torch.int32, device=device)
    return list(
        triton_fp8_per_group_colwise_scales(
            x, offs, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_jagged_colwise_fused(torch, device, dtype):
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_per_group_colwise_scales,
    )

    # K=256, one group -> rounded max group size 256 -> fused kernel
    x = torch.randn(256, 64, dtype=dtype, device=device).t().contiguous().t()
    offs = torch.tensor([256], dtype=torch.int32, device=device)
    return list(
        triton_fp8_per_group_colwise_scales(
            x, offs, output_dtype=torch.float8_e4m3fn, round_scales_to_power_of_2=False
        )
    )


def _moe_jagged_colwise_dual(torch, device, dtype):
    from torchao.prototype.moe_training.kernels import (
        triton_fp8_per_group_colwise_scales_dual,
    )

    a = torch.randn(128, 64, dtype=dtype, device=device).t().contiguous().t()
    b = torch.randn(128, 96, dtype=dtype, device=device).t().contiguous().t()
    offs = torch.tensor([64, 128], dtype=torch.int32, device=device)
    return list(
        triton_fp8_per_group_colwise_scales_dual(
            a,
            b,
            offs,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=False,
        )
    )


def _moe_mx_swizzle_m(torch, device, dtype):
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_2d_M_groups,
    )

    scales = torch.randint(0, 256, (64, 8), dtype=torch.uint8, device=device)
    offs = torch.tensor([32, 64], dtype=torch.int32, device=device)
    return [triton_mx_block_rearrange_2d_M_groups(scales, offs)]


def _moe_mx_swizzle_3d(torch, device, dtype):
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_per_group_3d,
    )

    scales = torch.randint(0, 256, (2, 128, 4), dtype=torch.uint8, device=device)
    return [triton_mx_block_rearrange_per_group_3d(scales)]


def _moe_mx_swizzle_k(torch, device, dtype):
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_2d_K_groups,
    )

    scales = torch.randint(0, 256, (128, 8), dtype=torch.uint8, device=device)
    offs = torch.tensor([4, 8], dtype=torch.int32, device=device)
    return [triton_mx_block_rearrange_2d_K_groups(scales, offs)]


def _moe_permute_bwd(torch, device, dtype):
    from torchao.prototype.moe_training.ep.permute import _triton_permute_bwd

    grad = torch.randn(8, 16, dtype=dtype, device=device)
    idx = torch.tensor([3, -1, 0, 5, 2, -1, 1, 4], dtype=torch.int32, device=device)
    return [_triton_permute_bwd(grad, idx, 6, 16)]


def _moe_fill_indices(torch, device, dtype):
    from torchao.prototype.moe_training.ep.kernels import generate_permute_indices

    tokens = torch.tensor([4, 2, 1, 3, 1, 2, 3, 4], dtype=torch.int32, device=device)
    outs = generate_permute_indices(tokens, 4, 2, 512, 32)
    return list(outs)


# ---- DeepSeek-style 128-blockwise fp8 training quant + GEMM


def _bw_gemm_1x128_128x128(torch, device, dtype):
    from torchao.float8.config import e4m3_dtype
    from torchao.prototype.blockwise_fp8_training.kernels import (
        triton_fp8_blockwise_act_quant_lhs,
        triton_fp8_blockwise_weight_quant_transposed_rhs,
        triton_fp8_gemm_1x128_128x128,
    )

    M, N, K = 256, 256, 256
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(N, K, dtype=dtype, device=device)
    A_q, A_s = triton_fp8_blockwise_act_quant_lhs(A, dtype=e4m3_dtype)
    B_t_q, B_t_s = triton_fp8_blockwise_weight_quant_transposed_rhs(B, dtype=e4m3_dtype)
    return [triton_fp8_gemm_1x128_128x128(A_q, B_t_q, A_s, B_t_s, out_dtype=dtype)]


def _bw_gemm_1x128_128x1(torch, device, dtype):
    from torchao.float8.config import e4m3_dtype
    from torchao.prototype.blockwise_fp8_training.kernels import (
        triton_fp8_blockwise_act_quant_rhs,
        triton_fp8_blockwise_act_quant_transposed_lhs,
        triton_fp8_gemm_1x128_128x1,
    )

    M, N, K = 256, 256, 256
    A = torch.randn(K, M, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    A_t_q, A_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(A, dtype=e4m3_dtype)
    B_q, B_s = triton_fp8_blockwise_act_quant_rhs(B, dtype=e4m3_dtype)
    return [triton_fp8_gemm_1x128_128x1(A_t_q, B_q, A_t_s, B_s, out_dtype=dtype)]


def _bw_weight_quant_rhs(torch, device, dtype):
    from torchao.float8.config import e4m3_dtype
    from torchao.prototype.blockwise_fp8_training.kernels import (
        triton_fp8_blockwise_weight_quant_rhs,
    )

    x = torch.randn(256, 256, dtype=dtype, device=device)
    return list(triton_fp8_blockwise_weight_quant_rhs(x, dtype=e4m3_dtype))


# ---- DeepGEMM-layout grouped quantization


def _dg_weight_quant_transposed(torch, device, dtype):
    from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
        triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm,
    )

    weight = torch.randn(3, 256, 384, dtype=dtype, device=device)  # (E, N, K)
    B_t = weight.contiguous().transpose(-2, -1)
    return list(triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm(B_t))


def _dg_weight_quant_grouped(torch, device, dtype):
    from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
        triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm,
    )

    weight = torch.randn(3, 256, 384, dtype=dtype, device=device)
    B_t = weight.contiguous().transpose(-2, -1)
    return list(triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm(B_t))


def _dg_act_quant_grouped_generic(torch, device, dtype):
    from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
        triton_fp8_blockwise_act_quant_k_grouped_deepgemm,
    )

    # D % 128 != 0 -> generic kernel
    offs = torch.tensor([256, 512, 640], dtype=torch.int32, device=device)
    x = torch.randn(640, 64, dtype=dtype, device=device)
    return list(triton_fp8_blockwise_act_quant_k_grouped_deepgemm(x, offs))


def _dg_act_quant_grouped_compact(torch, device, dtype):
    from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
        triton_fp8_blockwise_act_quant_k_grouped_deepgemm,
    )

    # D % 128 == 0 -> compact kernel
    offs = torch.tensor([256, 640, 768], dtype=torch.int32, device=device)
    x = torch.randn(768, 384, dtype=dtype, device=device)
    return list(triton_fp8_blockwise_act_quant_k_grouped_deepgemm(x, offs))


# ---- float8nocompile tensorwise casts (atomic_max amax vs reduction)


def _f8nc(torch, device, dtype, fn_name, **kwargs):
    import torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise as m
    from torchao.float8.float8_training_tensor import LinearMMConfig

    x = torch.randn(32, 16, dtype=dtype, device=device)
    out = getattr(m, fn_name)(x, torch.float8_e4m3fn, LinearMMConfig(), **kwargs)
    return list(out) if isinstance(out, tuple) else [out]


def _f8nc_row_major_atomic(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_row_major")


def _f8nc_row_major_reduction(torch, device, dtype):
    import torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise as m

    return _f8nc(
        torch, device, dtype, "hp_to_fp8_row_major", algo=m.KernelAlgorithm.REDUCTION
    )


def _f8nc_row_major_t(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_row_major_t")


def _f8nc_col_major(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_col_major")


def _f8nc_col_major_t(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_col_major_t")


def _f8nc_row_and_col(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_row_and_col_major")


def _f8nc_row_major_t_non_t(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_row_major_t_and_non_t")


def _f8nc_col_major_t_non_t(torch, device, dtype):
    return _f8nc(torch, device, dtype, "hp_to_fp8_col_major_t_and_non_t")


# ---- torchao/kernel: blockwise fp8, int matmuls, BSR addmm


def _k_blockwise_gemm(torch, device, dtype):
    from torchao.kernel.blockwise_quantization import (
        blockwise_fp8_gemm,
        fp8_blockwise_act_quant,
        fp8_blockwise_weight_quant,
    )

    A = torch.randn(128, 128, device=device)
    B = torch.randn(512, 128, device=device)
    A_q, A_s = fp8_blockwise_act_quant(A)
    B_q, B_s = fp8_blockwise_weight_quant(B)
    return [blockwise_fp8_gemm(A_q, A_s, B_q, B_s)]


def _k_blockwise_dequant(torch, device, dtype):
    from torchao.kernel.blockwise_quantization import (
        fp8_blockwise_weight_dequant,
        fp8_blockwise_weight_quant,
    )

    x = torch.randn(256, 256, device=device)
    qx, s = fp8_blockwise_weight_quant(x)
    return [fp8_blockwise_weight_dequant(qx, s)]


# torchao's own intmm autotuner (get_best_config_fn) exhaustively
# BENCHMARKS its config table on first launch — >10 min per op, so the
# capture drives the module-level single-config wrappers the autotuner
# itself dispatches to, with one config from its int8_mm_kernel_configs
# table


def _intmm_config(triton):
    return triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=2,
        num_warps=4,
    )


def _k_int_matmul(torch, device, dtype):
    import triton
    from torchao.kernel.intmm_triton import int_matmul_kernel

    a = torch.randint(-8, 8, (128, 64), dtype=torch.int8, device=device)
    b = torch.randint(-8, 8, (64, 16), dtype=torch.int8, device=device)
    c = torch.empty((128, 16), dtype=torch.int32, device=device)
    return [int_matmul_kernel(a, b, c, _intmm_config(triton))]


def _k_int_scaled_matmul(torch, device, dtype):
    import triton
    from torchao.kernel.intmm_triton import int_scaled_matmul_kernel

    a = torch.randint(-8, 8, (128, 64), dtype=torch.int8, device=device)
    b = torch.randint(-8, 8, (64, 16), dtype=torch.int8, device=device)
    scales = torch.randn(128, 1, dtype=dtype, device=device)
    c = torch.empty((128, 16), dtype=scales.dtype, device=device)
    return [int_scaled_matmul_kernel(a, b, scales, c, _intmm_config(triton))]


def _k_bsr_dense_addmm(torch, device, dtype):
    from torchao.kernel.bsr_triton_ops import bsr_dense_addmm

    M, K, N, BM = 128, 128, 128, 16
    W = torch.randn(M, K, dtype=torch.float16, device=device)
    W.view(M // BM, BM, K // BM, BM)[::2, :, ::2, :] = 0.0
    bsr = W.to_sparse_bsr(blocksize=BM)
    dense = torch.randn(K, N, dtype=torch.float16, device=device)
    inp = torch.zeros(M, N, dtype=torch.float16, device=device)
    return [bsr_dense_addmm(inp, bsr, dense, beta=0, alpha=1)]


# ---- hqq int4 mixed mm, int8 scaled mm, split-k matmul, mx_formats


def _hqq_mixed_mm(torch, device, dtype):
    from torchao.prototype.hqq import pack_2xint4, triton_mixed_mm

    M, N, K = 16, 128, 128
    group_size = 128
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    W_q = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=device)
    packed_w = pack_2xint4(W_q.T)
    n_groups = K // group_size
    scales = torch.randn(N, n_groups, dtype=torch.float16, device=device).abs() + 0.1
    zeros = torch.zeros(N, n_groups, dtype=torch.float16, device=device)
    return [
        triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            group_size=group_size,
            transposed=False,
            fp8_fast_accum=False,
            kernel_type="compute_bound",
        )
    ]


def _int8_scaled_mm(torch, device, dtype):
    from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm

    M, N, K = 256, 256, 256
    A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
    row_scale = torch.randn(M, 1, device=device)
    col_scale = torch.randn(1, N, device=device)
    return [scaled_int8_mm(A, B, row_scale, col_scale)]


def _common_matmul_fp16(torch, device, dtype):
    from torchao.prototype.common.triton.matmul import matmul

    a = torch.randn(256, 512, device=device, dtype=torch.float16)
    b = torch.randn(512, 256, device=device, dtype=torch.float16)
    return [matmul(a, b)]


def _mx_dequant_dim0(torch, device, dtype):
    from torchao.prototype.mx_formats.kernels import triton_mxfp8_dequant_dim0
    from torchao.prototype.mx_formats.mx_tensor import to_mx

    x = torch.randn(128, 128, dtype=dtype, device=device)
    scale_e8m0, x_data = to_mx(x, torch.float8_e4m3fn, 32)
    return [triton_mxfp8_dequant_dim0(x_data, scale_e8m0, dtype, 32)]


def _mx_block_rearrange(torch, device, dtype):
    from torchao.prototype.mx_formats.kernels import triton_mx_block_rearrange

    scales = torch.randint(0, 256, (128, 4), device=device, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return [triton_mx_block_rearrange(scales)]


CASES: dict = {
    "attnq_rope_qkv": ("attn_quant", False, _attnq_rope_qkv),
    "attnq_qkv": ("attn_quant", False, _attnq_qkv),
    "attnq_qkv_gqa": ("attn_quant", False, _attnq_qkv_gqa),
    "attnq_hadamard_rope": ("attn_quant", False, _attnq_hadamard_rope),
    "attnq_hadamard_qkv": ("attn_quant", False, _attnq_hadamard_qkv),
    "attnq_inverse_hadamard": ("attn_quant", False, _attnq_inverse_hadamard),
    "moe_rowwise_3d_transpose": ("moe_scales", False, _moe_rowwise_3d_transpose),
    "moe_rowwise_3d_fused_reduction": (
        "moe_scales",
        False,
        _moe_rowwise_3d_fused_reduction,
    ),
    "moe_colwise_3d": ("moe_scales", False, _moe_colwise_3d),
    "moe_rowwise_2d": ("moe_scales", False, _moe_rowwise_2d),
    "moe_jagged_rowwise": ("moe_scales", False, _moe_jagged_rowwise),
    "moe_jagged_colwise": ("moe_scales", False, _moe_jagged_colwise),
    "moe_jagged_colwise_fused": ("moe_scales", False, _moe_jagged_colwise_fused),
    "moe_jagged_colwise_dual": ("moe_scales", False, _moe_jagged_colwise_dual),
    "moe_mx_swizzle_m": ("moe_scales", False, _moe_mx_swizzle_m),
    "moe_mx_swizzle_3d": ("moe_scales", False, _moe_mx_swizzle_3d),
    "moe_mx_swizzle_k": ("moe_scales", False, _moe_mx_swizzle_k),
    "moe_permute_bwd": ("moe_scales", False, _moe_permute_bwd),
    "moe_fill_indices": ("moe_scales", False, _moe_fill_indices),
    "bw_gemm_1x128_128x128": ("blockwise_fp8", False, _bw_gemm_1x128_128x128),
    "bw_gemm_1x128_128x1": ("blockwise_fp8", False, _bw_gemm_1x128_128x1),
    "bw_weight_quant_rhs": ("blockwise_fp8", False, _bw_weight_quant_rhs),
    "dg_weight_quant_transposed": ("deepgemm", False, _dg_weight_quant_transposed),
    "dg_weight_quant_grouped": ("deepgemm", False, _dg_weight_quant_grouped),
    "dg_act_quant_grouped_generic": (
        "deepgemm",
        False,
        _dg_act_quant_grouped_generic,
    ),
    "dg_act_quant_grouped_compact": (
        "deepgemm",
        False,
        _dg_act_quant_grouped_compact,
    ),
    "f8nc_row_major_atomic": ("float8nocompile", False, _f8nc_row_major_atomic),
    "f8nc_row_major_reduction": ("float8nocompile", False, _f8nc_row_major_reduction),
    "f8nc_row_major_t": ("float8nocompile", False, _f8nc_row_major_t),
    "f8nc_col_major": ("float8nocompile", False, _f8nc_col_major),
    "f8nc_col_major_t": ("float8nocompile", False, _f8nc_col_major_t),
    "f8nc_row_and_col": ("float8nocompile", False, _f8nc_row_and_col),
    "f8nc_row_major_t_non_t": ("float8nocompile", False, _f8nc_row_major_t_non_t),
    "f8nc_col_major_t_non_t": ("float8nocompile", False, _f8nc_col_major_t_non_t),
    "k_blockwise_gemm": ("kernel_ops", False, _k_blockwise_gemm),
    "k_blockwise_dequant": ("kernel_ops", False, _k_blockwise_dequant),
    "k_int_matmul": ("kernel_ops", False, _k_int_matmul),
    "k_int_scaled_matmul": ("kernel_ops", False, _k_int_scaled_matmul),
    "k_bsr_dense_addmm": ("kernel_ops", False, _k_bsr_dense_addmm),
    "hqq_mixed_mm": ("matmul", False, _hqq_mixed_mm),
    "int8_scaled_mm": ("matmul", False, _int8_scaled_mm),
    "common_matmul_fp16": ("matmul", False, _common_matmul_fp16),
    "mx_dequant_dim0": ("mx", False, _mx_dequant_dim0),
    "mx_block_rearrange": ("mx", False, _mx_block_rearrange),
}


def main() -> None:
    from evaluation.capture_common import capture_one_case, run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        result = capture_one_case(
            CASES, args.one, dtype_name="bfloat16", module_prefix="torchao."
        )
        args.out.write_text(json.dumps(result, indent=1))
        return

    from evaluation.runner import _torchao_provenance

    prov = _torchao_provenance()
    run_case_capture(
        "evaluation.torchao_capture",
        CASES,
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "torchao": prov.get("torchao"),
            "upstream_commit": prov.get("torchao_commit"),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
