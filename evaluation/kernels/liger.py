"""Phase C corpus: a liger-kernel subset (plan S5) — production Triton
kernels analyzed AS INSTALLED (no vendoring).

Requires ``uv pip install liger-kernel`` (evaluation-only dependency, not
in pyproject); loading this corpus without it raises a clear error.

23 kernels across 15 ops, each at one representative launch, all labeled
race-free (production code). The point of this sweep is the LADDER
DISTRIBUTION on real code — which kernels prove, at which rung, and which
abstention kinds dominate (multi-loop row scans, pid-dependent loop
bounds); "unsupported dominating is itself the data".
"""

import torch

try:
    from liger_kernel.ops import (
        cross_entropy,
        geglu,
        group_norm,
        jsd,
        kl_div,
        layer_norm,
        poly_norm,
        relu_squared,
        rms_norm,
        softmax,
        sparsemax,
        swiglu,
        tvd,
        utils,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the liger corpus needs liger-kernel: uv pip install liger-kernel"
    ) from e

import triton.language as tl

from evaluation.spec import Corpus, LaunchSpec

CORPUS = Corpus("liger")

M, N = 8, 64  # rows x cols for the row-parallel ops
_G = lambda seed: torch.Generator().manual_seed(seed)  # noqa: E731


def _f32(shape, seed, positive=False):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(shape, generator=g) if positive else torch.randn(shape, generator=g)
    return t.float()


def _probs(shape, seed):
    t = _f32(shape, seed, positive=True) + 0.05
    return t / t.sum(dim=-1, keepdim=True)


def _add(name, kernel, signature, constexprs, make_args, grid, note):
    CORPUS.add(
        LaunchSpec(
            name=name,
            kernel_fn=kernel,
            signature=signature,
            constexprs=constexprs,
            make_args=make_args,
            grid=grid,
            expected="race-free",
            pattern="liger",
            params_note=note,
        )
    )


# ── rms_norm ─────────────────────────────────────────────────────

_add(
    "liger_rms_norm_fwd",
    rms_norm._rms_norm_forward_kernel,
    {
        "Y_ptr": "*fp32",
        "Y_row_stride": "i32",
        "X_ptr": "*fp32",
        "X_row_stride": "i32",
        "W_ptr": "*fp32",
        "W_row_stride": "i32",
        "RSTD_ptr": "*fp32",
        "RSTD_row_stride": "i32",
        "n_cols": "i32",
        "eps": "fp32",
        "offset": "fp32",
        "casting_mode": "constexpr",
        "elementwise_affine": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"casting_mode": 0, "elementwise_affine": True, "BLOCK_SIZE": 64},
    lambda seed: (
        torch.zeros(M * N),
        N,
        _f32(M * N, seed),
        N,
        _f32(N, seed + 1),
        1,
        torch.zeros(M),
        1,
        N,
        1e-6,
        0.0,
    ),  # fmt: skip
    (M,),
    "row-parallel RMS norm (llama casting)",
)

_add(
    "liger_rms_norm_bwd",
    rms_norm._rms_norm_backward_kernel,
    {
        "dY_ptr": "*fp32",
        "dY_row_stride": "i32",
        "dX_ptr": "*fp32",
        "dX_row_stride": "i32",
        "X_ptr": "*fp32",
        "X_row_stride": "i32",
        "X_dtype": "constexpr",
        "W_ptr": "*fp32",
        "W_row_stride": "i32",
        "RSTD_ptr": "*fp32",
        "RSTD_row_stride": "i32",
        "dW_ptr": "*fp32",
        "dW_row_stride": "i32",
        "n_rows": "i32",
        "n_cols": "i32",
        "offset": "fp32",
        "rows_per_program": "i32",
        "casting_mode": "constexpr",
        "elementwise_affine": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {
        "X_dtype": tl.float32,
        "casting_mode": 0,
        "elementwise_affine": True,
        "BLOCK_SIZE": 64,
    },
    lambda seed: (
        _f32(M * N, seed),
        N,
        torch.zeros(M * N),
        N,
        _f32(M * N, seed + 1),
        N,
        _f32(N, seed + 2),
        1,
        torch.ones(M),
        1,
        torch.zeros(4 * N),
        N,
        M,
        N,
        0.0,
        2,
    ),  # fmt: skip
    (4,),
    "per-program row slab (pid-dependent loop bounds expected to abstain)",
)


# ── layer_norm ───────────────────────────────────────────────────

_add(
    "liger_layer_norm_fwd",
    layer_norm._layer_norm_forward_kernel,
    {
        "Y_ptr": "*fp32",
        "Y_row_stride": "i32",
        "X_ptr": "*fp32",
        "X_row_stride": "i32",
        "W_ptr": "*fp32",
        "W_row_stride": "i32",
        "B_ptr": "*fp32",
        "B_row_stride": "i32",
        "Mean_ptr": "*fp32",
        "Mean_row_stride": "i32",
        "RSTD_ptr": "*fp32",
        "RSTD_row_stride": "i32",
        "n_cols": "i32",
        "eps": "fp32",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64},
    lambda seed: (
        torch.zeros(M * N),
        N,
        _f32(M * N, seed),
        N,
        _f32(N, seed + 1),
        1,
        _f32(N, seed + 2),
        1,
        torch.zeros(M),
        1,
        torch.zeros(M),
        1,
        N,
        1e-5,
    ),  # fmt: skip
    (M,),
    "row-parallel layer norm forward",
)

_add(
    "liger_layer_norm_bwd",
    layer_norm._layer_norm_backward_kernel,
    {
        "X_ptr": "*fp32",
        "stride_x": "i32",
        "W_ptr": "*fp32",
        "Mean_ptr": "*fp32",
        "stride_mean": "i32",
        "RSTD_ptr": "*fp32",
        "stride_rstd": "i32",
        "DX_ptr": "*fp32",
        "stride_dx": "i32",
        "DW_ptr": "*fp32",
        "stride_dw": "i32",
        "DB_ptr": "*fp32",
        "stride_db": "i32",
        "DY_ptr": "*fp32",
        "stride_dy": "i32",
        "n_rows": "i32",
        "n_cols": "i32",
        "rows_per_program": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"rows_per_program": 2, "BLOCK_SIZE": 64},
    lambda seed: (
        _f32(M * N, seed),
        N,
        _f32(N, seed + 1),
        torch.zeros(M),
        1,
        torch.ones(M),
        1,
        torch.zeros(M * N),
        N,
        torch.zeros(4 * N),
        N,
        torch.zeros(4 * N),
        N,
        _f32(M * N, seed + 2),
        N,
        M,
        N,
    ),  # fmt: skip
    (4,),
    "per-program row slab writing per-program dW/dB stripes",
)


# ── geglu / swiglu / relu² / element-mul ─────────────────────────

_add(
    "liger_geglu_tanh_fwd",
    geglu._geglu_tanh_forward_kernel,
    {
        "a": "*fp32",
        "b": "*fp32",
        "c": "*fp32",
        "stride": "i32",
        "n_cols": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"n_cols": N, "BLOCK_SIZE": 64},
    lambda seed: (_f32(M * N, seed), _f32(M * N, seed + 1), torch.zeros(M * N), N),
    (M,),
    "gated GELU forward (tanh approximation in value position)",
)

_add(
    "liger_geglu_tanh_bwd",
    geglu._geglu_tanh_backward_kernel,
    {
        "dc": "*fp32",
        "a": "*fp32",
        "b": "*fp32",
        "stride": "i32",
        "n_cols": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"n_cols": N, "BLOCK_SIZE": 64},
    lambda seed: (_f32(M * N, seed), _f32(M * N, seed + 1), _f32(M * N, seed + 2), N),
    (M,),
    "backward writes grads IN PLACE into a/b (per-row disjoint)",
)

_add(
    "liger_swiglu_fwd",
    swiglu._swiglu_forward_kernel,
    {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "stride": "i32",
        "gate_multiplier": "fp32",
        "n_cols": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"n_cols": N, "BLOCK_SIZE": 64},
    lambda seed: (
        _f32(M * N, seed),
        _f32(M * N, seed + 1),
        torch.zeros(M * N),
        N,
        1.0,
    ),
    (M,),
    "SiLU-gated MLP forward",
)

_add(
    "liger_swiglu_bwd",
    swiglu._swiglu_backward_kernel,
    {
        "dc_ptr": "*fp32",
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "stride": "i32",
        "gate_multiplier": "fp32",
        "n_cols": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"n_cols": N, "BLOCK_SIZE": 64},
    lambda seed: (
        _f32(M * N, seed),
        _f32(M * N, seed + 1),
        _f32(M * N, seed + 2),
        N,
        1.0,
    ),
    (M,),
    "in-place grads into a/b",
)

_add(
    "liger_relu_squared_fwd",
    relu_squared._relu_squared_forward_kernel,
    {
        "Y_ptr": "*fp32",
        "Y_stride": "i32",
        "X_ptr": "*fp32",
        "X_stride": "i32",
        "n_cols": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"n_cols": N, "BLOCK_SIZE": 64},
    lambda seed: (torch.zeros(M * N), N, _f32(M * N, seed), N),
    (M,),
    "relu(x)^2 forward",
)

_add(
    "liger_relu_squared_bwd",
    relu_squared._relu_squared_backward_kernel,
    {
        "dX_ptr": "*fp32",
        "dX_stride": "i32",
        "dY_ptr": "*fp32",
        "dY_stride": "i32",
        "X_ptr": "*fp32",
        "X_stride": "i32",
        "n_cols": "constexpr",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"n_cols": N, "BLOCK_SIZE": 64},
    lambda seed: (
        torch.zeros(M * N),
        N,
        _f32(M * N, seed),
        N,
        _f32(M * N, seed + 1),
        N,
    ),
    (M,),
    "relu(x)^2 backward",
)

_add(
    "liger_element_mul",
    utils.element_mul_kernel,
    {
        "X_ptr": "*fp32",
        "X_stride": "i32",
        "grad_output_ptr": "*fp32",
        "n_cols": "i32",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64},
    lambda seed: (_f32(M * N, seed), N, _f32(1, seed + 1), N),
    (M,),
    "in-place scale by a broadcast scalar load",
)


# ── softmax family ───────────────────────────────────────────────

_SOFTMAX_FWD_SIG = {
    "Y_ptr": "*fp32", "Y_row_stride": "i32",
    "X_ptr": "*fp32", "X_row_stride": "i32",
    "n_cols": "i32", "BLOCK_SIZE": "constexpr",
}  # fmt: skip

_add(
    "liger_softmax_fwd_single",
    softmax._softmax_single_block_forward_kernel,
    _SOFTMAX_FWD_SIG,
    {"BLOCK_SIZE": 64},
    lambda seed: (torch.zeros(M * N), N, _f32(M * N, seed), N, N),
    (M,),
    "one block per row",
)

_add(
    "liger_softmax_bwd_single",
    softmax._softmax_single_block_backward_kernel,
    {
        "dy_ptr": "*fp32",
        "dy_stride": "i32",
        "y_ptr": "*fp32",
        "y_stride": "i32",
        "dx_ptr": "*fp32",
        "dx_stride": "i32",
        "n_cols": "i32",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64},
    lambda seed: (
        _f32(M * N, seed),
        N,
        _probs((M, N), seed + 1).reshape(-1),
        N,
        torch.zeros(M * N),
        N,
        N,
    ),  # fmt: skip
    (M,),
    "one block per row, backward",
)

_add(
    "liger_softmax_fwd_multi",
    softmax._softmax_multi_block_forward_kernel,
    _SOFTMAX_FWD_SIG,
    {"BLOCK_SIZE": 32},
    lambda seed: (torch.zeros(M * N), N, _f32(M * N, seed), N, N),
    (M,),
    "multi-block row: liger 0.8 uses tl.float32(...) as a callable, which "
    "triton 3.6 rejects — a recorded compile-error row (library/compiler "
    "version skew is itself sweep data)",
)


# ── sparsemax ────────────────────────────────────────────────────

_add(
    "liger_sparsemax_fwd",
    sparsemax._sparsemax_forward_kernel,
    {
        "x_ptr": "*fp32",
        "x_stride_row": "i32",
        "sorted_x_ptr": "*fp32",
        "sorted_x_stride_row": "i32",
        "o_ptr": "*fp32",
        "o_stride_row": "i32",
        "n_cols": "i32",
        "BLOCK_SIZE": "constexpr",
        "num_warps": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64, "num_warps": 4},
    lambda seed: (
        _f32(M * N, seed),
        N,
        torch.sort(
            _f32(M * N, seed).reshape(M, N), dim=-1, descending=True
        ).values.reshape(-1),
        N,
        torch.zeros(M * N),
        N,
        N,
    ),  # fmt: skip
    (M,),
    "threshold from the pre-sorted row (cumsum in value position)",
)

_add(
    "liger_sparsemax_bwd",
    sparsemax._sparsemax_backward_kernel,
    {
        "o_ptr": "*fp32",
        "go_ptr": "*fp32",
        "gi_ptr": "*fp32",
        "stride": "i32",
        "n_cols": "i32",
        "BLOCK_SIZE": "constexpr",
        "num_warps": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64, "num_warps": 4},
    lambda seed: (
        _probs((M, N), seed).reshape(-1),
        _f32(M * N, seed + 1),
        torch.zeros(M * N),
        N,
        N,
    ),
    (M,),
    "support-masked gradient",
)


# ── divergence losses ────────────────────────────────────────────

_add(
    "liger_kldiv_fwd",
    kl_div._kldiv_kernel_forward,
    {
        "y_ptr": "*fp32",
        "y_stride": "i32",
        "gt_ptr": "*fp32",
        "gt_stride": "i32",
        "loss_ptr": "*fp32",
        "loss_stride": "i32",
        "n_cols": "i32",
        "eps": "fp32",
        "BLOCK_SIZE": "constexpr",
        "log_target": "constexpr",
        "reduction": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 32, "log_target": False, "reduction": 3},
    lambda seed: (
        _probs((M, N), seed).log().reshape(-1),
        N,
        _probs((M, N), seed + 1).reshape(-1),
        N,
        torch.zeros(M * N),
        N,
        N,
        1e-10,
    ),  # fmt: skip
    (M,),
    "batchmean KL: column loop per row",
)

_add(
    "liger_kldiv_bwd",
    kl_div._kldiv_kernel_backward,
    {
        "target_ptr": "*fp32",
        "target_stride": "i32",
        "new_grads_ptr": "*fp32",
        "new_grads_stride": "i32",
        "n_cols": "i32",
        "BLOCK_SIZE": "constexpr",
        "log_target": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 32, "log_target": False},
    lambda seed: (
        _probs((M, N), seed).reshape(-1),
        N,
        torch.zeros(M * N),
        N,
        N,
    ),  # fmt: skip
    (M,),
    "KL backward column loop",
)

_add(
    "liger_tvd",
    tvd._tv_distance_kernel,
    {
        "p_ptr": "*fp32",
        "p_stride": "i32",
        "q_ptr": "*fp32",
        "q_stride": "i32",
        "loss_ptr": "*fp32",
        "loss_stride": "i32",
        "grads_ptr": "*fp32",
        "grads_stride": "i32",
        "label_ptr": "*i32",
        "ignore_index": "constexpr",
        "n_cols": "i32",
        "scale": "fp32",
        "BLOCK_SIZE": "constexpr",
        "HAS_LABEL": "constexpr",
        "reduction": "constexpr",
    },  # fmt: skip
    {"ignore_index": -100, "BLOCK_SIZE": 32, "HAS_LABEL": False, "reduction": 3},
    lambda seed: (
        _probs((M, N), seed).reshape(-1),
        N,
        _probs((M, N), seed + 1).reshape(-1),
        N,
        torch.zeros(M * N),
        N,
        torch.zeros(M * N),
        N,
        torch.zeros(M, dtype=torch.int32),
        N,
        1.0,
    ),  # fmt: skip
    (M,),
    "total variation distance with fused grads",
)

_add(
    "liger_jsd",
    jsd._jsd_kernel,
    {
        "X_ptr": "*fp32",
        "X_stride": "i32",
        "Y_ptr": "*fp32",
        "Y_stride": "i32",
        "loss_ptr": "*fp32",
        "loss_stride": "i32",
        "dX_ptr": "*fp32",
        "dX_stride": "i32",
        "label_ptr": "*i32",
        "beta": "constexpr",
        "n_non_ignore": "i32",
        "ignore_index": "constexpr",
        "n_cols": "i32",
        "BLOCK_SIZE": "constexpr",
        "HAS_LABEL": "constexpr",
    },  # fmt: skip
    {"beta": 0.5, "ignore_index": -100, "BLOCK_SIZE": 32, "HAS_LABEL": False},
    lambda seed: (
        _probs((M, N), seed).log().reshape(-1),
        N,
        _probs((M, N), seed + 1).log().reshape(-1),
        N,
        torch.zeros(M * N),
        N,
        torch.zeros(M * N),
        N,
        torch.zeros(M, dtype=torch.int32),
        M,
        N,
    ),  # fmt: skip
    (M,),
    "generalized JSD with fused grads",
)


# ── cross entropy (in-place gradient) ────────────────────────────

_add(
    "liger_cross_entropy",
    cross_entropy.liger_cross_entropy_kernel,
    {
        "X_ptr": "*fp32",
        "X_stride": "i32",
        "Y_ptr": "*i32",
        "Y_stride": "i32",
        "weight_ptr": "*fp32",
        "loss_ptr": "*fp32",
        "z_loss_ptr": "*fp32",
        "loss_stride": "i32",
        "token_accuracy_ptr": "*fp32",
        "token_accuracy_stride": "i32",
        "predicted_tokens_ptr": "*i32",
        "predicted_tokens_stride": "i32",
        "n_cols": "i32",
        "n_non_ignore": "fp32",
        "sum_non_ignore_weight": "fp32",
        "weight_sum": "fp32",
        "ignore_index": "i32",
        "lse_square_scale": "constexpr",
        "label_smoothing": "constexpr",
        "reduction": "constexpr",
        "softcap": "fp32",
        "RETURN_Z_LOSS": "constexpr",
        "RETURN_TOKEN_ACCURACY": "constexpr",
        "RETURN_PREDICTED_TOKENS": "constexpr",
        "BLOCK_SIZE": "constexpr",
        "HAS_WEIGHT": "constexpr",
        "HAS_SOFTCAPPING": "constexpr",
        "HAS_GRADIENTS": "constexpr",
    },  # fmt: skip
    {
        "lse_square_scale": 0.0,
        "label_smoothing": 0.0,
        "reduction": "mean",
        "RETURN_Z_LOSS": False,
        "RETURN_TOKEN_ACCURACY": False,
        "RETURN_PREDICTED_TOKENS": False,
        "BLOCK_SIZE": 32,
        "HAS_WEIGHT": False,
        "HAS_SOFTCAPPING": False,
        "HAS_GRADIENTS": True,
    },
    lambda seed: (
        _f32(M * N, seed),
        N,
        torch.randint(0, N, (M,), dtype=torch.int32, generator=_G(seed + 1)),
        1,
        torch.ones(N),
        torch.zeros(M),
        torch.zeros(M),
        1,
        torch.zeros(M),
        1,
        torch.zeros(M, dtype=torch.int32),
        1,
        N,
        float(M),
        float(M),
        float(N),
        -100,
        0.0,
    ),  # fmt: skip
    (M,),
    "fused CE writing the gradient IN PLACE into the logits row",
)


# ── poly / group norm ────────────────────────────────────────────

_add(
    "liger_poly_norm_fwd",
    poly_norm._poly_norm_forward_kernel,
    {
        "Y_ptr": "*fp32",
        "Y_row_stride": "i32",
        "X_ptr": "*fp32",
        "X_row_stride": "i32",
        "W_ptr": "*fp32",
        "B_ptr": "*fp32",
        "RSTD_ptr": "*fp32",
        "RSTD_row_stride": "i32",
        "n_cols": "i32",
        "eps": "fp32",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64},
    lambda seed: (
        torch.zeros(M * N),
        N,
        _f32(M * N, seed),
        N,
        _f32(3, seed + 1),
        _f32(1, seed + 2),
        torch.zeros(M * 3),
        3,
        N,
        1e-6,
    ),  # fmt: skip
    (M,),
    "x^3/x^2/x norms with cached rstd triple",
)

_add(
    "liger_group_norm_fwd",
    group_norm._group_norm_forward_kernel,
    {
        "Y_ptr": "*fp32",
        "Y_row_stride": "i32",
        "Y_col_stride": "i32",
        "X_ptr": "*fp32",
        "X_row_stride": "i32",
        "X_col_stride": "i32",
        "Mean_ptr": "*fp32",
        "Mean_row_stride": "i32",
        "Mean_col_stride": "i32",
        "RSTD_ptr": "*fp32",
        "RSTD_row_stride": "i32",
        "RSTD_col_stride": "i32",
        "W_ptr": "*fp32",
        "B_ptr": "*fp32",
        "hidden_size": "i32",
        "channels_per_group": "i32",
        "eps": "fp32",
        "BLOCK_SIZE": "constexpr",
    },  # fmt: skip
    {"BLOCK_SIZE": 64},
    # 4 batches x 2 groups x (2 channels x 32 hidden): 2-D grid
    lambda seed: (
        torch.zeros(4 * 2 * 2 * 32),
        2 * 2 * 32,
        2 * 32,
        _f32(4 * 2 * 2 * 32, seed),
        2 * 2 * 32,
        2 * 32,
        torch.zeros(4 * 2),
        2,
        1,
        torch.zeros(4 * 2),
        2,
        1,
        torch.ones(4),
        torch.zeros(4),
        32,
        2,
        1e-6,
    ),  # fmt: skip
    (4, 2),
    "2-D grid (batch x group), per-group hidden stripe",
)
