"""Minimal single-head scaled dot-product attention NKI holdout.

This is intentionally a small, correct NKI kernel with the real attention
dataflow:

    scores = Q @ K^T / sqrt(d)
    probs  = softmax(scores)
    out    = probs @ V

It is not a flash-attention implementation: there is no block-wise online
softmax or KV streaming.  It is a structural holdout for two-stage TensorE
matmuls separated by a softmax normalization pass, while the existing
operator holdouts cover elementwise/norm/reduction/matmul families.

The kernel uses only source-level ``nl.*`` primitives so the exact same
``@nki.jit`` function can be CPU-traced by Triton-Viz and benchmarked on
hardware.  The Q and K operands are loaded PF-transposed, matching the natural
TensorE layout of ``Q @ K^T`` (head dimension on the partition/contraction
axis), and the second matmul uses the high-level ``nl.matmul`` API whose
compiler-inserted stationary transpose is part of the real attention dataflow.
"""

import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_scaled_dot_attention_kernel(q, k, v):
    """Compute ``softmax(q @ k.T / sqrt(d)) @ v`` for one attention head."""
    seq, d = q.shape
    assert seq == k.shape[0] == v.shape[0]
    assert d == k.shape[1] == v.shape[1]
    assert 1 <= seq <= 128
    assert 1 <= d <= 128

    out = nl.ndarray((seq, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # Attention's natural TensorE layout transposes Q and K into (d, seq)
    # SBUF tiles during the HBM load, putting the head dimension (the
    # contraction of Q @ K^T) on the partition axis.
    q_t = nl.load_transpose2d(q)
    k_t = nl.load_transpose2d(k)
    v_tile = nl.load(v[...])

    # scores = Q @ K^T. ``transpose_x=True`` declares that the stationary
    # operand is already in the transposed (d, seq) layout, so no extra
    # source-level transpose is needed.
    scores_psum = nl.matmul(q_t, k_t, transpose_x=True)
    scores = nl.copy(scores_psum, dtype=nl.float32)

    scale = 1.0 / math.sqrt(d)
    scores_scaled = nl.multiply(scores, scale)
    row_max = nl.max(scores_scaled, axis=1, keepdims=True)
    centered = nl.subtract(scores_scaled, row_max)
    exp_scores = nl.exp(centered)
    row_sum = nl.sum(exp_scores, axis=1, keepdims=True)
    probs_fp32 = nl.divide(exp_scores, row_sum)
    # Cast the softmax probabilities to the operand dtype so the second
    # TensorE matmul is homogeneous (bf16 @ bf16 or fp32 @ fp32), matching the
    # cast idiom used by the Tilebench softmax/matmul epilogues.
    probs = nl.add(probs_fp32, 0.0, dtype=q.dtype)

    # out = probs @ V. The compiler inserts the stationary transpose for this
    # high-level matmul, completing the two-stage TensorE attention dataflow.
    out_psum = nl.matmul(probs, v_tile)
    out_sbuf = nl.copy(out_psum, dtype=q.dtype)
    nl.store(out[...], value=out_sbuf)
    return out


def run(q, k, v, **kwargs):
    """Tilebench-compatible entrypoint used by the hardware reference."""
    return nki_scaled_dot_attention_kernel[(1,)](q, k, v)


def get_last_config() -> dict | None:
    return None
