import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def _rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _rope_reference(q, k, cos, sin):
    cos_b = cos[np.newaxis, :, :]
    sin_b = sin[np.newaxis, :, :]
    out_q = q * cos_b + _rotate_half(q) * sin_b
    out_k = k * cos_b + _rotate_half(k) * sin_b
    return out_q, out_k


def _rope_tables(seq_len, head_dim, base=10000.0):
    positions = np.arange(seq_len, dtype=np.float32)[:, None]
    inv_freq = 1.0 / (
        base ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim))
    )
    freqs = positions * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rope_kernel(q, k, cos, sin, out_q, out_k):
    """Apply RoPE to q/k tiles using beta2 primitives."""
    heads, seq_len, head_dim = q.shape
    assert k.shape == q.shape
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    assert head_dim % 2 == 0

    tile_s = nl.tile_size.pmax
    assert (
        seq_len % tile_s == 0
    ), f"Expected seq_len ({seq_len}) to be a multiple of {tile_s}"
    half = head_dim // 2

    for h in nl.affine_range(heads):
        for seq_tile in nl.affine_range(seq_len // tile_s):
            seq_start = seq_tile * tile_s
            q_tile = nl.ndarray((tile_s, head_dim), dtype=q.dtype, buffer=nl.sbuf)
            k_tile = nl.ndarray((tile_s, head_dim), dtype=k.dtype, buffer=nl.sbuf)
            cos_tile = nl.ndarray((tile_s, head_dim), dtype=cos.dtype, buffer=nl.sbuf)
            sin_tile = nl.ndarray((tile_s, head_dim), dtype=sin.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_tile, src=q[h, nl.ds(seq_start, tile_s), :])
            nisa.dma_copy(dst=k_tile, src=k[h, nl.ds(seq_start, tile_s), :])
            nisa.dma_copy(dst=cos_tile, src=cos[nl.ds(seq_start, tile_s), :])
            nisa.dma_copy(dst=sin_tile, src=sin[nl.ds(seq_start, tile_s), :])

            q_out_tile = nl.ndarray(
                (tile_s, head_dim), dtype=out_q.dtype, buffer=nl.sbuf
            )
            k_out_tile = nl.ndarray(
                (tile_s, head_dim), dtype=out_k.dtype, buffer=nl.sbuf
            )

            q_out_tile[:, :half] = nl.subtract(
                nl.multiply(q_tile[:, :half], cos_tile[:, :half]),
                nl.multiply(q_tile[:, half:], sin_tile[:, :half]),
            )
            q_out_tile[:, half:] = nl.add(
                nl.multiply(q_tile[:, half:], cos_tile[:, half:]),
                nl.multiply(q_tile[:, :half], sin_tile[:, half:]),
            )
            k_out_tile[:, :half] = nl.subtract(
                nl.multiply(k_tile[:, :half], cos_tile[:, :half]),
                nl.multiply(k_tile[:, half:], sin_tile[:, :half]),
            )
            k_out_tile[:, half:] = nl.add(
                nl.multiply(k_tile[:, half:], cos_tile[:, half:]),
                nl.multiply(k_tile[:, :half], sin_tile[:, half:]),
            )

            nisa.dma_copy(dst=out_q[h, nl.ds(seq_start, tile_s), :], src=q_out_tile)
            nisa.dma_copy(dst=out_k[h, nl.ds(seq_start, tile_s), :], src=k_out_tile)


def _run_demo():
    kernel_grid = (1, 1, 1)
    heads = 4
    seq_len = 128
    head_dim = 64
    q = np.linspace(-1.0, 1.0, heads * seq_len * head_dim, dtype=np.float32).reshape(
        heads, seq_len, head_dim
    )
    k = np.linspace(1.0, -1.0, heads * seq_len * head_dim, dtype=np.float32).reshape(
        heads, seq_len, head_dim
    )
    cos, sin = _rope_tables(seq_len, head_dim)
    out_q = np.empty_like(q)
    out_k = np.empty_like(k)
    kernel_args = (q, k, cos, sin, out_q, out_k)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki")(rope_kernel)
        traced_kernel[kernel_grid](*kernel_args)
        triton_viz.launch(share=False)
    else:
        compiled_kernel = nki.jit(rope_kernel, kernel_return=False)
        nki.simulate_kernel(compiled_kernel[kernel_grid], *kernel_args)

    expected_q, expected_k = _rope_reference(q, k, cos, sin)
    assert np.allclose(expected_q, out_q, atol=1e-5, rtol=1e-5)
    assert np.allclose(expected_k, out_k, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    _run_demo()
