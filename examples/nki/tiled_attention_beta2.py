import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def tiled_attention_kernel(q_t, k_t, v, out):
    """Compute tiled attention: softmax((q @ k.T) / sqrt(d)) @ v."""
    d_size, m_size = q_t.shape
    d_size_k, n_size = k_t.shape
    n_size_v, dv_size = v.shape
    assert d_size == d_size_k == n_size_v, "q_t, k_t, and v contraction dims must match"

    tile_m = nl.tile_size.gemm_stationary_fmax
    tile_d = nl.tile_size.pmax
    tile_n = nl.tile_size.gemm_moving_fmax
    assert d_size == tile_d, f"Expected d_size ({d_size}) == {tile_d}"
    assert (
        m_size % tile_m == 0
    ), f"Expected m_size ({m_size}) to be a multiple of {tile_m}"
    assert n_size <= tile_d, f"Expected n_size ({n_size}) <= {tile_d}"
    assert n_size <= tile_n, f"Expected n_size ({n_size}) <= {tile_n}"

    inv_sqrt_d = 1.0 / np.sqrt(float(d_size))

    k_sb = nl.ndarray((d_size, n_size), dtype=k_t.dtype, buffer=nl.sbuf)
    v_sb = nl.ndarray((n_size, dv_size), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_sb, src=k_t)
    nisa.dma_copy(dst=v_sb, src=v)

    for m_tile_idx in nl.affine_range(m_size // tile_m):
        m_start = m_tile_idx * tile_m
        q_tile = nl.ndarray((d_size, tile_m), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q_t[:, nl.ds(m_start, tile_m)])

        scores_psum = nl.ndarray((tile_m, n_size), dtype=np.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=scores_psum, stationary=q_tile, moving=k_sb)

        scores = nl.ndarray((tile_m, n_size), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=scores, src=scores_psum)
        nisa.tensor_scalar(
            dst=scores, data=scores, op0=nl.multiply, operand0=inv_sqrt_d
        )

        row_max = nl.ndarray((tile_m, 1), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=row_max, op=nl.maximum, data=scores, axis=1, keepdims=True
        )

        centered = nl.ndarray((tile_m, n_size), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=centered,
            data1=scores,
            data2=row_max.broadcast_to((tile_m, n_size)),
            op=nl.subtract,
        )

        exp_scores = nl.ndarray((tile_m, n_size), dtype=np.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_scores, op=np.exp, data=centered)

        row_sum = nl.ndarray((tile_m, 1), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=row_sum, op=nl.add, data=exp_scores, axis=1, keepdims=True
        )

        inv_sum = nl.ndarray((tile_m, 1), dtype=np.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sum, data=row_sum)

        probs = nl.ndarray((tile_m, n_size), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=probs,
            data1=exp_scores,
            data2=inv_sum.broadcast_to((tile_m, n_size)),
            op=nl.multiply,
        )

        probs_t = nl.ndarray((n_size, tile_m), dtype=np.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=probs_t, data=probs)

        out_psum = nl.ndarray((tile_m, dv_size), dtype=np.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=out_psum, stationary=probs_t, moving=v_sb)

        out_tile = nl.ndarray((tile_m, dv_size), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_tile, src=out_psum)
        nisa.dma_copy(dst=out[nl.ds(m_start, tile_m), :], src=out_tile)


def _numpy_tiled_attention(q_t, k_t, v):
    scores = (q_t.T @ k_t) / np.sqrt(float(q_t.shape[0]))
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ v


def _run_demo():
    kernel_grid = (1, 1, 1)
    d_size = 128
    m_size = 256
    n_size = 128
    dv_size = 64

    q_t = np.linspace(-1.0, 1.0, d_size * m_size, dtype=np.float32).reshape(
        d_size, m_size
    )
    k_t = np.linspace(1.0, -1.0, d_size * n_size, dtype=np.float32).reshape(
        d_size, n_size
    )
    v = np.linspace(-0.5, 0.5, n_size * dv_size, dtype=np.float32).reshape(
        n_size, dv_size
    )
    out = np.empty((m_size, dv_size), dtype=np.float32)
    kernel_args = (q_t, k_t, v, out)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki")(
            tiled_attention_kernel
        )
        traced_kernel[kernel_grid](*kernel_args)
        triton_viz.launch(share=False)
    else:
        compiled_kernel = nki.jit(tiled_attention_kernel, kernel_return=False)
        nki.simulate_kernel(compiled_kernel[kernel_grid], *kernel_args)

    expected = _numpy_tiled_attention(q_t, k_t, v)
    assert np.allclose(expected, out, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    _run_demo()
