import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def tiled_attention_kernel(q, k, v, out):
    """Compute tiled attention ``softmax((q @ k.T) / sqrt(d)) @ v``.

    Args:
        q: Query tensor with shape ``[batch, num_heads, m, d]``.
        k: Key tensor with shape ``[batch, num_heads, n, d]``.
        v: Value tensor with shape ``[batch, num_heads, n, d_value]``.
        out: Output tensor with shape ``[batch, num_heads, m, d_value]`` written in place.
    """
    batch, num_heads, m_size, d_size = q.shape
    batch_k, num_heads_k, n_size, d_size_k = k.shape
    batch_v, num_heads_v, n_size_v, dv_size = v.shape

    assert batch_k == batch and batch_v == batch
    assert num_heads_k == num_heads and num_heads_v == num_heads
    assert d_size == d_size_k, "q and k contraction dims must match"
    assert n_size == n_size_v, "k and v sequence dims must match"
    assert out.shape == (batch, num_heads, m_size, dv_size)

    tile_m = nl.tile_size.gemm_stationary_fmax
    tile_d = nl.tile_size.pmax
    tile_n = nl.tile_size.gemm_moving_fmax
    assert d_size == tile_d, f"Expected d_size ({d_size}) == {tile_d}"
    assert (
        m_size % tile_m == 0
    ), f"Expected m_size ({m_size}) to be a multiple of {tile_m}"
    assert n_size <= tile_d, f"Expected n_size ({n_size}) <= {tile_d}"
    assert n_size <= tile_n, f"Expected n_size ({n_size}) <= {tile_n}"

    inv_sqrt_d = 1.0 / nl.sqrt(float(d_size))

    for batch_idx in nl.affine_range(batch):
        for head_idx in nl.affine_range(num_heads):
            k_tile = nl.ndarray((n_size, d_size), dtype=k.dtype, buffer=nl.sbuf)
            k_t = nl.ndarray((d_size, n_size), dtype=k.dtype, buffer=nl.psum)
            v_tile = nl.ndarray((n_size, dv_size), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_tile, src=k[batch_idx, head_idx, :, :])
            nisa.dma_copy(dst=v_tile, src=v[batch_idx, head_idx, :, :])
            nisa.nc_transpose(dst=k_t, data=k_tile)

            for m_tile_idx in nl.affine_range(m_size // tile_m):
                m_start = m_tile_idx * tile_m
                q_tile = nl.ndarray((tile_m, d_size), dtype=q.dtype, buffer=nl.sbuf)
                q_t = nl.ndarray((d_size, tile_m), dtype=q.dtype, buffer=nl.psum)
                nisa.dma_copy(
                    dst=q_tile,
                    src=q[batch_idx, head_idx, nl.ds(m_start, tile_m), :],
                )
                nisa.nc_transpose(dst=q_t, data=q_tile)

                scores_psum = nl.ndarray(
                    (tile_m, n_size), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(dst=scores_psum, stationary=q_t, moving=k_t)

                scores = nl.ndarray((tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores, src=scores_psum)
                nisa.tensor_scalar(
                    dst=scores, data=scores, op0=nl.multiply, operand0=inv_sqrt_d
                )

                row_max = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=row_max, op=nl.maximum, data=scores, axis=1, keepdims=True
                )

                centered = nl.ndarray(
                    (tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    dst=centered,
                    data1=scores,
                    data2=row_max.broadcast_to((tile_m, n_size)),
                    op=nl.subtract,
                )

                exp_scores = nl.ndarray(
                    (tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(dst=exp_scores, op=nl.exp, data=centered)

                row_sum = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=row_sum, op=nl.add, data=exp_scores, axis=1, keepdims=True
                )

                inv_sum = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum, data=row_sum)

                probs = nl.ndarray((tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=probs,
                    data1=exp_scores,
                    data2=inv_sum.broadcast_to((tile_m, n_size)),
                    op=nl.multiply,
                )

                probs_t = nl.ndarray((n_size, tile_m), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=probs_t, data=probs)

                out_psum = nl.ndarray(
                    (tile_m, dv_size), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(dst=out_psum, stationary=probs_t, moving=v_tile)

                out_tile = nl.ndarray(
                    (tile_m, dv_size), dtype=out.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=out_tile, src=out_psum)
                nisa.dma_copy(
                    dst=out[batch_idx, head_idx, nl.ds(m_start, tile_m), :],
                    src=out_tile,
                )


def _numpy_tiled_attention(q, k, v):
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(float(q.shape[-1]))
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return probs @ v


def _run_with_xla(kernel, kernel_grid, *arrays):
    """Run one beta2 kernel invocation on an XLA device."""
    import torch
    from torch_xla.core import xla_model as xm

    device = xm.xla_device()
    tensors = [torch.as_tensor(array, device=device) for array in arrays]
    compiled_kernel = nki.jit(kernel, kernel_return=False)
    compiled_kernel[kernel_grid](*tensors)
    xm.mark_step()
    return [tensor.cpu().numpy() for tensor in tensors]


def _run_demo():
    """Run attention with q ``[1, 1, 256, 128]``, k ``[1, 1, 128, 128]``, v ``[1, 1, 128, 64]``."""
    kernel_grid = (1, 1, 1)
    batch = 1
    num_heads = 1
    d_size = 128
    m_size = 256
    n_size = 128
    dv_size = 64

    q = np.linspace(
        -1.0,
        1.0,
        batch * num_heads * m_size * d_size,
        dtype=np.float32,
    ).reshape(batch, num_heads, m_size, d_size)
    k = np.linspace(
        1.0,
        -1.0,
        batch * num_heads * n_size * d_size,
        dtype=np.float32,
    ).reshape(batch, num_heads, n_size, d_size)
    v = np.linspace(
        -0.5,
        0.5,
        batch * num_heads * n_size * dv_size,
        dtype=np.float32,
    ).reshape(batch, num_heads, n_size, dv_size)
    out = np.empty((batch, num_heads, m_size, dv_size), dtype=np.float32)
    kernel_args = (q, k, v, out)
    expected = _numpy_tiled_attention(q, k, v)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki")(
            tiled_attention_kernel
        )
        traced_kernel[kernel_grid](*kernel_args)
        assert np.allclose(expected, out, atol=2e-4, rtol=2e-4)
        print("actual equals expected")
        triton_viz.launch(share=False)
    else:
        _, _, _, out = _run_with_xla(tiled_attention_kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected, out, atol=2e-4, rtol=2e-4)
        print("actual equals expected")


if __name__ == "__main__":
    _run_demo()
