import math

import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def tiled_attention_kernel(
    q,
    k,
    v,
    out,
    batch,
    num_heads,
    num_heads_k,
    num_heads_v,
    m_size,
    n_size,
    d_size,
    dv_size,
):
    """Compute tiled attention over flattened 2D inputs.

    q shape: ``[batch * num_heads * m_size, d_size]``
    k shape: ``[batch * num_heads_k * n_size, d_size]``
    v shape: ``[batch * num_heads_v * n_size, dv_size]``
    out shape: ``[batch * num_heads * m_size, dv_size]``
    """
    assert num_heads % num_heads_k == 0 and num_heads % num_heads_v == 0
    assert q.shape == (batch * num_heads * m_size, d_size)
    assert k.shape == (batch * num_heads_k * n_size, d_size)
    assert v.shape == (batch * num_heads_v * n_size, dv_size)
    assert out.shape == (batch * num_heads * m_size, dv_size)

    tile_m = nl.tile_size.gemm_stationary_fmax
    tile_n = nl.tile_size.gemm_moving_fmax
    tile_p = nl.tile_size.pmax

    assert d_size == tile_p, f"Expected d_size ({d_size}) == {tile_p}"
    assert n_size <= tile_p, f"Expected n_size ({n_size}) <= {tile_p}"
    assert n_size <= tile_n, f"Expected n_size ({n_size}) <= {tile_n}"
    assert dv_size <= tile_n, f"Expected dv_size ({dv_size}) <= {tile_n}"
    assert (
        m_size % tile_m == 0
    ), f"Expected m_size ({m_size}) to be a multiple of {tile_m}"

    inv_sqrt_d = 1.0 / math.sqrt(float(d_size))

    for batch_idx in nl.affine_range(batch):
        for head_idx in nl.affine_range(num_heads):
            head_k = (head_idx * num_heads_k) // num_heads
            head_v = (head_idx * num_heads_v) // num_heads
            k_row_start = (batch_idx * num_heads_k + head_k) * n_size
            v_row_start = (batch_idx * num_heads_v + head_v) * n_size

            k_tile = nl.ndarray((n_size, d_size), dtype=k.dtype, buffer=nl.sbuf)
            v_tile = nl.ndarray((n_size, dv_size), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_tile, src=k[nl.ds(k_row_start, n_size), :])
            nisa.dma_copy(dst=v_tile, src=v[nl.ds(v_row_start, n_size), :])

            k_t = nl.ndarray((d_size, n_size), dtype=k.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=k_t, data=k_tile)

            for m_tile_idx in nl.affine_range(m_size // tile_m):
                m_start = m_tile_idx * tile_m
                q_row_start = (batch_idx * num_heads + head_idx) * m_size + m_start

                q_tile = nl.ndarray((tile_m, d_size), dtype=q.dtype, buffer=nl.sbuf)
                q_t = nl.ndarray((d_size, tile_m), dtype=q.dtype, buffer=nl.psum)
                nisa.dma_copy(dst=q_tile, src=q[nl.ds(q_row_start, tile_m), :])
                nisa.nc_transpose(dst=q_t, data=q_tile)

                scores_psum = nl.ndarray(
                    (tile_m, n_size), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(dst=scores_psum, stationary=q_t, moving=k_t)

                scores = nl.ndarray((tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores, src=scores_psum)
                nisa.tensor_scalar(
                    dst=scores,
                    data=scores,
                    op0=nl.multiply,
                    operand0=inv_sqrt_d,
                )

                row_max = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=row_max, op=nl.maximum, data=scores, axis=1, keepdims=True
                )

                centered = nl.ndarray(
                    (tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=centered,
                    data=scores,
                    op0=nl.subtract,
                    operand0=row_max,
                )

                exp_scores = nl.ndarray(
                    (tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.exponential(dst=exp_scores, src=centered)

                row_sum = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=row_sum, op=nl.add, data=exp_scores, axis=1, keepdims=True
                )

                inv_sum = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum, data=row_sum)

                probs = nl.ndarray((tile_m, n_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=probs,
                    data=exp_scores,
                    op0=nl.multiply,
                    operand0=inv_sum,
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
                nisa.dma_copy(dst=out[nl.ds(q_row_start, tile_m), :], src=out_tile)
    return out


def _numpy_tiled_attention(q, k, v):
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(float(q.shape[-1]))
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return probs @ v


def _run_with_xla(kernel, kernel_grid, *arrays):
    """Run one beta2 kernel invocation on an XLA device."""
    import torch
    import torch_xla

    device = torch_xla.device()
    tensors = [
        torch.as_tensor(array, device=device)
        if isinstance(array, np.ndarray)
        else array
        for array in arrays
    ]
    compiled_kernel = nki.jit(kernel, platform_target="trn2")
    result = compiled_kernel[kernel_grid](*tensors)
    torch_xla.sync()
    return result.cpu().numpy()


def _run_demo():
    """Run attention on 4D inputs and pass flattened views into the kernel."""
    kernel_grid = (1,)
    batch = 2
    num_heads = 2
    num_heads_k = 2
    num_heads_v = 2
    d_size = 128
    m_size = 256
    n_size = 128
    dv_size = 64

    q4d = np.linspace(
        -1.0,
        1.0,
        batch * num_heads * m_size * d_size,
        dtype=np.float32,
    ).reshape(batch, num_heads, m_size, d_size)
    k4d = np.linspace(
        1.0,
        -1.0,
        batch * num_heads_k * n_size * d_size,
        dtype=np.float32,
    ).reshape(batch, num_heads_k, n_size, d_size)
    v4d = np.linspace(
        -0.5,
        0.5,
        batch * num_heads_v * n_size * dv_size,
        dtype=np.float32,
    ).reshape(batch, num_heads_v, n_size, dv_size)

    q2d = q4d.reshape(-1, d_size)
    k2d = k4d.reshape(-1, d_size)
    v2d = v4d.reshape(-1, dv_size)
    out2d = np.empty((batch * num_heads * m_size, dv_size), dtype=np.float32)

    kernel_args = (
        q2d,
        k2d,
        v2d,
        out2d,
        batch,
        num_heads,
        num_heads_k,
        num_heads_v,
        m_size,
        n_size,
        d_size,
        dv_size,
    )
    expected2d = _numpy_tiled_attention(q4d, k4d, v4d).reshape(-1, dv_size)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki_beta2")(
            tiled_attention_kernel
        )
        traced_kernel[kernel_grid](*kernel_args)
        assert np.allclose(expected2d, out2d, atol=2e-4, rtol=2e-4)
        print("☑️ Actual equals expected!")
        triton_viz.launch(share=False)
    else:
        out2d = _run_with_xla(tiled_attention_kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected2d, out2d, atol=2e-4, rtol=2e-4)
        print("☑️ Actual equals expected!")


if __name__ == "__main__":
    _run_demo()
