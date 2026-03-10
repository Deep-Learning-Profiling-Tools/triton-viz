import math
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

TRITON_VIZ_ENABLED = True
PRE_TRACE = True  # if True, run the NKI Beta 2 tracer before running interpreter. Can be set to false, though has less guarantees with matching NKI compiler behavior.


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
    """Compute tiled attention over 4D inputs."""
    assert num_heads % num_heads_k == 0 and num_heads % num_heads_v == 0
    q_rank = len(q.shape)
    k_rank = len(k.shape)
    v_rank = len(v.shape)
    out_rank = len(out.shape)
    assert q_rank == 4
    assert k_rank == q_rank
    assert v_rank == q_rank
    assert out_rank == q_rank

    assert q.shape == (batch, num_heads, m_size, d_size)
    assert k.shape == (batch, num_heads_k, n_size, d_size)
    assert v.shape == (batch, num_heads_v, n_size, dv_size)
    assert out.shape == (batch, num_heads, m_size, dv_size)

    tile_m = nl.tile_size.gemm_stationary_fmax
    tile_n = nl.tile_size.gemm_moving_fmax
    tile_p = nl.tile_size.pmax
    kv_tile_n = tile_p

    assert d_size == tile_p, f"Expected d_size ({d_size}) == {tile_p}"
    assert dv_size <= tile_n, f"Expected dv_size ({dv_size}) <= {tile_n}"
    assert (
        m_size % tile_m == 0
    ), f"Expected m_size ({m_size}) to be a multiple of {tile_m}"
    assert (
        n_size % kv_tile_n == 0
    ), f"Expected n_size ({n_size}) to be a multiple of {kv_tile_n}"

    inv_sqrt_d = 1.0 / math.sqrt(float(d_size))

    for batch_idx in nl.affine_range(batch):
        for head_idx in nl.affine_range(num_heads):
            head_k = (head_idx * num_heads_k) // num_heads
            head_v = (head_idx * num_heads_v) // num_heads

            for m_tile_idx in nl.affine_range(m_size // tile_m):
                m_start = m_tile_idx * tile_m

                q_tile = nl.ndarray((tile_m, d_size), dtype=q.dtype, buffer=nl.sbuf)
                q_t_psum = nl.ndarray((d_size, tile_m), dtype=q.dtype, buffer=nl.psum)
                nisa.dma_copy(
                    dst=q_tile,
                    src=q[batch_idx, head_idx, nl.ds(m_start, tile_m), :],
                )
                nisa.nc_transpose(dst=q_t_psum, data=q_tile)
                q_t = nl.ndarray((d_size, tile_m), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_t, src=q_t_psum)

                row_max = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                row_sum = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                acc = nl.ndarray((tile_m, dv_size), dtype=nl.float32, buffer=nl.sbuf)

                for n_tile_idx in nl.affine_range(n_size // kv_tile_n):
                    n_start = n_tile_idx * kv_tile_n

                    k_tile = nl.ndarray(
                        (kv_tile_n, d_size), dtype=k.dtype, buffer=nl.sbuf
                    )
                    v_tile = nl.ndarray(
                        (kv_tile_n, dv_size), dtype=v.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=k_tile,
                        src=k[batch_idx, head_k, nl.ds(n_start, kv_tile_n), :],
                    )
                    nisa.dma_copy(
                        dst=v_tile,
                        src=v[batch_idx, head_v, nl.ds(n_start, kv_tile_n), :],
                    )

                    k_t_psum = nl.ndarray(
                        (d_size, kv_tile_n), dtype=k.dtype, buffer=nl.psum
                    )
                    nisa.nc_transpose(dst=k_t_psum, data=k_tile)
                    k_t = nl.ndarray((d_size, kv_tile_n), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=k_t, src=k_t_psum)

                    scores_psum = nl.ndarray(
                        (tile_m, kv_tile_n), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=scores_psum, stationary=q_t, moving=k_t)

                    scores = nl.ndarray(
                        (tile_m, kv_tile_n), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=scores, src=scores_psum)
                    nisa.tensor_scalar(
                        dst=scores,
                        data=scores,
                        op0=nl.multiply,
                        operand0=inv_sqrt_d,
                    )

                    block_neg_row_max = nl.ndarray(
                        (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_reduce(
                        dst=block_neg_row_max,
                        op=nl.maximum,
                        data=scores,
                        axis=1,
                        negate=True,
                        keepdims=True,
                    )
                    block_row_max = nl.ndarray(
                        (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=block_row_max,
                        data=block_neg_row_max,
                        op0=nl.multiply,
                        operand0=-1.0,
                    )

                    block_row_sum = nl.ndarray(
                        (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    probs_t_psum = nl.ndarray(
                        (kv_tile_n, tile_m), dtype=scores.dtype, buffer=nl.psum
                    )
                    exp_scores = nl.ndarray(
                        (tile_m, kv_tile_n), dtype=nl.float32, buffer=nl.sbuf
                    )
                    if n_tile_idx == 0:
                        neg_block_row_max = nl.ndarray(
                            (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_scalar(
                            dst=neg_block_row_max,
                            data=block_row_max,
                            op0=nl.multiply,
                            operand0=-1.0,
                        )
                        nisa.activation(
                            dst=exp_scores,
                            op=nl.exp,
                            data=scores,
                            bias=neg_block_row_max,
                        )
                        nisa.tensor_reduce(
                            dst=block_row_sum,
                            op=nl.add,
                            data=exp_scores,
                            axis=1,
                            keepdims=True,
                        )
                        nisa.tensor_copy(dst=row_max, src=block_row_max)
                        nisa.tensor_copy(dst=row_sum, src=block_row_sum)
                    else:
                        new_row_max = nl.ndarray(
                            (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_tensor(
                            dst=new_row_max,
                            data1=row_max,
                            data2=block_row_max,
                            op=nl.maximum,
                        )

                        row_max_delta = nl.ndarray(
                            (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_tensor(
                            dst=row_max_delta,
                            data1=row_max,
                            data2=new_row_max,
                            op=nl.subtract,
                        )
                        prev_scale = nl.ndarray(
                            (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.activation(dst=prev_scale, op=nl.exp, data=row_max_delta)

                        neg_new_row_max = nl.ndarray(
                            (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_scalar(
                            dst=neg_new_row_max,
                            data=new_row_max,
                            op0=nl.multiply,
                            operand0=-1.0,
                        )
                        nisa.activation(
                            dst=exp_scores,
                            op=nl.exp,
                            data=scores,
                            bias=neg_new_row_max,
                        )

                        scaled_row_sum = nl.ndarray(
                            (tile_m, 1), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_scalar(
                            dst=scaled_row_sum,
                            data=row_sum,
                            op0=nl.multiply,
                            operand0=prev_scale,
                        )
                        nisa.tensor_reduce(
                            dst=block_row_sum,
                            op=nl.add,
                            data=exp_scores,
                            axis=1,
                            keepdims=True,
                        )
                        nisa.tensor_tensor(
                            dst=row_sum,
                            data1=scaled_row_sum,
                            data2=block_row_sum,
                            op=nl.add,
                        )
                        nisa.tensor_copy(dst=row_max, src=new_row_max)

                    nisa.nc_transpose(dst=probs_t_psum, data=exp_scores)
                    probs_t = nl.ndarray(
                        (kv_tile_n, tile_m), dtype=exp_scores.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=probs_t, src=probs_t_psum)

                    block_acc_psum = nl.ndarray(
                        (tile_m, dv_size), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(
                        dst=block_acc_psum, stationary=probs_t, moving=v_tile
                    )
                    block_acc = nl.ndarray(
                        (tile_m, dv_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=block_acc, src=block_acc_psum)

                    if n_tile_idx == 0:
                        nisa.tensor_copy(dst=acc, src=block_acc)
                    else:
                        scaled_acc = nl.ndarray(
                            (tile_m, dv_size), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_scalar(
                            dst=scaled_acc,
                            data=acc,
                            op0=nl.multiply,
                            operand0=prev_scale,
                        )
                        nisa.tensor_tensor(
                            dst=acc,
                            data1=scaled_acc,
                            data2=block_acc,
                            op=nl.add,
                        )

                inv_sum = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum, data=row_sum)

                out_float = nl.ndarray(
                    (tile_m, dv_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=out_float,
                    data=acc,
                    op0=nl.multiply,
                    operand0=inv_sum,
                )

                out_tile = nl.ndarray(
                    (tile_m, dv_size), dtype=out.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=out_tile, src=out_float)
                nisa.dma_copy(
                    dst=out[batch_idx, head_idx, nl.ds(m_start, tile_m), :],
                    src=out_tile,
                )
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
    compiled_kernel = nki.jit(kernel, platform_target="trn1")
    result = compiled_kernel[kernel_grid](*tensors)
    torch_xla.sync()
    return result.cpu().numpy()


def _run_demo():
    """Run attention on 4D inputs."""
    kernel_grid = (1,)
    batch = 2
    num_heads = 2
    num_heads_k = 2
    num_heads_v = 2
    d_size = 128
    m_size = 128
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

    out4d = np.empty((batch, num_heads, m_size, dv_size), dtype=np.float32)

    kernel_args = (
        q4d,
        k4d,
        v4d,
        out4d,
        batch,
        num_heads,
        num_heads_k,
        num_heads_v,
        m_size,
        n_size,
        d_size,
        dv_size,
    )
    expected4d = _numpy_tiled_attention(q4d, k4d, v4d)

    if TRITON_VIZ_ENABLED:
        import triton_viz

        traced_kernel = triton_viz.trace("tracer", backend="nki_beta2")(
            tiled_attention_kernel
        )
        traced_kernel[kernel_grid](*kernel_args, pre_trace=PRE_TRACE)
        assert np.allclose(expected4d, out4d)
        print("☑️ Actual equals expected!")
        triton_viz.launch(share=False)
    else:
        out4d = _run_with_xla(tiled_attention_kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected4d, out4d)
        print("☑️ Actual equals expected!")


if __name__ == "__main__":
    _run_demo()
