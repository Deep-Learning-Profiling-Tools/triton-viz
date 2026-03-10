import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True
PRE_TRACE = True  # if True, run the NKI Beta 2 tracer before running interpreter. Can be set to false, though has less guarantees with matching NKI compiler behavior.


def _rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _rope_reference(q, k, cos, sin):
    cos_b = cos[np.newaxis, np.newaxis, :, :]
    sin_b = sin[np.newaxis, np.newaxis, :, :]
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


def rope_kernel_2d(
    q2, k2, cos, sin, out_q2, out_k2, batch, num_heads, seq_len, head_dim
):
    """Apply RoPE using flattened 2D q/k tensors."""
    assert q2.shape == (batch * num_heads * seq_len, head_dim)
    assert k2.shape == q2.shape
    assert out_q2.shape == q2.shape
    assert out_k2.shape == k2.shape
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    assert head_dim % 2 == 0

    tile_s = nl.tile_size.pmax
    assert seq_len % tile_s == 0, f"Expected seq_len ({seq_len}) multiple of {tile_s}"
    half = head_dim // 2

    for batch_idx in nl.affine_range(batch):
        for head_idx in nl.affine_range(num_heads):
            row_base = (batch_idx * num_heads + head_idx) * seq_len
            for seq_tile in nl.affine_range(seq_len // tile_s):
                seq_start = seq_tile * tile_s
                row_start = row_base + seq_start

                q_tile = nl.ndarray((tile_s, head_dim), dtype=q2.dtype, buffer=nl.sbuf)
                k_tile = nl.ndarray((tile_s, head_dim), dtype=k2.dtype, buffer=nl.sbuf)
                cos_tile = nl.ndarray(
                    (tile_s, head_dim), dtype=cos.dtype, buffer=nl.sbuf
                )
                sin_tile = nl.ndarray(
                    (tile_s, head_dim), dtype=sin.dtype, buffer=nl.sbuf
                )

                nisa.dma_copy(dst=q_tile, src=q2[nl.ds(row_start, tile_s), :])
                nisa.dma_copy(dst=k_tile, src=k2[nl.ds(row_start, tile_s), :])
                nisa.dma_copy(dst=cos_tile, src=cos[nl.ds(seq_start, tile_s), :])
                nisa.dma_copy(dst=sin_tile, src=sin[nl.ds(seq_start, tile_s), :])

                q_out = nl.ndarray(
                    (tile_s, head_dim), dtype=out_q2.dtype, buffer=nl.sbuf
                )
                k_out = nl.ndarray(
                    (tile_s, head_dim), dtype=out_k2.dtype, buffer=nl.sbuf
                )
                tmp1 = nl.ndarray((tile_s, half), dtype=out_q2.dtype, buffer=nl.sbuf)
                tmp2 = nl.ndarray((tile_s, half), dtype=out_q2.dtype, buffer=nl.sbuf)

                nisa.tensor_tensor(
                    dst=tmp1,
                    data1=q_tile[:, :half],
                    data2=cos_tile[:, :half],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=tmp2,
                    data1=q_tile[:, half:],
                    data2=sin_tile[:, :half],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=q_out[:, :half], data1=tmp1, data2=tmp2, op=nl.subtract
                )

                nisa.tensor_tensor(
                    dst=tmp1,
                    data1=q_tile[:, half:],
                    data2=cos_tile[:, half:],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=tmp2,
                    data1=q_tile[:, :half],
                    data2=sin_tile[:, half:],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=q_out[:, half:], data1=tmp1, data2=tmp2, op=nl.add
                )

                nisa.tensor_tensor(
                    dst=tmp1,
                    data1=k_tile[:, :half],
                    data2=cos_tile[:, :half],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=tmp2,
                    data1=k_tile[:, half:],
                    data2=sin_tile[:, :half],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=k_out[:, :half], data1=tmp1, data2=tmp2, op=nl.subtract
                )

                nisa.tensor_tensor(
                    dst=tmp1,
                    data1=k_tile[:, half:],
                    data2=cos_tile[:, half:],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=tmp2,
                    data1=k_tile[:, :half],
                    data2=sin_tile[:, half:],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=k_out[:, half:], data1=tmp1, data2=tmp2, op=nl.add
                )

                nisa.dma_copy(dst=out_q2[nl.ds(row_start, tile_s), :], src=q_out)
                nisa.dma_copy(dst=out_k2[nl.ds(row_start, tile_s), :], src=k_out)

    return out_q2, out_k2


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
    out_q, out_k = compiled_kernel[kernel_grid](*tensors)
    torch_xla.sync()
    return out_q.cpu().numpy(), out_k.cpu().numpy()


def _run_demo():
    """Run RoPE on 4D inputs and pass flattened views into the kernel."""
    kernel_grid = (1,)
    batch = 1
    num_heads = 4
    seq_len = 128
    head_dim = 64

    q4d = np.linspace(
        -1.0,
        1.0,
        batch * num_heads * seq_len * head_dim,
        dtype=np.float32,
    ).reshape(batch, num_heads, seq_len, head_dim)
    k4d = np.linspace(
        1.0,
        -1.0,
        batch * num_heads * seq_len * head_dim,
        dtype=np.float32,
    ).reshape(batch, num_heads, seq_len, head_dim)

    cos, sin = _rope_tables(seq_len, head_dim)
    expected_q, expected_k = _rope_reference(q4d, k4d, cos, sin)

    q2d = q4d.reshape(-1, head_dim)
    k2d = k4d.reshape(-1, head_dim)
    out_q2d = np.empty_like(q2d)
    out_k2d = np.empty_like(k2d)

    kernel_args = (
        q2d,
        k2d,
        cos,
        sin,
        out_q2d,
        out_k2d,
        batch,
        num_heads,
        seq_len,
        head_dim,
    )

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki_beta2")(rope_kernel_2d)
        traced_kernel[kernel_grid](*kernel_args, pre_trace=PRE_TRACE)
        assert np.allclose(expected_q.reshape(-1, head_dim), out_q2d)
        assert np.allclose(expected_k.reshape(-1, head_dim), out_k2d)
        print("☑️ Actual equals expected!")
        triton_viz.launch(share=False)
    else:
        out_q2d, out_k2d = _run_with_xla(rope_kernel_2d, kernel_grid, *kernel_args)
        assert np.allclose(expected_q.reshape(-1, head_dim), out_q2d)
        assert np.allclose(expected_k.reshape(-1, head_dim), out_k2d)
        print("☑️ Actual equals expected!")


if __name__ == "__main__":
    _run_demo()
