import os
from typing import Tuple

import neuronxcc.nki.language as nl
import torch
from neuronxcc import nki
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
import numpy as np

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# Ideally remove disable-dge for production performance, but kept for debugging context
os.environ["NEURON_CC_FLAGS"] = " --disable-dge "


def generate_pos_embedding(
    head_dim: int, position_ids: torch.Tensor, base: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate positional embeddings for rotary position encoding (Llama style).
    """
    # Core RoPE block
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    # Expand to [HeadDim/2, 1]
    inv_freq_expanded = inv_freq[:, None].float()
    # Expand to [1, SeqLen]
    position_ids_expanded = position_ids[None, :].float()

    # MatMul -> [HeadDim/2, SeqLen] -> Transpose -> [SeqLen, HeadDim/2]
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)

    # Concatenate to match HeadDim -> [Batch, SeqLen, HeadDim]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _nki_apply_rotary_embedding_core(q_tile, k_tile, cos_tile, sin_tile, output_tile):
    """
    Core NKI implementation of rotary position embedding computation.

    Parameters
    ----------
    q_tile : nl.Tensor
        Query tensor tile
    k_tile : nl.Tensor
        Key tensor tile
    cos_tile : nl.Tensor
        Cosine embedding tile
    sin_tile : nl.Tensor
        Sine embedding tile
    output_tile : nl.Tensor
        Output buffer for results

    Notes
    -----
    The function applies rotary position embedding to query and key tensors
    using the provided cosine and sine embeddings.
    """

    assert q_tile.shape[-1] % 2 == 0, "Sequence length for q_tile must be even!"
    assert k_tile.shape[-1] % 2 == 0, "Sequence length for k_tile must be even!"
    assert (
        q_tile.shape[-1] == k_tile.shape[-1]
    ), "q_tile and k_tile must have the same sequence length"

    seq_len = q_tile.shape[-1]

    # Rotate Q
    output_tile[0, :, :] = q_tile * cos_tile
    output_tile[0, :, : seq_len // 2] = output_tile[0, :, : seq_len // 2] + (
        -1 * q_tile[:, seq_len // 2 :] * sin_tile[:, : seq_len // 2]
    )
    output_tile[0, :, seq_len // 2 :] = output_tile[0, :, seq_len // 2 :] + (
        q_tile[:, : seq_len // 2] * sin_tile[:, seq_len // 2 :]
    )

    # Rotate K
    output_tile[1, :, :] = k_tile * cos_tile
    output_tile[1, :, : seq_len // 2] = output_tile[1, :, : seq_len // 2] + (
        -1 * k_tile[:, seq_len // 2 :] * sin_tile[:, : seq_len // 2]
    )
    output_tile[1, :, seq_len // 2 :] = output_tile[1, :, seq_len // 2 :] + (
        k_tile[:, : seq_len // 2] * sin_tile[:, seq_len // 2 :]
    )


def nki_rope_kernel(q, k, cos, sin, output_q, output_k):
    """
    NKI implementation of rotary position embedding.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
    k : torch.Tensor
        Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
    cos : torch.Tensor
        Cosine embeddings
    sin : torch.Tensor
        Sine embeddings

    Returns
    -------
    nl.Tensor
        Output tensor containing transformed query and key tensors

    Raises
    ------
    AssertionError
        If input tensor shapes don't match or head dimension > 128
    """
    assert (
        q.shape == k.shape
    ), f"Shape of Q Tensor: {q.shape} doesn't match shape of K Tensor: {k.shape}"
    assert (
        cos.shape == sin.shape
    ), f"Shape of cos Tensor: {cos.shape} doesn't match shape of sin Tensor: {sin.shape}"
    PMAX = 128
    assert (
        q.shape[-1] <= PMAX
    ), f"Shape of head dim (last dim) is more than PMAX: {q.shape}"

    head_id = nl.program_id(axis=0)
    seq_len = q.shape[1]
    num_seq_batches = div_ceil(seq_len, nl.tile_size.pmax)
    # output = nl.ndarray([2] + list(q.shape), dtype=q.dtype, buffer=nl.shared_hbm)
    i_p, i_f = nl.mgrid[0:PMAX, 0 : q.shape[-1]]
    # for seq_batch_id in nl.affine_range(0, num_seq_batches): # TODO
    for seq_batch_id in nl.affine_range(num_seq_batches):
        # q_hbm_tile = q[batch_id, head_id]
        # k_hbm_tile = k[batch_id, head_id]
        # cos_hbm_tile = cos[batch_id]
        # sin_hbm_tile = sin[batch_id]

        q_tile = nl.load(
            # q_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            q[head_id, seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        k_tile = nl.load(
            # k_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            k[head_id, seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        output_tile = nl.ndarray(
            # [2] + [nl.par_dim(k_tile.shape[0]), k_tile.shape[1]], # TODO
            [2] + [k_tile.shape[0], k_tile.shape[1]],
            dtype=k_tile.dtype,
            buffer=nl.sbuf,
        )
        cos_tile = nl.load(
            # cos_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            cos[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        sin_tile = nl.load(
            # sin_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            sin[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )

        _nki_apply_rotary_embedding_core(
            q_tile, k_tile, cos_tile, sin_tile, output_tile
        )

        # output_q_hbm_tile = output[0, batch_id, head_id]
        # output_k_hbm_tile = output[1, batch_id, head_id]

        nl.store(
            # output_q_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            output_q[head_id, seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            output_tile[0, :, :],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )
        nl.store(
            # output_k_hbm_tile[seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            output_k[head_id, seq_batch_id * nl.tile_size.pmax + i_p, i_f],
            output_tile[1, :, :],
            mask=(seq_batch_id * nl.tile_size.pmax + i_p < seq_len),
        )

    # return output


# Torch reference implementation
def torch_rope_kernel(q, k, cos, sin):
    """Simple torch reference for rotary embedding"""

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    #
    # FIX: Unsqueeze cos/sin to shape [B, 1, S, D] to broadcast over Heads [B, H, S, D]
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def _run_demo():
    triton_viz_enabled = True
    h_dim, s_dim, d_dim = 2, 4, 8
    kernel_grid = (h_dim,)

    if triton_viz_enabled:
        q = torch.randn(h_dim, s_dim, d_dim)
        k = torch.randn(h_dim, s_dim, d_dim)
        position_ids = torch.arange(s_dim)
        cos, sin = generate_pos_embedding(d_dim, position_ids)

        output_q = torch.empty((h_dim, s_dim, d_dim))
        output_k = torch.empty((h_dim, s_dim, d_dim))
        kernel_args = (
            q.numpy(),
            k.numpy(),
            cos.numpy(),
            sin.numpy(),
            output_q.numpy(),
            output_k.numpy(),
        )

        print("Executing rotary embedding kernel with NKI interpreter...")
        traced_kernel = triton_viz.trace(client=Tracer(), backend="nki")(
            nki_rope_kernel
        )
        kernel_instance = traced_kernel[kernel_grid]
        kernel_instance(*kernel_args)

        print(f"Number of launches: {len(launches)}")
        if launches:
            launch = launches[-1]
            print(f"Number of records: {len(launch.records)}")

        try:
            triton_viz.launch(share=False)
        except Exception as e:
            print(f"Visualization error: {e}")
    else:
        q = torch.randn(h_dim, s_dim, d_dim, dtype=torch.float32)
        k = torch.randn(h_dim, s_dim, d_dim, dtype=torch.float32)
        position_ids = torch.arange(s_dim)
        cos, sin = generate_pos_embedding(d_dim, position_ids)

        print("Executing NKI JIT-ed rotary embedding kernel...")
        compiled_kernel = nki.jit(nki_rope_kernel, kernel_return=False)
        output_q_np = np.zeros((h_dim, s_dim, d_dim), dtype=np.float32)
        output_k_np = np.zeros((h_dim, s_dim, d_dim), dtype=np.float32)

        nki.simulate_kernel(
            compiled_kernel[kernel_grid],
            q.numpy(),
            k.numpy(),
            cos.numpy(),
            sin.numpy(),
            output_q_np,
            output_k_np,
        )

        expected_q, expected_k = torch_rope_kernel(q, k, cos, sin)
        actual_q = torch.from_numpy(output_q_np)
        actual_k = torch.from_numpy(output_k_np)

        max_diff_q = torch.max(torch.abs(expected_q - actual_q))
        max_diff_k = torch.max(torch.abs(expected_k - actual_k))

        print(f"Q max diff: {max_diff_q}")
        print(f"K max diff: {max_diff_k}")

        assert torch.allclose(expected_q, actual_q, rtol=1e-3, atol=1e-3), "Q mismatch"
        assert torch.allclose(expected_k, actual_k, rtol=1e-3, atol=1e-3), "K mismatch"
        print("Results match!")


if __name__ == "__main__":
    _run_demo()
