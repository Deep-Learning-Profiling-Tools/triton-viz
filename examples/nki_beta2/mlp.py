import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np

TRITON_VIZ_ENABLED = True
PRE_TRACE = True  # if True, run the NKI Beta 2 tracer before running interpreter. Can be set to false, though has less guarantees with matching NKI compiler behavior.
TILE_M = 128
TILE_K = 128
TILE_H = 128
TILE_N = 512


def mlp_kernel(x_t, w1, w2, out):
    """Compute tiled ``relu(x_t.T @ w1).T @ w2`` into ``out``."""
    k_dim, batch = x_t.shape
    k_dim_w1, hidden = w1.shape
    hidden_w2, out_dim = w2.shape
    assert k_dim == k_dim_w1, "x_t and w1 must share the input dimension"
    assert hidden == hidden_w2, "w1 and w2 must share the hidden dimension"

    tile_m = TILE_M
    tile_k = TILE_K
    tile_h = TILE_H
    tile_n = TILE_N
    assert batch % tile_m == 0, f"Expected batch ({batch}) to be a multiple of {tile_m}"
    assert (
        k_dim % tile_k == 0
    ), f"Expected input dim ({k_dim}) to be a multiple of {tile_k}"
    assert (
        hidden % tile_h == 0
    ), f"Expected hidden dim ({hidden}) to be a multiple of {tile_h}"
    assert (
        out_dim % tile_n == 0
    ), f"Expected output dim ({out_dim}) to be a multiple of {tile_n}"

    for batch_idx in nl.affine_range(batch // tile_m):
        batch_start = batch_idx * tile_m
        for out_idx in nl.affine_range(out_dim // tile_n):
            out_start = out_idx * tile_n
            out_psum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)

            for hidden_idx in nl.affine_range(hidden // tile_h):
                hidden_start = hidden_idx * tile_h
                hidden_psum = nl.ndarray(
                    (tile_m, tile_h), dtype=nl.float32, buffer=nl.psum
                )

                for k_idx in nl.affine_range(k_dim // tile_k):
                    k_start = k_idx * tile_k
                    x_tile = nl.ndarray(
                        (tile_k, tile_m), dtype=x_t.dtype, buffer=nl.sbuf
                    )
                    w1_tile = nl.ndarray(
                        (tile_k, tile_h), dtype=w1.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=x_tile,
                        src=x_t[
                            nl.ds(k_start, tile_k),
                            nl.ds(batch_start, tile_m),
                        ],
                    )
                    nisa.dma_copy(
                        dst=w1_tile,
                        src=w1[
                            nl.ds(k_start, tile_k),
                            nl.ds(hidden_start, tile_h),
                        ],
                    )
                    nisa.nc_matmul(dst=hidden_psum, stationary=x_tile, moving=w1_tile)

                hidden_tile = nl.ndarray(
                    (tile_m, tile_h), dtype=out.dtype, buffer=nl.sbuf
                )
                hidden_t_psum = nl.ndarray(
                    (tile_h, tile_m), dtype=out.dtype, buffer=nl.psum
                )
                hidden_t = nl.ndarray((tile_h, tile_m), dtype=out.dtype, buffer=nl.sbuf)
                w2_tile = nl.ndarray((tile_h, tile_n), dtype=w2.dtype, buffer=nl.sbuf)

                nisa.tensor_copy(dst=hidden_tile, src=hidden_psum)
                nisa.tensor_scalar(
                    dst=hidden_tile,
                    data=hidden_tile,
                    op0=nl.maximum,
                    operand0=0.0,
                )
                nisa.nc_transpose(dst=hidden_t_psum, data=hidden_tile)
                nisa.tensor_copy(dst=hidden_t, src=hidden_t_psum)
                nisa.dma_copy(
                    dst=w2_tile,
                    src=w2[
                        nl.ds(hidden_start, tile_h),
                        nl.ds(out_start, tile_n),
                    ],
                )
                nisa.nc_matmul(dst=out_psum, stationary=hidden_t, moving=w2_tile)

            out_tile = nl.ndarray((tile_m, tile_n), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=out_psum)
            nisa.dma_copy(
                dst=out[
                    nl.ds(batch_start, tile_m),
                    nl.ds(out_start, tile_n),
                ],
                src=out_tile,
            )
    return out


def _round_up(size, tile):
    """Round ``size`` up to the nearest ``tile`` multiple."""
    return ((size + tile - 1) // tile) * tile


def _pad_matrix(x, shape):
    """Zero-pad a 2D matrix into ``shape``."""
    padded = np.zeros(shape, dtype=x.dtype)
    padded[: x.shape[0], : x.shape[1]] = x
    return padded


def _run_with_xla(kernel, kernel_grid, *arrays):
    """Run one beta2 kernel invocation on an XLA device."""
    import torch
    import torch_xla

    device = torch_xla.device()
    tensors = [torch.as_tensor(array, device=device) for array in arrays]
    compiled_kernel = nki.jit(kernel, platform_target="trn1")
    result = compiled_kernel[kernel_grid](*tensors)
    torch_xla.sync()
    return result.cpu().numpy()


def _run_demo():
    """Run the tiled MLP example on non-tile-aligned 2D matrices."""
    kernel_grid = (1,)
    batch = 190
    in_dim = 170
    hidden = 222
    out_dim = 333
    tile_m = TILE_M
    tile_k = TILE_K
    tile_h = TILE_H
    tile_n = TILE_N
    batch_pad = _round_up(batch, tile_m)
    in_dim_pad = _round_up(in_dim, tile_k)
    hidden_pad = _round_up(hidden, tile_h)
    out_dim_pad = _round_up(out_dim, tile_n)

    x = np.linspace(-1.0, 1.0, batch * in_dim, dtype=np.float32).reshape(batch, in_dim)
    w1 = np.linspace(-0.5, 0.5, in_dim * hidden, dtype=np.float32).reshape(
        in_dim, hidden
    )
    w2 = np.linspace(-0.25, 0.75, hidden * out_dim, dtype=np.float32).reshape(
        hidden, out_dim
    )
    # add 100s to make the distribution a bit weird to prevent math effects (e.g. dot(randn, randn) ~ 0)
    x[0, 0] = 100
    w1[0, 0] = 100
    w2[0, 0] = 100

    x_t = _pad_matrix(x.T, (in_dim_pad, batch_pad))
    w1_padded = _pad_matrix(w1, (in_dim_pad, hidden_pad))
    w2_padded = _pad_matrix(w2, (hidden_pad, out_dim_pad))
    out = np.empty((batch_pad, out_dim_pad), dtype=np.float32)
    kernel_args = (x_t, w1_padded, w2_padded, out)
    expected = np.maximum(x @ w1, 0.0) @ w2

    if TRITON_VIZ_ENABLED:
        import triton_viz

        traced_kernel = triton_viz.trace("tracer", backend="nki_beta2")(mlp_kernel)
        traced_kernel[kernel_grid](*kernel_args, pre_trace=PRE_TRACE)
        assert np.allclose(expected, out[:batch, :out_dim], atol=1e-4, rtol=1e-4)
        print("☑️ Actual equals expected!")
        triton_viz.launch(share=False)
    else:
        out = _run_with_xla(mlp_kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected, out[:batch, :out_dim], atol=1e-4, rtol=1e-4)
        print("☑️ Actual equals expected!")


if __name__ == "__main__":
    _run_demo()
