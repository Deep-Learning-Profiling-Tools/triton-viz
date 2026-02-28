import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def rmsnorm_kernel(x, gamma, out, eps=1e-6):
    """Compute RMSNorm on rows of x using beta2 primitives."""
    batch, dim = x.shape
    tile_p = nl.tile_size.pmax
    assert batch % tile_p == 0, f"Expected batch ({batch}) to be a multiple of {tile_p}"
    assert gamma.shape == (dim,), f"Expected gamma shape ({dim},), got {gamma.shape}"

    gamma_tile = nl.ndarray((1, dim), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_tile, src=gamma.reshape((1, dim)))

    for tile_idx in nl.affine_range(batch // tile_p):
        row_start = tile_idx * tile_p
        x_tile = nl.ndarray((tile_p, dim), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[nl.ds(row_start, tile_p), :])

        sq_tile = nl.ndarray((tile_p, dim), dtype=np.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq_tile, op=np.square, data=x_tile)

        sq_mean = nl.ndarray((tile_p, 1), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sq_mean, op=nl.add, data=sq_tile, axis=1, keepdims=True)
        nisa.tensor_scalar(
            dst=sq_mean, data=sq_mean, op0=nl.multiply, operand0=1.0 / float(dim)
        )

        inv_rms = nl.ndarray((tile_p, 1), dtype=np.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_rms, op=nl.rsqrt, data=sq_mean, bias=eps)

        norm_tile = nl.ndarray((tile_p, dim), dtype=np.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=norm_tile,
            data1=x_tile,
            data2=inv_rms.broadcast_to((tile_p, dim)),
            op=nl.multiply,
        )

        out_tile = nl.ndarray((tile_p, dim), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=out_tile,
            data1=norm_tile,
            data2=gamma_tile.broadcast_to((tile_p, dim)),
            op=nl.multiply,
        )
        nisa.dma_copy(dst=out[nl.ds(row_start, tile_p), :], src=out_tile)


def _numpy_rmsnorm(x, gamma, eps=1e-6):
    mean_sq = np.mean(x * x, axis=1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)) * gamma.reshape((1, -1))


def _run_demo():
    kernel_grid = (1, 1, 1)
    batch = 256
    dim = 128
    x = np.linspace(-2.0, 2.0, batch * dim, dtype=np.float32).reshape(batch, dim)
    gamma = np.linspace(0.5, 1.5, dim, dtype=np.float32)
    out = np.empty_like(x)
    kernel_args = (x, gamma, out)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki")(rmsnorm_kernel)
        traced_kernel[kernel_grid](*kernel_args)
        triton_viz.launch(share=False)
    else:
        compiled_kernel = nki.jit(rmsnorm_kernel, kernel_return=False)
        nki.simulate_kernel(compiled_kernel[kernel_grid], *kernel_args)

    expected = _numpy_rmsnorm(x, gamma)
    assert np.allclose(expected, out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    _run_demo()
