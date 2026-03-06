import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def rmsnorm_kernel(x, gamma, out, eps=1e-6):
    """Compute RMSNorm on rows of ``x``.

    Args:
        x: Input tensor with shape ``[batch, dim]``.
        gamma: Scale tensor with shape ``[batch, dim]``.
        out: Output tensor with shape ``[batch, dim]`` written in place.
        eps: Stability epsilon added before rsqrt.
    """
    batch, dim = x.shape
    tile_p = nl.tile_size.pmax
    assert batch % tile_p == 0, f"Expected batch ({batch}) to be a multiple of {tile_p}"
    assert gamma.shape == (
        batch,
        dim,
    ), f"Expected gamma shape ({batch}, {dim}), got {gamma.shape}"

    for tile_idx in nl.affine_range(batch // tile_p):
        row_start = tile_idx * tile_p
        x_tile = nl.ndarray((tile_p, dim), dtype=x.dtype, buffer=nl.sbuf)
        gamma_tile = nl.ndarray((tile_p, dim), dtype=gamma.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[nl.ds(row_start, tile_p), :])
        nisa.dma_copy(dst=gamma_tile, src=gamma[nl.ds(row_start, tile_p), :])

        sq_tile = nl.ndarray((tile_p, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq_tile, data1=x_tile, data2=x_tile, op=nl.multiply)

        sq_mean = nl.ndarray((tile_p, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sq_mean, op=nl.add, data=sq_tile, axis=1, keepdims=True)
        nisa.tensor_scalar(
            dst=sq_mean, data=sq_mean, op0=nl.multiply, operand0=1.0 / float(dim)
        )

        nisa.tensor_scalar(dst=sq_mean, data=sq_mean, op0=nl.add, operand0=eps)
        inv_rms = nl.ndarray((tile_p, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_rms, op=nl.rsqrt, data=sq_mean)

        norm_tile = nl.ndarray((tile_p, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=norm_tile,
            data=x_tile,
            op0=nl.multiply,
            operand0=inv_rms,
        )

        out_tile = nl.ndarray((tile_p, dim), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=out_tile,
            data=norm_tile,
            op0=nl.multiply,
            operand0=gamma_tile,
        )
        nisa.dma_copy(dst=out[nl.ds(row_start, tile_p), :], src=out_tile)
    return out


def _numpy_rmsnorm(x, gamma, eps=1e-6):
    mean_sq = np.mean(x * x, axis=1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)) * gamma


def _run_with_xla(kernel, kernel_grid, *arrays):
    """Run one beta2 kernel invocation on an XLA device."""
    import torch
    import torch_xla

    device = torch_xla.device()
    tensors = [torch.as_tensor(array, device=device) for array in arrays]
    compiled_kernel = nki.jit(kernel, platform_target="trn2")
    result = compiled_kernel[kernel_grid](*tensors)
    torch_xla.sync()
    return result.cpu().numpy()


def _run_demo():
    """Run the RMSNorm example with x ``[256, 128]`` and gamma ``[128]``."""
    kernel_grid = (1,)
    batch = 256
    dim = 128
    x = np.linspace(-2.0, 2.0, batch * dim, dtype=np.float32).reshape(batch, dim)
    gamma_base = np.linspace(0.5, 1.5, dim, dtype=np.float32)
    gamma = np.broadcast_to(gamma_base.reshape(1, dim), (batch, dim)).copy()
    out = np.empty_like(x)
    kernel_args = (x, gamma, out)
    expected = _numpy_rmsnorm(x, gamma)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki_beta2")(rmsnorm_kernel)
        traced_kernel[kernel_grid](*kernel_args)
        assert np.allclose(expected, out)
        print("☑️ Actual equals expected!")
        triton_viz.launch(share=False)
    else:
        out = _run_with_xla(rmsnorm_kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected, out)
        print("☑️ Actual equals expected!")


if __name__ == "__main__":
    _run_demo()
