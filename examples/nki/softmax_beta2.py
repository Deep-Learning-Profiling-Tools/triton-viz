import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def softmax_kernel(x, out):
    """Compute row-wise softmax on a 2D tensor.

    Args:
        x: Input tensor with shape ``[batch, dim]``.
        out: Output tensor with shape ``[batch, dim]`` written in place.
    """
    batch, dim = x.shape
    tile_p = nl.tile_size.pmax
    assert batch % tile_p == 0, f"Expected batch ({batch}) to be a multiple of {tile_p}"

    for tile_idx in nl.affine_range(batch // tile_p):
        row_start = tile_idx * tile_p
        x_tile = nl.ndarray((tile_p, dim), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[nl.ds(row_start, tile_p), :])

        row_max = nl.ndarray((tile_p, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=row_max, op=nl.maximum, data=x_tile, axis=1, keepdims=True
        )

        centered = nl.ndarray((tile_p, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=centered,
            data1=x_tile,
            data2=row_max.broadcast_to((tile_p, dim)),
            op=nl.subtract,
        )

        exp_tile = nl.ndarray((tile_p, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=np.exp, data=centered)

        row_sum = nl.ndarray((tile_p, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, op=nl.add, data=exp_tile, axis=1, keepdims=True)

        inv_sum = nl.ndarray((tile_p, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sum, data=row_sum)

        out_tile = nl.ndarray((tile_p, dim), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=out_tile,
            data1=exp_tile,
            data2=inv_sum.broadcast_to((tile_p, dim)),
            op=nl.multiply,
        )
        nisa.dma_copy(dst=out[nl.ds(row_start, tile_p), :], src=out_tile)


def _numpy_softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


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
    """Run the softmax example with input/output shape ``[256, 64]``."""
    kernel_grid = (1, 1, 1)
    batch = 256
    dim = 64
    x = np.linspace(-3.0, 3.0, batch * dim, dtype=np.float32).reshape(batch, dim)
    out = np.empty_like(x)
    kernel_args = (x, out)
    expected = _numpy_softmax(x)

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki")(softmax_kernel)
        traced_kernel[kernel_grid](*kernel_args)
        assert np.allclose(expected, out, atol=1e-5, rtol=1e-5)
        print("actual equals expected")
        triton_viz.launch(share=False)
    else:
        _, out = _run_with_xla(softmax_kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected, out, atol=1e-5, rtol=1e-5)
        print("actual equals expected")


if __name__ == "__main__":
    _run_demo()
