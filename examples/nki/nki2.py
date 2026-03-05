import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np
import torch
import triton_viz

TRITON_VIZ_ENABLED = True


def nki_tensor_add_kernel(a_input, b_input, result):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    """

    # Check both input tensor shapes are the same for element-wise operation.
    assert a_input.shape == b_input.shape

    # Check the first dimension's size to ensure it does not exceed on-chip
    # memory tile size, since this simple kernel does not tile inputs.

    assert a_input.shape[0] <= nl.tile_size.pmax

    # Allocate space for the input tensors in SBUF and copy the inputs from HBM
    # to SBUF with DMA copy. Note: 'sbuf' is a keyword in NKI.
    a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)  # noqa: F821
    nisa.dma_copy(dst=b_tile, src=b_input)

    # Allocate space for the result and use tensor_tensor to perform
    # element-wise addition. Note: the first argument of 'tensor_tensor'
    # is the destination tensor.
    c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)  # noqa: F821
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    # Create a tensor in HBM and copy the result into HBM. Note: Similar to
    # 'sbuf', 'hbm' is a keyword in NKI.
    # c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)
    # nisa.dma_copy(dst=c_output, src=c_tile)
    nisa.dma_copy(dst=result, src=c_tile)


def _run_demo():
    kernel_grid = (1, 1, 1)
    a = torch.ones((4, 3))
    b = torch.ones((4, 3))
    result = torch.empty((4, 3))
    kernel_args = (a, b, result)

    if TRITON_VIZ_ENABLED:
        print("Executing matmul_kernel with NKI interpreter...")
        nl.tile_size.pmax = 128  # nl.tile_size.pmax = None if no Trn device found?
        traced_kernel = triton_viz.trace("tracer", backend="nki")(nki_tensor_add_kernel)
        kernel_instance = traced_kernel[kernel_grid]
        kernel_instance(*kernel_args)
        triton_viz.launch(share=False)
    else:  # NOTE: we must have trainium device to run this (no CPU interpreter for NKI Beta 2 yet)
        print("Executing NKI JIT-ed matmul_kernel...")
        from torch_xla.core import xla_model as xm

        device = xm.xla_device()
        a = a.to(device)
        b = b.to(device)
        # Invoke the kernel to add the results.
        compiled_kernel = nki.jit(nki_tensor_add_kernel)
        c = compiled_kernel(*kernel_args).cpu()
        assert np.allclose(a + b, c)

    z2 = result
    z1 = a + b
    print(np.max(np.abs(z1 - z2)))
    assert np.allclose(z1, z2)


if __name__ == "__main__":
    _run_demo()
