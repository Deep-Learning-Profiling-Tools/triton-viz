import nki
import nki.isa as nisa
import nki.language as nl
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
    nisa.dma_copy(dst=result, src=c_tile)
    return result


def _run_with_xla(kernel, kernel_grid, *arrays):
    """Run one beta2 kernel invocation on an XLA (Trainium) device."""
    import torch
    import torch_xla

    device = torch_xla.device()
    tensors = [torch.as_tensor(array, device=device) for array in arrays]
    compiled_kernel = nki.jit(kernel, platform_target="trn2")
    result = compiled_kernel[kernel_grid](*tensors)
    torch_xla.sync()
    return result.cpu().numpy()


def _run_demo():
    kernel_grid = (1,)
    a = torch.ones((4, 3), dtype=torch.float32)
    b = torch.ones((4, 3), dtype=torch.float32)
    result = torch.empty((4, 3), dtype=torch.float32)
    kernel = nki_tensor_add_kernel
    kernel_args = (a, b, result)
    expected = a + b

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki_beta2")(kernel)
        traced_kernel[kernel_grid](*kernel_args)
        assert np.allclose(expected, result)
        print("☑️ Actual equals expected!")
        triton_viz.launch(share=False)
    else:  # Note: no official NKI Beta 2 CPU interpreter exists so run on XLA
        result = _run_with_xla(kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected, result)
        print("☑️ Actual equals expected!")


if __name__ == "__main__":
    _run_demo()
