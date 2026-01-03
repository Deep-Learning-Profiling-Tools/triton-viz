from neuronxcc import nki
import neuronxcc.nki.language as nl

import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
import numpy as np
import math


def softmax_kernel(in_tensor, out_tensor):
    """NKI softmax_kernel to compute softmax on the last dimension

    Args:
        in_tensor: an input tensor of shape [B,D], where B is a multiple of 128
        out_tensor: the resulting output tensor of shape [B,D]
    """
    B, D = in_tensor.shape

    # assert nl.tile_size.pmax == 128
    num_tiles = math.ceil(B / 128)
    for tile_idx in nl.affine_range(num_tiles):
        i_p = tile_idx * 128 + nl.arange(128)[:, None]
        i_f = nl.arange(D)[None, :]
        mask = (i_p < B) | (i_f < 0)
        tile = nl.load(in_tensor[i_p, i_f], mask=mask)
        tile_exp = nl.exp(tile)
        tile_expsum = nl.sum(tile_exp, -1, keepdims=True, mask=mask)
        tile_softmax = tile_exp / tile_expsum
        nl.store(out_tensor[i_p, i_f], value=tile_softmax, mask=mask)


TRITON_VIZ = True
kernel_grid = (1, 1, 1)
x_small = np.random.rand(16, 32).astype(np.float32)
y_small = np.empty(x_small.shape, dtype=x_small.dtype)
kernel_args = (x_small, y_small)

if TRITON_VIZ:
    print("Executing softmax_kernel with NKI interpreter...")
    traced_kernel = triton_viz.trace(client=Tracer(), backend="nki")(softmax_kernel)
    kernel_instance = traced_kernel[kernel_grid]
    kernel_instance(*kernel_args)

    print(f"Number of launches: {len(launches)}")
    if launches:
        launch = launches[-1]
        print(f"Number of records: {len(launch.records)}")
        for i, record in enumerate(launch.records):
            print(f"Record {i}: {type(record).__name__}")
            if hasattr(record, "ptr"):
                print(f"  ptr: {record.ptr}")
            if hasattr(record, "offsets"):
                print(f"  offsets shape: {record.offsets.shape}")
            if hasattr(record, "masks"):
                print(f"  masks shape: {record.masks.shape}")

    # Try to launch visualization
    try:
        triton_viz.launch(share=False)
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback

        traceback.print_exc()
else:
    print("Executing NKI JIT-ed softmax_kernel...")
    compiled_kernel = nki.jit(softmax_kernel, kernel_return=False)
    nki.simulate_kernel(compiled_kernel[kernel_grid], *kernel_args)

y_expected = np.exp(x_small) / np.exp(x_small).sum(-1, keepdims=True)
print(np.max(np.abs(y_expected - y_small)))
assert np.allclose(y_expected, y_small)
