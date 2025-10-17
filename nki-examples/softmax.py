from neuronxcc import nki
import neuronxcc.nki.language as nl

import torch
import triton_viz
import torch_xla.core.xla_model as xm
from triton_viz.clients import Tracer
from triton_viz.core import config as cfg
from triton_viz.core.trace import launches
import numpy as np
import math


@nki.jit
def softmax(in_tensor):
    # assume B,D; softmax on D for now
    B, D = in_tensor.shape
    out_tensor = nl.ndarray((B, D), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

    #assert nl.tile_size.pmax == 128
    num_tiles = math.ceil(B / 128)
    for tile_idx in nl.affine_range(num_tiles):
        i_p = tile_idx * 128 + nl.arange(128)[:, None]
        i_f = nl.arange(D)[None, :]
        mask = i_p < B
        tile = nl.load(in_tensor[i_p, i_f], mask=mask)
        tile_exp = nl.exp(tile)
        tile_expsum = nl.sum(tile_exp, -1, keepdims=True, mask=mask)
        tile_softmax = tile_exp / tile_expsum
        nl.store(out_tensor[i_p, i_f], value=tile_softmax, mask=mask)
    return out_tensor


if __name__ == "__main__":
    cfg.reset()
    device = "cpu"
    B, D = 16, 32
    torch.manual_seed(0)
    x = torch.rand((B, D), dtype=torch.float32, device=device)
    y = np.exp(x) / np.exp(x).sum(-1, keepdim=True)
    TRITON_VIZ = False

    if TRITON_VIZ:
        softmax = triton_viz.trace()(softmax)
        softmax[(1,1,1)](x)

        # Print records to see what's being captured
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
            triton_viz.launch()
        except Exception as e:
            print(f"\nError during visualization: {e}")
            import traceback

            traceback.print_exc()
    else:
        y_pred = nki.simulate_kernel(softmax, x.numpy())
        assert np.allclose(y, y_pred)
