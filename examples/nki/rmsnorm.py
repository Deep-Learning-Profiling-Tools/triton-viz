# CODEX NOTE: IGNORE THIS FILE, IT IS THE DEPRECATED NKI INTERPRETER
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
import math
import numpy as np
import torch
import triton_viz


def nki_rmsnorm_kernel(a_tensor, g_tensor, result):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    B, D = a_tensor.shape

    # Make sure shapes match
    assert D == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    B_TILE = 8
    ix = nl.arange(B_TILE)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(D)[None, :]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, D))[iw, iy], mask=iy < D)

    # Process 2 rows at a time due to 2-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently
    for i in nl.affine_range(math.ceil(B / B_TILE)):
        # Load input data from external memory to on-chip memory
        mask = (i * B_TILE + ix < B) & (iy < D)
        a_tile = nl.load(a_tensor[i * B_TILE + ix, iy], mask=mask)

        # Compute element-wise square of a_tensor
        in_square = nl.square(a_tile)

        # Calculate sum of squared elements, along last dimension
        # square_sum = nl.sum(in_square, axis=[1], mask=mask) #[:, None]
        square_sum = nl.sum(in_square, axis=1, mask=mask)[:, None]

        # Scale and get a reciprocal
        mean = square_sum / D

        # Take square root of mean and then reciprocal with
        # rsqrt API (one ISA instruction)
        rms_reciprocal = nl.rsqrt(mean)

        # Scale the input tensor
        out_tile = nl.multiply(a_tile, rms_reciprocal)

        # Broadcast weight along first axis to match tensor shape
        # B_active = min(B - i * 2, 2)
        g_bcast = g_tile.broadcast_to((B_TILE, D))

        # Multiply with the RMSNorm weight
        out_tile = nl.multiply(out_tile, g_bcast, mask=(i * B_TILE + ix < B))

        # store the addition results back to external memory (out_tensor)
        nl.store(result[i * B_TILE + ix, iy], value=out_tile, mask=mask)


# ref
def torch_rmsnorm_kernel(a_tensor, g_tensor):
    # Square the tensor (element-wise)
    in_square = a_tensor.pow(2)
    # Calculate means in the free dimension
    mean = in_square.mean(dim=1, keepdim=True)
    # Scale by reciprocal of sqrt(mean)
    tensor = a_tensor * torch.rsqrt(mean)

    # Scale the output by the weight
    return tensor * g_tensor


def _run_demo():
    triton_viz_enabled = True
    kernel_grid = (1, 1, 1)
    b_dim, d_dim = 32, 32
    a_tensor = torch.arange(b_dim * d_dim).float().view(b_dim, d_dim)
    g_tensor = torch.arange(d_dim).float()
    result = torch.empty_like(a_tensor).numpy()
    kernel_args = (a_tensor.numpy(), g_tensor.numpy(), result)

    if triton_viz_enabled:
        print("Executing kernel with NKI interpreter...")
        traced_kernel = triton_viz.trace(client=Tracer(), backend="nki")(
            nki_rmsnorm_kernel
        )
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

        try:
            triton_viz.launch(share=False)
        except Exception as e:
            print(f"\nError during visualization: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Executing NKI JIT-ed matmul_kernel...")
        compiled_kernel = nki.jit(nki_rmsnorm_kernel, kernel_return=False)
        nki.simulate_kernel(compiled_kernel[kernel_grid], *kernel_args)

    z2 = result
    z1 = torch_rmsnorm_kernel(a_tensor, g_tensor).numpy()
    print(np.max(np.abs(z1 - z2)))
    assert np.allclose(z1, z2)


if __name__ == "__main__":
    _run_demo()
