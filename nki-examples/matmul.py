from neuronxcc import nki
import neuronxcc.nki.language as nl

import tp
import torch
import triton_viz
import torch_xla.core.xla_model as xm
from triton_viz.clients import Tracer
from triton_viz.core import config as cfg
from triton_viz.core.trace import launches
import numpy as np
import math

def matmul_kernel(lhs, rhs):
    """NKI matmul_kernel to compute a matrix multiplication operation in a tiled manner

    Args:
        lhs: an input tensor of shape [K,M], where both K and M are multiples for
          128.  It is the left-hand-side argument of the matrix multiplication,
          delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
          is a multiple of 512.  It is the right-hand-side argument of the
          matrix multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    M, K = lhs.shape
    K_, N = rhs.shape
    assert K == K_, "lhs and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

    TILE_M = 2
    TILE_K = 2
    TILE_N = 4

    # Use affine_range to loop over tiles
    m_range = nl.arange(M)[:, None]
    n_range = nl.arange(N)[None, :]
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.int32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Declare the tiles on SBUF
                lhs_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhs.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                # Load tiles from lhs and rhs
                lhs_p = nl.arange(TILE_M)[:, None] + m * TILE_M
                lhs_f = nl.arange(TILE_K)[None, :] + k * TILE_K
                lhs_tile = nl.load(lhs[lhs_p, lhs_f], mask=(lhs_p < M) & (lhs_f < K))
                rhs_p = nl.arange(TILE_K)[:, None] + k * TILE_K
                rhs_f = nl.arange(TILE_N)[None, :] + n * TILE_N
                rhs_tile = nl.load(rhs[rhs_p, rhs_f], mask=(rhs_p < K) & (rhs_f < N))

                # Accumulate partial-sums into PSUM
                res_psum += nl.matmul(lhs_tile[...], rhs_tile[...], transpose_x=False)

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            out_mask = (m * TILE_M <= m_range) & (m_range < (m + 1) * TILE_M)
            out_mask &= (n * TILE_N <= n_range) & (n_range < (n + 1) * TILE_N)
            res_sb = nl.copy(res_psum, dtype=result.dtype)

            #nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
            #         value=res_sb) # TODO: uncomment this
            nl.store(result, value=res_sb, mask=out_mask) # TODO: remove this - incorrect NKI syntax but currently needed for interpreter

    return result

TRITON_VIZ = True
kernel_grid = (1, 1, 1)
lhs_small = np.arange(64).astype(np.float32).reshape(8, 8)
rhs_small = np.arange(128).astype(np.float32).reshape(8, 16)
#lhs_small = np.arange(16).astype(np.float32).reshape(4, 4)
#rhs_small = np.arange(32).astype(np.float32).reshape(4, 8)
kernel_args = (lhs_small, rhs_small)

if TRITON_VIZ:
    tp.log('Executing matmul_kernel with NKI interpreter...')
    matmul_kernel = triton_viz.trace(clients=Tracer(), backend="nki")(matmul_kernel)
    kk = matmul_kernel[kernel_grid]
    z2 = kk(*kernel_args)

    #print(f"Number of launches: {len(launches)}")
    #if launches:
    #    launch = launches[-1]
    #    print(f"Number of records: {len(launch.records)}")
    #    for i, record in enumerate(launch.records):
    #        print(f"Record {i}: {type(record).__name__}")
    #        if hasattr(record, "ptr"):
    #            print(f"  ptr: {record.ptr}")
    #        if hasattr(record, "offsets"):
    #            print(f"  offsets shape: {record.offsets.shape}")
    #        if hasattr(record, "masks"):
    #            print(f"  masks shape: {record.masks.shape}")

    ## Try to launch visualization
    #try:
    #    triton_viz.launch(share=False)
    #except Exception as e:
    #    print(f"\nError during visualization: {e}")
    #    import traceback

    #    traceback.print_exc()
else:
    tp.log('Executing NKI JIT-ed matmul_kernel...')
    matmul_kernel = nki.jit(matmul_kernel)
    z2 = nki.simulate_kernel(matmul_kernel[kernel_grid], *kernel_args)

z1 = lhs_small @ rhs_small
print(np.max(np.abs(z1-z2)))
assert np.allclose(z1, z2)
