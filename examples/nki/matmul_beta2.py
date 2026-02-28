import nki
import nki.isa as nisa
import nki.language as nl

import numpy as np
import triton_viz

TRITON_VIZ_ENABLED = True


def matmul_kernel(lhsT, rhs, result):
    """Compute tiled matrix multiplication ``result = lhsT.T @ rhs``.

    Args:
        lhsT: Input matrix with shape ``[K, M]``.
        rhs: Input matrix with shape ``[K, N]``.
        result: Output matrix with shape ``[M, N]`` written in place.
    """

    # Verify that the lhsT and rhs have the same contraction dimension.
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    # Lookup the device matrix multiply dimensions.
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Verify that the input matrices are a multiple of the tile dimensions.
    assert (
        M % TILE_M == 0
    ), f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
    assert (
        N % TILE_N == 0
    ), f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
    assert (
        K % TILE_K == 0
    ), f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

    # Use affine_range to loop over tiles
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Allocate a tensor in PSUM
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Declare the tiles on SBUF
                lhsT_tile = nl.ndarray(
                    (TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf
                )
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                # Load tiles from lhsT and rhs
                nisa.dma_copy(
                    dst=lhsT_tile,
                    src=lhsT[
                        k * TILE_K : (k + 1) * TILE_K, m * TILE_M : (m + 1) * TILE_M
                    ],
                )
                nisa.dma_copy(
                    dst=rhs_tile,
                    src=rhs[
                        k * TILE_K : (k + 1) * TILE_K, n * TILE_N : (n + 1) * TILE_N
                    ],
                )

                # Accumulate partial-sums into PSUM
                nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)

            # Copy the result from SBUF to HBM.
            nisa.dma_copy(
                dst=result[
                    m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N
                ],
                src=res_sb,
            )


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
    """Run the matmul example with lhsT ``[128, 128]`` and rhs ``[128, 512]``."""
    kernel_grid = (1, 1, 1)
    m_dim = 128
    k_dim = 128
    n_dim = 512
    lhs_small = np.arange(k_dim * m_dim, dtype=np.float32).reshape(k_dim, m_dim)
    rhs_small = np.arange(k_dim * n_dim, dtype=np.float32).reshape(k_dim, n_dim)
    kernel = matmul_kernel

    result = np.empty((m_dim, n_dim), dtype=lhs_small.dtype)
    kernel_args = (lhs_small, rhs_small, result)
    expected = lhs_small.T @ rhs_small

    if TRITON_VIZ_ENABLED:
        traced_kernel = triton_viz.trace("tracer", backend="nki")(kernel)
        traced_kernel[kernel_grid](*kernel_args)
        assert np.allclose(expected, result)
        print("actual equals expected")
        triton_viz.launch(share=False)
    else:
        _, _, result = _run_with_xla(kernel, kernel_grid, *kernel_args)
        assert np.allclose(expected, result)
        print("actual equals expected")


if __name__ == "__main__":
    _run_demo()
