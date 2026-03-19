from neuronxcc import nki
import neuronxcc.nki.language as nl

import triton_viz
import numpy as np
import math

TRITON_VIZ_ENABLED = True  # True = enable visualizer, False = run compiler
TRANSPOSED = False  # True = run matmul_kernel, False = run matmul_kernel_lhsT


def matmul_kernel(lhs, rhs, result):
    """NKI matmul_kernel to compute a matrix multiplication operation in a tiled manner

    Args:
        lhs: an input tensor of shape [M, K]. It is the left-hand-side
            argument of the matrix multiplication.
        rhs: an input tensor of shape [K, N]. It is theright-hand-side
            argument of the matrix multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    M, K = lhs.shape
    K_, N = rhs.shape
    assert K == K_, "lhs and rhs must have the same contraction dimension"

    TILE_M = 2
    TILE_K = 4
    TILE_N = 8

    # Use affine_range to loop over tiles
    for m in nl.affine_range(math.ceil(M / TILE_M)):
        for n in nl.affine_range(math.ceil(N / TILE_N)):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.int32, buffer=nl.psum)

            for k in nl.affine_range(math.ceil(K / TILE_K)):
                # Load tiles from lhs and rhs
                lhs_p = nl.arange(TILE_M)[:, None] + m * TILE_M
                lhs_f = nl.arange(TILE_K)[None, :] + k * TILE_K
                lhs_mask = (lhs_p < M) & (lhs_f < K)
                lhs_tile = nl.load(lhs[lhs_p, lhs_f], mask=lhs_mask)

                rhs_p = nl.arange(TILE_K)[:, None] + k * TILE_K
                rhs_f = nl.arange(TILE_N)[None, :] + n * TILE_N
                rhs_mask = (rhs_p < K) & (rhs_f < N)
                rhs_tile = nl.load(rhs[rhs_p, rhs_f], mask=rhs_mask)

                # Accumulate partial-sums into PSUM
                x = nl.matmul(lhs_tile[...], rhs_tile[...], transpose_x=False)
                res_psum += x

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)

            out_p = nl.arange(TILE_M)[:, None] + m * TILE_M
            out_f = nl.arange(TILE_N)[None, :] + n * TILE_N
            out_mask = (out_p < M) & (out_f < N)
            nl.store(
                result[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
                value=res_sb,
                mask=out_mask,
            )


def matmul_kernel_lhsT(lhs, rhs, result):
    """NKI matmul_kernel but the LHS is pre-TRANSPOSED as this is how the NeuronCore Tensor Engine prefers it.

    Args:
        lhs: an input tensor of shape [K,M]. It is the left-hand-side argument
            of the matrix multiplication, delivered TRANSPOSED for optimal performance.
        rhs: an input tensor of shape [K,N]. It is the right-hand-side
            argument of the matrix multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhs.shape
    K_, N = rhs.shape
    assert K == K_, "lhs and rhs must have the same contraction dimension"

    TILE_M = 2
    TILE_K = 4
    TILE_N = 8

    # Use affine_range to loop over tiles
    for m in nl.affine_range(math.ceil(M / TILE_M)):
        for n in nl.affine_range(math.ceil(N / TILE_N)):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.int32, buffer=nl.psum)

            for k in nl.affine_range(math.ceil(K / TILE_K)):
                # Load tiles from lhs and rhs
                lhs_p = nl.arange(TILE_K)[:, None] + k * TILE_K
                lhs_f = nl.arange(TILE_M)[None, :] + m * TILE_M
                lhs_mask = (lhs_p < K) & (lhs_f < M)
                lhs_tile = nl.load(lhs[lhs_p, lhs_f], mask=lhs_mask)

                rhs_p = nl.arange(TILE_K)[:, None] + k * TILE_K
                rhs_f = nl.arange(TILE_N)[None, :] + n * TILE_N
                rhs_mask = (rhs_p < K) & (rhs_f < N)
                rhs_tile = nl.load(rhs[rhs_p, rhs_f], mask=rhs_mask)

                # Accumulate partial-sums into PSUM
                x = nl.matmul(
                    lhs_tile[...], rhs_tile[...], transpose_x=True
                )  # transpose x (LHS)
                res_psum += x

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)

            out_p = nl.arange(TILE_M)[:, None] + m * TILE_M
            out_f = nl.arange(TILE_N)[None, :] + n * TILE_N
            out_mask = (out_p < M) & (out_f < N)
            nl.store(
                result[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
                value=res_sb,
                mask=out_mask,
            )


def _run_demo():
    kernel_grid = (1, 1, 1)
    M = 4
    K = 8
    N = 16
    lhs_small = np.arange(M * K).astype(np.float32).reshape(M, K)
    rhs_small = np.arange(K * N).astype(np.float32).reshape(K, N)
    kernel = matmul_kernel
    if TRANSPOSED:
        # if you don't make contiguous, it'll mess up the positions of highlighted cells in the visualizer
        lhs_small = np.ascontiguousarray(lhs_small.T)
        kernel = matmul_kernel_lhsT

    result = np.empty((M, N), dtype=lhs_small.dtype)
    kernel_args = (lhs_small, rhs_small, result)

    if TRITON_VIZ_ENABLED:
        print("Executing matmul_kernel with NKI interpreter...")
        traced_kernel = triton_viz.trace("tracer", backend="nki")(kernel)
        kernel_instance = traced_kernel[kernel_grid]
        kernel_instance(*kernel_args)
        triton_viz.launch(share=False)
    else:
        print("Executing NKI JIT-ed matmul kernel...")
        compiled_kernel = nki.jit(kernel, kernel_return=False)
        nki.simulate_kernel(compiled_kernel[kernel_grid], *kernel_args)

    z2 = result
    if TRANSPOSED:
        lhs_small = lhs_small.T
    z1 = lhs_small @ rhs_small
    print(np.max(np.abs(z1 - z2)))
    assert np.allclose(z1, z2)


if __name__ == "__main__":
    _run_demo()
