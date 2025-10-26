from neuronxcc import nki
import neuronxcc.nki.language as nl
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
import numpy as np

tensors = []  # THTODO: rm
records = [triton_viz.core.data.Grid((0, 0, 0))]  # THTODO: rm


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

    tensors.append(lhs)
    tensors.append(rhs)
    tensors.append(result)
    # Use affine_range to loop over tiles
    time_idx = 0
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
                lhs_mask = (lhs_p < M) & (lhs_f < K)
                lhs_tile = nl.load(lhs[lhs_p, lhs_f], mask=lhs_mask)
                records.append(
                    triton_viz.core.data.Load(
                        lhs.data_ptr(),
                        (4 * (lhs_p * K + lhs_f))._value,
                        lhs_mask._value,
                    )
                )  # THTODO: rm
                # NKI meta for viz (HBM->SBUF)
                try:
                    bytes_lhs = int(np.count_nonzero(lhs_mask._value)) * int(
                        lhs.dtype.itemsize
                    )
                except Exception:
                    bytes_lhs = 0
                records[-1].mem_src = "HBM"
                records[-1].mem_dst = "SBUF"
                records[-1].bytes = bytes_lhs
                records[-1].time_idx = time_idx
                records[-1].backend = "nki"
                time_idx += 1

                rhs_p = nl.arange(TILE_K)[:, None] + k * TILE_K
                rhs_f = nl.arange(TILE_N)[None, :] + n * TILE_N
                rhs_mask = (rhs_p < K) & (rhs_f < N)
                rhs_tile = nl.load(rhs[rhs_p, rhs_f], mask=rhs_mask)
                records.append(
                    triton_viz.core.data.Load(
                        rhs.data_ptr(),
                        (4 * (rhs_p * N + rhs_f))._value,
                        rhs_mask._value,
                    )
                )  # THTODO: rm
                try:
                    bytes_rhs = int(np.count_nonzero(rhs_mask._value)) * int(
                        rhs.dtype.itemsize
                    )
                except Exception:
                    bytes_rhs = 0
                records[-1].mem_src = "HBM"
                records[-1].mem_dst = "SBUF"
                records[-1].bytes = bytes_rhs
                records[-1].time_idx = time_idx
                records[-1].backend = "nki"
                time_idx += 1

                # Accumulate partial-sums into PSUM
                x = nl.matmul(lhs_tile[...], rhs_tile[...], transpose_x=False)
                res_psum += x

                # For visualization, use standard matmul convention A(M,K)·B(K,N)->C(M,N)
                # Our lhs_tile is (K, M) physically; transpose to (M, K) for Dot record
                lhs_viz = lhs_tile._value.T
                rhs_viz = rhs_tile._value
                lhs_shape_viz = (lhs_tile.shape[1], lhs_tile.shape[0])  # (M, K)
                rhs_shape_viz = rhs_tile.shape  # (K, N)
                out_shape_viz = res_psum.shape  # (M, N)

                records.append(
                    triton_viz.core.data.Dot(
                        lhs_shape_viz,
                        rhs_shape_viz,
                        out_shape_viz,
                        lhs_viz,
                        rhs_viz,
                        x._value,
                    )
                )  # THTODO: rm
                records[-1].mem_src = "SBUF"
                records[-1].mem_dst = "PSUM"
                records[-1].tile_shape = (TILE_M, TILE_N)
                records[-1].k = TILE_K
                records[-1].time_idx = time_idx
                records[-1].backend = "nki"
                time_idx += 1

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)

            out_p = nl.arange(TILE_M)[:, None] + m * TILE_M
            out_f = nl.arange(TILE_N)[None, :] + n * TILE_N
            out_mask = (out_p < M) & (out_f < N)
            nl.store(
                result[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
                value=res_sb,
            )
            records.append(
                triton_viz.core.data.Store(
                    result.data_ptr(), (4 * (out_p * N + out_f))._value, out_mask._value
                )
            )  # THTODO: rm
            try:
                bytes_out = int(np.count_nonzero(out_mask._value)) * int(
                    result.dtype.itemsize
                )
            except Exception:
                bytes_out = 0
            records[-1].mem_src = "SBUF"
            records[-1].mem_dst = "HBM"
            records[-1].bytes = bytes_out
            records[-1].time_idx = time_idx
            records[-1].backend = "nki"
            time_idx += 1

    return result


TRITON_VIZ = True
kernel_grid = (1, 1, 1)
lhs_small = np.arange(16).astype(np.float32).reshape(4, 4)
rhs_small = np.arange(32).astype(np.float32).reshape(4, 8)
kernel_args = (lhs_small, rhs_small)

if TRITON_VIZ:
    print("Executing matmul_kernel with NKI interpreter...")
    matmul_kernel = triton_viz.trace(clients=Tracer(), backend="nki")(matmul_kernel)
    kk = matmul_kernel[kernel_grid]
    z2 = kk(*kernel_args)

    print(f"Number of launches: {len(launches)}")
    if launches:
        launch = launches[-1]
        launch.records = records  # THTODO: rm
        launch.tensors = tensors  # THTODO: rm
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
        triton_viz.launch(share=True, port=8005)
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback

        traceback.print_exc()
else:
    print("Executing NKI JIT-ed matmul_kernel...")
    matmul_kernel = nki.jit(matmul_kernel)
    z2 = nki.simulate_kernel(matmul_kernel[kernel_grid], *kernel_args)

z1 = lhs_small @ rhs_small
print(np.max(np.abs(z1 - z2)))
assert np.allclose(z1, z2)
