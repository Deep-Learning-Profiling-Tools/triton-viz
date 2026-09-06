
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['K', 'N'],
)


@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_bk, stride_bn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n[None, :]
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)

def matmul_dequantize_int8(a, b, b_scale, out=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    if out == None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    dequantize_kernel[grid](
        b, b_scale, fp_b,
        K, N,
        b.stride(0), b.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    torch.mm(a, fp_b, out=c)
    return c




##################################################################################################################################################


import torch

# Define the test function
def test_matmul_dequantize_int8():
    # Define the dimensions
    M, K, N = 64, 128, 256  # Example dimensions

    # Create input tensors
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')  # Matrix A
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')  # Matrix B (int8)
    b_scale = torch.rand((N,), dtype=torch.float32, device='cuda')  # Scale factors for B

    # Create different configurations to test all branches
    test_cases = {}

    for config in [
        {'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 3, 'num_warps': 4},
        {'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'num_stages': 3, 'num_warps': 8},
        {'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'num_stages': 4, 'num_warps': 4},
        {'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8},
    ]:
        # Override the config
        grid = lambda META: (
            triton.cdiv(K, config['BLOCK_SIZE_K']), triton.cdiv(N, config['BLOCK_SIZE_N']),
        )

        # Run the kernel with the current configuration
        fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
        dequantize_kernel[grid](
            b, b_scale, fp_b,
            K, N,
            b.stride(0), b.stride(1),
            fp_b.stride(0), fp_b.stride(1)
        )
        result = torch.mm(a, fp_b)

        # Store the result in the test_cases dictionary
        test_cases[f'test_case_{config["BLOCK_SIZE_N"]}_{config["BLOCK_SIZE_K"]}'] = result

    return test_cases

# Execute the test and store the results
result_gold = test_matmul_dequantize_int8()
