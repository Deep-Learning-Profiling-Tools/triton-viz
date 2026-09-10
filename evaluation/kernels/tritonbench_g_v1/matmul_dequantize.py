
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Additional configurations are omitted for brevity...
    ],
    key=['M', 'N', 'K', 'NO_GROUPS'],
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N//8) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is K // groupsize.
    Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    bits = 4
    infearure_per_bits = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    if NO_GROUPS:
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32    
        zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
        zeros = zeros * scales
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated    
        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
            ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32    
            zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
            zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16    
        b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
        b = b * scales[None, :] - zeros[None, :]  # Scale and shift
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk  
    c = accumulator.to(tl.float16)  
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_dequantize_int4_gptq(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size, output=None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (..., K) float16
    qweight is of shape (K//8, N) int32
    scales is of shape (G, N) float16
    qzeros is of shape (G, N//8) int32
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = K // groupsize

    Returns C of shape (..., N) float16
    """
    assert x.shape[-1] == (qweight.shape[0] * 8), "A must be a multiple of 8 in the last dimension"
    assert x.is_contiguous(), "A must be contiguous"

    M, K = x.shape
    N = qweight.shape[1]

    if output is None:
        inplace = False
        output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    else:
        inplace = True

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul4_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size, group_size == K,
    )
    if not inplace:
        return output


@triton.autotune(
    configs=[
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Additional configurations are omitted for brevity...
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bzp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
    ):
    """
    Matrix multiplication kernel with dequantization for 4-bit integers
    A: [M, K], B: [K//8, N], C: [M, N]
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        bs_ptrs = bs_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
            + offs_bn[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
            + (offs_bn[None, :] // 8) * stride_bzpn
        b_shift_bits = (offs_k[:, None] % 8) * 4 
        bzp_shift_bits = (offs_bn[None, :] % 8) * 4
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        bs = tl.load(bs_ptrs)
        bzp = tl.load(bzp_ptrs)

        int_b = (b >> b_shift_bits) & 0xF
        int_bzp = (bzp >> bzp_shift_bits) & 0xF
        b = ((int_b - int_bzp) * bs).to(tl.float16)
        accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    """
    Perform matrix multiplication and dequantize the result from 4-bit integers.
    """
    assert x.is_contiguous(), "A must be contiguous"
    assert qweight.is_contiguous(), "B must be contiguous"  
    M, K = x.shape
    N = scales.shape[1]
    if output is None:
        output = torch.zeros((M, N), device=x.device, dtype=torch.float16)  
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        # Additional configurations are omitted for brevity...
    ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize tile [BLOCK_SIZE_K, BLOCK_SIZE_N] in full precision.
    We should assert BLOCK_SIZE_N % 8 == 0.
    weight[K // 8, N], scale[K // group_size, N], zp[K // group_size, N // group_size]
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    fpb_offs = offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
    b_offs = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = (offs_k[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
    bs_offs = (offs_k[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
    b_shift = (offs_k[:, None] % 8) * 4
    bzp_shift = (offs_n[None, :] % 8) * 4
    fp_weight = (((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b
    tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)


def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    Kw, N = b.shape
    K = Kw * 8
    fp_b = torch.ones((K, N), device=device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        K, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return fp_b


def matmul_dequantize_int4_s1(a, b, b_scale, b_zero_point, group_size=128, out=None):
    """
    Matmul dequantize int4 s1 dequantize weight to `fp_b` and do fp16 torch.mm,
    this is for `prefill` stage, since weight size is fixed so is dequantize overhead,
    perfill stage have more tokens to amortize dequant cost.
    """
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=out)
    fp_b = None
    return out




##################################################################################################################################################


import torch

# Test for matmul_dequantize_int4_gptq
def test_multiple_matmul():
    M, K, N = 128, 256, 512
    group_size_1 = 32
    group_size_2 = 128

    x = torch.randn((M, K), dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 16, (K // 8, N), dtype=torch.int32, device='cuda')
    scales = torch.randn((K // group_size_1, N), dtype=torch.float16, device='cuda')
    qzeros = torch.randint(0, 16, (K // group_size_1, N // 8), dtype=torch.int32, device='cuda')

    # Test case 1
    output_1 = matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, group_size_1)

    # Test case 2
    output_2 = matmul_dequantize_int4_s2(x, qweight, scales, qzeros, group_size_2)

    # Test case 3
    output_3 = matmul_dequantize_int4_s1(x, qweight, scales, qzeros, group_size_2)

    return {
        "test_case_1": output_1,
        "test_case_2": output_2,
        "test_case_3": output_3
    }

# Run tests
result_gold = test_multiple_matmul()
