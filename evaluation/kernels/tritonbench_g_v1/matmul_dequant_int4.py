import time
import torch
import triton
import triton.language as tl


@triton.autotune(
	configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), 
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
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    # zeros_ptrs is set up such that it repeats elements along the N axis 8 times
    zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
    # shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    # If G == 1, scales and zeros are the same for all K, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
        # Unpack zeros
        zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
        # zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16
        zeros = zeros * scales
    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
            ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
            # Unpack zeros
            zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
            zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16	
        # Now we need to unpack b (which is 4-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
        b = b * scales[None, :] - zeros[None, :]  # Scale and shift
        # print("data type", a, b)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk  
    c = accumulator.to(c_ptr.dtype.element_ty)  
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
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
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out is None:
        # Allocates output.
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=out)
    fp_b = None
    return out


def quantize_int4(weight, group_size=128, tp_rank=0):
    # Weight shape: [H1 // 8, H2]
    # Scale shape: [H1 // group_size, H2]
    # zero_pint shape: [H1 // group_size, H2 // 8]

    weight = weight.transpose(1, 0)
    h1, h2 = weight.shape
    assert h1 % 8 == 0 and h2 % 8 == 0, "H1 {} H2 {}".format(h1, h2)
    assert h2 % group_size == 0, "H1 {} H2 {}".format(h1, h2)
    weight = weight.contiguous().view(-1, group_size).cuda(tp_rank)
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min 
    scale = weight_range / (2 ** 4 - 1)
    zero_point = (-weight_min / scale).round().clamp(0, 15).to(torch.int32)
    weight = (weight / scale + zero_point).round().clamp(0, 15).to(torch.int32).view(h1, h2)
    int_weight = torch.empty(h1, h2 // 8).to(torch.int32).to(weight.device)
    int_zero_point = torch.zeros(h1 // 8, h2 // group_size).to(torch.int32).to(weight.device)
    zero_point = zero_point.view(h1, -1)
    scale = scale.view(h1, -1)
    # pack 8 int4 in an int32 number.
    # Weight pack in row.
    for pack in range(0, h2, 8):
        for i in range(8):
            int_weight[:, pack // 8] += weight[:, pack + i] << (i * 4)
    # zero point pack in col.
    for pack in range(0, h1, 8):
        for i in range(8):
            int_zero_point[pack // 8, :] += zero_point[pack + i, :] << (i * 4)
    '''
    fp_weight = torch.zeros(h1, h2).half().to(weight.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_weight[pack * 8 + i, :] = \
                ((int_weight[pack, :] << (28 - i * 4) >> 28) + 16) % 16
    print((fp_weight - weight).abs().sum())

    fp_zp = torch.zeros(zero_point.shape).half().to(zero_point.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_zp[pack * 8 + i, :] = \
                (int_zero_point[pack, :] >> (i * 4)) & 15

    print((fp_zp - zero_point).abs().sum())
    '''
    weight = None
    return int_weight.transpose(1, 0).contiguous(), scale.transpose(1, 0).contiguous(), int_zero_point.transpose(1, 0).contiguous(), group_size



##################################################################################################################################################


import torch

def test_correct_int4_s1(M=32, K=4096, N=4096):
    group_size = 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=group_size)
    results = {}
    
    # Test case 1
    triton_output_1 = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point, group_size)
    results['test_case_1'] = triton_output_1
    
    # Test case 2
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=256)
    triton_output_2 = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point, 256)
    results['test_case_2'] = triton_output_2
    
    # Test case 3
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=64)
    triton_output_3 = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point, 64)
    results['test_case_3'] = triton_output_3
    
    # Test case 4
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=32)
    triton_output_4 = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point, 32)
    results['test_case_4'] = triton_output_4
    
    return results

result_gold = test_correct_int4_s1()
