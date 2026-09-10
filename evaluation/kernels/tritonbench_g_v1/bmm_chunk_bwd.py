import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, res_ptr,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))

def _bmm_chunk_bwd(a, dout, residual=None, out=None):
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        assert residual.shape == (batch, seqlen, k) if not has_groups else (batch, seqlen, ngroups, k)
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    if out is not None:
        assert out.shape == a.shape
        assert out.stride(-1) == 1 or out.stride(1) == 1
    else:
        out = torch.empty_like(a)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16 or dout.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(k, META['BLOCK_SIZE_N']), batch,
                    nchunks if not has_groups else nchunks * ngroups)
    residual_strides = ((residual.stride(0), residual.stride(1), 0 if not has_groups else residual.stride(2),
                         residual.stride(-1))
                        if residual is not None else (0, 0, 0, 0))
    with torch.cuda.device(a.device.index):
        _bmm_chunk_bwd_kernel[grid](
            a, dout, out, residual,
            int(seqlen), int(chunk_size), int(k), int(ngroups if has_groups else 1),
            a.stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a.stride(-1),
            dout.stride(0), dout.stride(1), 0 if not has_groups else dout.stride(2), dout.stride(-2), dout.stride(-1),
            out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-1),
            residual_strides[0], residual_strides[1], residual_strides[2], residual_strides[3],
            dot_dtype,
            HAS_RESIDUAL=residual is not None,
        )
    return out




##################################################################################################################################################


import torch

# Test for _bmm_chunk_bwd
def test_bmm_chunk_bwd():
    results = {}
    
    # Test case 1: Without groups, no residual
    a = torch.randn(2, 128, 64, device='cuda', dtype=torch.float16)
    dout = torch.randn(2, 4, 32, 32, device='cuda', dtype=torch.float16)
    out = _bmm_chunk_bwd(a, dout)
    results['test_case_1'] = out.shape

    # Test case 2: With groups, with residual
    a = torch.randn(2, 128, 4, 64, device='cuda', dtype=torch.float16)
    dout = torch.randn(2, 4, 4, 32, 32, device='cuda', dtype=torch.float16)
    residual = torch.randn(2, 128, 4, 64, device='cuda', dtype=torch.float16)
    out = _bmm_chunk_bwd(a, dout, residual=residual)
    results['test_case_2'] = out.shape

    return results

# Run tests
result_gold = test_bmm_chunk_bwd()
