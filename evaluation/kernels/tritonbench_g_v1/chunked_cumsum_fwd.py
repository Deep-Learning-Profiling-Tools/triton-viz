
import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # Triton kernel implementation for chunked cumulative sum forward pass
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.log(1 + tl.exp(dt)), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Function to perform the forward cumulative sum operation in chunks.

    Arguments:
    - dt: (batch, seqlen, nheads), the input tensor.
    - A: (nheads,), the scaling factors.
    - chunk_size: The size of each chunk to process at a time.
    - dt_bias: (nheads,), optional, biases for dt if applicable.
    - dt_softplus: Boolean, whether to apply the softplus operation to dt.
    - dt_limit: Tuple, (min, max) limits for clamping dt values.

    Returns:
    - dA_cumsum: Cumulative sum result.
    - dt_out: Modified dt after processing.
    """
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_cumsum,
            int(batch), int(seqlen), int(nheads), int(chunk_size),
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out




##################################################################################################################################################


import torch

def test_chunk_cumsum_fwd():
    # Test case 1: Without dt_bias and without dt_softplus
    dt = torch.rand(2, 10, 4, device='cuda')  # (batch, seqlen, nheads)
    A = torch.rand(4, device='cuda')  # (nheads,)
    chunk_size = 5
    dA_cumsum_1, dt_out_1 = _chunk_cumsum_fwd(dt, A, chunk_size)

    # Test case 2: With dt_bias and without dt_softplus
    dt_bias = torch.rand(4, device='cuda')  # (nheads,)
    dA_cumsum_2, dt_out_2 = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias)

    # Test case 3: Without dt_bias and with dt_softplus
    dA_cumsum_3, dt_out_3 = _chunk_cumsum_fwd(dt, A, chunk_size, dt_softplus=True)

    # Test case 4: With dt_bias and with dt_softplus
    dA_cumsum_4, dt_out_4 = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)

    return {
        "test_case_1": (dA_cumsum_1, dt_out_1),
        "test_case_2": (dA_cumsum_2, dt_out_2),
        "test_case_3": (dA_cumsum_3, dt_out_3),
        "test_case_4": (dA_cumsum_4, dt_out_4),
    }

result_gold = test_chunk_cumsum_fwd()
