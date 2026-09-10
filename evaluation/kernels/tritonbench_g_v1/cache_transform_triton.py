import torch
import triton
import triton.language as tl


@triton.jit
def prefill_cache_kernel(
    cos_cache,
    sin_cache,
    cumsum_lengths,
    cos_output,
    sin_output,
    cache_stride,
    hidden_stride,
    total_length,
    HIDDEN_DIM: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx0 = tl.program_id(axis=0)
    idx1 = tl.program_id(axis=1)
    idx = idx0 * BLOCK_SIZE + idx1

    # original seq_idx and pos
    cumsum_lens = tl.load(cumsum_lengths + tl.arange(0, N_ELEMENTS))
    ori_seq_idx = idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, 0))
    cos_cache_part = tl.load(
        cos_cache + ori_seq_idx * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride, mask=idx < total_length
    )
    sin_cache_part = tl.load(
        sin_cache + ori_seq_idx * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride, mask=idx < total_length
    )
    tl.store(
        cos_output + idx * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride,
        cos_cache_part,
        mask=idx < total_length,
    )
    tl.store(
        sin_output + idx * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride,
        sin_cache_part,
        mask=idx < total_length,
    )


@triton.jit
def decoding_cache_kernel(
    cos_cache,
    sin_cache,
    lengths,
    cos_output,
    sin_output,
    cache_stride,
    hidden_stride,
    HIDDEN_DIM: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ori_seq_idx = tl.load(lengths + idx, mask=(idx < NUM_SEQS), other=None)  # [BLOCK_SIZE,]
    cos_cache_part = tl.load(
        cos_cache + ori_seq_idx[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride,
        mask=idx[:, None] < NUM_SEQS,
    )
    sin_cache_part = tl.load(
        sin_cache + ori_seq_idx[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride,
        mask=idx[:, None] < NUM_SEQS,
    )
    tl.store(
        cos_output + (idx[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride),
        cos_cache_part,
        mask=idx[:, None] < NUM_SEQS,
    )
    tl.store(
        sin_output + (idx[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride),
        sin_cache_part,
        mask=idx[:, None] < NUM_SEQS,
    )


def get_xine_cache(lengths: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, is_prompts: bool = False):
    assert cos_cache.shape[1] == sin_cache.shape[1]
    _, hidden_dim = cos_cache.shape
    num_seqs = lengths.numel()

    if hidden_dim >= 256:
        num_warps = 16
    elif hidden_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    cache_stride = cos_cache.stride(0)
    hidden_stride = cos_cache.stride(1)

    if is_prompts:
        BLOCK_SIZE = 16
        total_length = lengths.sum().item()
        cumsum_lens = torch.cumsum(lengths, dim=0)
        cos_output = torch.empty((total_length, hidden_dim), dtype=cos_cache.dtype, device=cos_cache.device)
        sin_output = torch.empty((total_length, hidden_dim), dtype=sin_cache.dtype, device=sin_cache.device)
        grid = (triton.cdiv(total_length, BLOCK_SIZE), BLOCK_SIZE)
        prefill_cache_kernel[grid](
            cos_cache,
            sin_cache,
            cumsum_lens,
            cos_output,
            sin_output,
            cache_stride,
            hidden_stride,
            total_length,
            HIDDEN_DIM=hidden_dim,
            N_ELEMENTS=triton.next_power_of_2(num_seqs),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        BLOCK_SIZE = 4
        nlengths = torch.as_tensor(lengths) - 1
        cos_output = torch.empty((num_seqs, hidden_dim), dtype=cos_cache.dtype, device=cos_cache.device)
        sin_output = torch.empty((num_seqs, hidden_dim), dtype=sin_cache.dtype, device=sin_cache.device)
        grid = (triton.cdiv(num_seqs, BLOCK_SIZE),)
        decoding_cache_kernel[grid](
            cos_cache,
            sin_cache,
            nlengths,
            cos_output,
            sin_output,
            cache_stride,
            hidden_stride,
            HIDDEN_DIM=hidden_dim,
            NUM_SEQS=num_seqs,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return cos_output, sin_output




##################################################################################################################################################


def test_get_xine_cache():
    # 测试参数
    num_seqs = 8            # 序列数量
    seq_len = 10            # 每个序列的长度
    hidden_dim = 64         # 隐藏层维度
    max_length = 20         # 最大序列长度
    is_prompts_list = [True, False]

    # 创建输入张量
    lengths = torch.randint(1, max_length, (num_seqs,), dtype=torch.int32, device='cuda')
    cos_cache = torch.randn((max_length, hidden_dim), dtype=torch.float32, device='cuda')
    sin_cache = torch.randn((max_length, hidden_dim), dtype=torch.float32, device='cuda')

    results = {}
    
    for i, is_prompts in enumerate(is_prompts_list, start=1):
        cos_output, sin_output = get_xine_cache(lengths, cos_cache, sin_cache, is_prompts=is_prompts)
        results[f"test_case_{i}"] = (cos_output.shape, sin_output.shape)

    return results

result_gold = test_get_xine_cache()
