from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    q,
    k,
    cos,
    sin,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,  # token range length
):
    cur_head_idx = tl.program_id(0)
    cur_token_block_idx = tl.program_id(1)

    tokens_range = cur_token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)
    loaded_sin = tl.load(sin + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)

    off_q0 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )
    loaded_q0 = tl.load(
        q + off_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

    tl.store(
        q + off_q0,
        out_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k0 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range0[None, None, :] * head_dim_stride
        )
        off_k1 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range1[None, None, :] * head_dim_stride
        )
        loaded_k0 = tl.load(
            k + off_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens),
            other=0.0,
        )
        loaded_k1 = tl.load(
            k + off_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens),
            other=0.0,
        )
        out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
        out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
        tl.store(
            k + off_k0,
            out_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens),
        )
        tl.store(
            k + off_k1,
            out_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens),
        )


@triton.jit
def fused_rotary_embedding_kernel_v2(
    q,
    k,
    cos,
    sin,
    kv_cache,
    BLOCK_TABLES,
    context_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    cacheb_stride,
    cacheh_stride,
    cachebs_stride,
    cached_stride,
    bts_stride,
    btb_stride,
    block_size,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_head_index = tl.program_id(0)
    if block_head_index >= Q_HEAD_NUM:
        return
    block_token_index = tl.program_id(1)

    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_q0 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range0 * head_dim_stride
    off_q1 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range1 * head_dim_stride
    off_k0 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range0 * head_dim_stride
    off_k1 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(
        q + off_q0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
    )

    loaded_k0 = tl.load(
        k + off_k0,
    )

    loaded_k1 = tl.load(
        k + off_k1,
    )

    off_cos_sin = block_token_index * cos_token_stride + dim_range0 * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(block_token_index < q_total_tokens), other=0.0)
    loaded_sin = tl.load(sin + off_cos_sin, mask=(block_token_index < q_total_tokens), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos  # total_tokens, head_num, head_dim

    past_kv_seq_len = tl.load(context_lengths + block_token_index) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + block_token_index * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride, mask=(block_token_index < q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride

    kv_range0 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range0 * cached_stride
    )
    kv_range1 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range1 * cached_stride
    )

    tl.store(
        kv_cache + kv_range0,
        out_k0,
    )
    tl.store(
        kv_cache + kv_range1,
        out_k1,
    )

    # concat
    tl.store(
        q + off_q0,
        out_q0,
    )
    tl.store(
        q + off_q1,
        out_q1,
    )


def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
):
    """
    Args:
        q: query tensor, [total_tokens, head_num, head_dim]
        k: key tensor, [total_tokens, kv_head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        k_cache (torch.Tensor):  Blocked key cache. [num_blocks, num_kv_heads, block_size, head_dim]
        kv_lengths, Past key/value sequence lengths plus current sequence length for each sequence. [bsz]
        block_tables: Block tables for each sequence. [bsz, max_blocks_per_sequence]
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    assert q.size(0) == k.size(0)
    BLOCK_TOKENS = 4

    if head_dim >= 512:
        num_warps = 16
    elif head_dim >= 256:
        num_warps = 8
    else:
        num_warps = 4

    k_head_num = k.size(1)
    q_token_stride, q_head_stride, head_dim_stride = q.stride()
    k_token_stride, k_head_stride, _ = k.stride()
    cos_token_stride, cos_stride = cos.stride()

    assert q_head_num % k_head_num == 0
    kv_group_num = q_head_num // k_head_num

    if k_cache == None:
        grid = lambda META: (
            q_head_num,
            triton.cdiv(q_total_tokens, META["BLOCK_TOKENS"]),
        )
        rotary_embedding_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_token_stride,
            q_head_stride,
            k_token_stride,
            k_head_stride,
            head_dim_stride,
            cos_token_stride,
            cos_stride,
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            KV_GROUP_NUM=kv_group_num,
            HEAD_DIM=head_dim,
            BLOCK_TOKENS=BLOCK_TOKENS,
            num_warps=num_warps,
        )
    else:
        grid = (triton.next_power_of_2(q_head_num), q_total_tokens)
        fused_rotary_embedding_kernel_v2[grid](
            q,
            k,
            cos,
            sin,
            k_cache,
            block_tables,
            kv_lengths,
            q_token_stride,
            q_head_stride,
            k_token_stride,
            k_head_stride,
            head_dim_stride,
            cos_token_stride,
            cos_stride,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            k_cache.size(-2),
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
        )
    return




##################################################################################################################################################


def test_rotary_embedding():
    # 测试参数
    total_tokens = 32  # 总 token 数
    head_num = 8       # Query 的头数量
    kv_head_num = 4    # Key/Value 的头数量
    head_dim = 64      # 每个头的维度
    max_position_len = 128  # 最大位置长度
    block_size = 4     # 块大小

    # 创建输入张量
    q = torch.randn((total_tokens, head_num, head_dim), dtype=torch.float32, device='cuda')  # Query
    k = torch.randn((total_tokens, kv_head_num, head_dim), dtype=torch.float32, device='cuda')  # Key
    cos = torch.randn((max_position_len, head_dim), dtype=torch.float32, device='cuda')  # Cosine
    sin = torch.randn((max_position_len, head_dim), dtype=torch.float32, device='cuda')  # Sine

    result = {}

    # 调用 rotary_embedding 分支 1 (不使用 k_cache)
    rotary_embedding(q, k, cos, sin)
    result["test_case_1"] = (q.clone(), k.clone())

    # 创建附加张量用于分支 2
    num_blocks = 4  # Number of blocks in k_cache
    batch_size = 2  # Batch size
    k_cache = torch.randn((num_blocks, kv_head_num, block_size, head_dim), dtype=torch.float32, device='cuda')  # Key cache
    block_tables = torch.randint(0, num_blocks, (batch_size, num_blocks), device='cuda')  # Block tables
    kv_lengths = torch.randint(1, total_tokens, (batch_size,), device='cuda')  # KV lengths

    # 调用 rotary_embedding 分支 2 (使用 k_cache)
    rotary_embedding(q, k, cos, sin, k_cache=k_cache, block_tables=block_tables, kv_lengths=kv_lengths)
    result["test_case_2"] = (q.clone(), k.clone(), k_cache.clone())

    return result

result_gold = test_rotary_embedding()
