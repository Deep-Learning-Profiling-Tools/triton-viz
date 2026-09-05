import warnings
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q,
    k,
    v,
    cos,
    sin,
    k_cache,
    v_cache,
    BLOCK_TABLES,
    context_lengths,
    x,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    kcb_stride,
    kch_stride,
    kcsplit_x_stride,
    kcs_stride,
    kcd_stride,
    vcb_stride,
    vch_stride,
    vcs_stride,
    vcd_stride,
    bts_stride,
    btb_stride,
    block_size,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_head_idx = tl.program_id(0)
    cur_token_idx = tl.program_id(1)

    dim_range = tl.arange(0, HEAD_DIM)
    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_q = cur_token_idx * q_token_stride + cur_head_idx * q_head_stride
    off_q0 = off_q + dim_range0 * head_dim_stride
    off_q1 = off_q + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(q + off_q0)
    loaded_q1 = tl.load(q + off_q1)
    off_cos_sin = cur_token_idx * cos_token_stride + dim_range0 * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin)
    loaded_sin = tl.load(sin + off_cos_sin)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0)
    tl.store(q + off_q1, out_q1)

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        cur_k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_kv = cur_token_idx * k_token_stride + cur_k_head_idx * k_head_stride
        off_k0 = off_kv + dim_range0 * head_dim_stride
        off_k1 = off_kv + dim_range1 * head_dim_stride
        loaded_k0 = tl.load(k + off_k0)
        loaded_k1 = tl.load(k + off_k1)

        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

        # NOTE The precondition here is that it's only for unpadded inputs during decoding stage,
        # and so that we could directly use the token index as the sequence index
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - 1

        last_block_idx = past_kv_seq_len // block_size
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * bts_stride + last_block_idx * btb_stride)
        offsets_in_last_block = past_kv_seq_len % block_size
        offsets_cache_base = block_ids * kcb_stride + cur_k_head_idx * kch_stride
        k_range0 = (
            offsets_cache_base
            + offsets_in_last_block * kcs_stride
            + (dim_range0 // x) * kcsplit_x_stride
            + (dim_range0 % x) * kcd_stride
        )
        k_range1 = (
            offsets_cache_base
            + offsets_in_last_block * kcs_stride
            + (dim_range1 // x) * kcsplit_x_stride
            + (dim_range1 % x) * kcd_stride
        )
        tl.store(k_cache + k_range0, out_k0)
        tl.store(k_cache + k_range1, out_k1)

        off_v = off_kv + dim_range * head_dim_stride
        loaded_v = tl.load(v + off_v)
        v_range = (
            block_ids * vcb_stride
            + cur_k_head_idx * vch_stride
            + offsets_in_last_block * vcs_stride
            + dim_range * vcd_stride
        )
        tl.store(v_cache + v_range, loaded_v)


def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
    use_new_kcache_layout: bool = False,
):
    """
    Args:
        q: query tensor, [total_tokens, head_num, head_dim]
        k: key tensor, [total_tokens, kv_head_num, head_dim]
        v: value tensor, [total tokens, kv_head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        k_cache (torch.Tensor):  Blocked key cache. [num_blocks, kv_head_num, block_size, head_dim]
        v_cache (torch.Tensor):  Blocked value cache. [num_blocks, kv_head_num, block_size, head_dim]
        kv_lengths, Past key/value sequence lengths plus current sequence length for each sequence. [bsz]
        block_tables: Block tables for each sequence. [bsz, max_blocks_per_sequence]
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    assert q.size(0) == k.size(0) == v.size(0)

    if head_dim >= 512:
        num_warps = 16
    elif head_dim >= 256:
        num_warps = 8
    else:
        num_warps = 4
    k_head_num = k.size(1)
    kv_group_num = q_head_num // k_head_num

    # For KCache and VCache with the same layout
    x = head_dim
    kcsplit_x_stride, kcs_stride, kcd_stride = 0, k_cache.stride(2), k_cache.stride(3)
    # For KCache layout [num_blocks, num_kv_heads, head_dim//x, block_size, x]
    if use_new_kcache_layout:
        assert (
            k_cache.dim() == 5
            and k_cache.shape[1] == v_cache.shape[1]
            and k_cache.shape[2] * k_cache.shape[4] == v_cache.shape[3]
        ), f"Invalid KCache shape {k_cache.shape} and VCache shape {v_cache.shape}"
        x = k_cache.size(-1)
        kcsplit_x_stride, kcs_stride, kcd_stride = k_cache.stride()[-3:]

    grid = (q_head_num, q_total_tokens)
    decoding_fused_rotary_embedding_kernel[grid](
        q,
        k,
        v,
        cos,
        sin,
        k_cache,
        v_cache,
        block_tables,
        kv_lengths,
        x,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q.stride(2),
        cos.stride(0),
        cos.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        kcsplit_x_stride,
        kcs_stride,
        kcd_stride,
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        k_cache.size(-2),
        KV_GROUP_NUM=kv_group_num,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
    )
    return




##################################################################################################################################################


def test_decoding_fused_rotary_embedding():
    # 定义测试参数
    total_tokens = 16       # 总 token 数
    q_head_num = 8          # Query 的头数量
    kv_head_num = 4         # Key/Value 的头数量
    head_dim = 64           # 每个头的维度
    max_position_len = 128  # 最大位置长度
    block_size = 4          # 块大小
    num_blocks = 4          # Key/Value cache 块数量
    batch_size = 2          # 批大小

    # 初始化输入张量
    q = torch.randn((total_tokens, q_head_num, head_dim), dtype=torch.float32, device='cuda')  # Query
    k = torch.randn((total_tokens, kv_head_num, head_dim), dtype=torch.float32, device='cuda')  # Key
    v = torch.randn((total_tokens, kv_head_num, head_dim), dtype=torch.float32, device='cuda')  # Value
    cos = torch.randn((max_position_len, head_dim), dtype=torch.float32, device='cuda')  # Cosine
    sin = torch.randn((max_position_len, head_dim), dtype=torch.float32, device='cuda')  # Sine

    # 初始化 Key/Value 缓存和辅助张量
    k_cache = torch.zeros((num_blocks, kv_head_num, block_size, head_dim), dtype=torch.float32, device='cuda')
    v_cache = torch.zeros((num_blocks, kv_head_num, block_size, head_dim), dtype=torch.float32, device='cuda')
    block_tables = torch.randint(0, num_blocks, (batch_size, num_blocks), dtype=torch.int32, device='cuda')
    kv_lengths = torch.randint(1, total_tokens, (batch_size,), dtype=torch.int32, device='cuda')

    results = {}

    # 测试默认 k_cache 布局
    decoding_fused_rotary_embedding(
        q=q,
        k=k,
        v=v,
        cos=cos,
        sin=sin,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        kv_lengths=kv_lengths,
        use_new_kcache_layout=False,
    )
    results['test_case_1'] = {
        'q_shape': q.shape,
        'k_cache_shape': k_cache.shape,
        'v_cache_shape': v_cache.shape
    }

    # 测试新的 k_cache 布局
    x = 16  # 分割因子
    k_cache = torch.zeros((num_blocks, kv_head_num, head_dim // x, block_size, x), dtype=torch.float32, device='cuda')
    v_cache = torch.zeros((num_blocks, kv_head_num, block_size, head_dim), dtype=torch.float32, device='cuda')

    # 测试新的 k_cache 布局
    decoding_fused_rotary_embedding(
        q=q,
        k=k,
        v=v,
        cos=cos,
        sin=sin,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        kv_lengths=kv_lengths,
        use_new_kcache_layout=True,
    )
    results['test_case_2'] = {
        'q_shape': q.shape,
        'k_cache_shape': k_cache.shape,
        'v_cache_shape': v_cache.shape
    }

    return results

result_gold = test_decoding_fused_rotary_embedding()

