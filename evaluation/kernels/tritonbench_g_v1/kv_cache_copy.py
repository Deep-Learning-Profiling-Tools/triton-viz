import torch
import triton
import triton.language as tl


# supports two types of cache layouts
# 1. [num_blocks, num_kv_heads, block_size, head_dim]
# 2. [num_blocks, num_kv_heads, head_dim // x, block_size, x]
@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K,
    V,
    KCache,
    VCache,
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcd,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_vcd,
    stride_bts,
    stride_btb,
    block_size,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        offsets_kcache = (
            block_id * stride_kcb
            + cur_kv_head_idx * stride_kch
            + split_x * stride_kcsplit_x
            + offsets_in_last_block * stride_kcs
            + range_x
        )
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = (
            block_id * stride_vcb
            + cur_kv_head_idx * stride_vch
            + offsets_in_last_block * stride_vcs
            + offsets_dmodel_x_partition * stride_vcd
        )
        tl.store(VCache + offsets_vcache, v)
    return


def copy_kv_to_blocked_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    use_new_kcache_layout: bool = False,
):
    """
    Copy keys or values to the blocked key/value cache during decoding stage.

    Args:
        k (torch.Tensor): [bsz, 1, num_kv_heads, head_dim]/[bsz, num_kv_heads, head_dim] - Keys during decoding with seq len 1.
        v (torch.Tensor): [bsz, 1, num_kv_heads, head_dim]/[bsz, num_kv_heads, head_dim] - Values during decoding with seq len 1.
        k_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_dim] - Blocked key cache.
        v_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_dim] - Blocked value cache.
        kv_lengths (torch.Tensor): [bsz] - Past key/value sequence lengths plus current sequence length for each sequence.
        block_tables (torch.Tensor): [bsz, max_blocks_per_sequence] - Block tables for each sequence.
        use_new_kcache_layout (bool): Whether to use the new layout for kcache. Default to False.
    """
    k_cache_shape = k_cache.shape
    v_cache_shape = v_cache.shape

    if use_new_kcache_layout:
        assert (
            len(k_cache_shape) == 5
            and k_cache_shape[1] == v_cache_shape[1]
            and k_cache_shape[2] * k_cache_shape[4] == v_cache_shape[3]
        ), f"Invalid KCache shape {k_cache_shape} and VCache shape {v_cache_shape}"
    else:
        assert k.size(-1) == k_cache_shape[-1], "Incompatible head dim"
        assert (
            k_cache_shape == v_cache_shape
        ), f"Incompatible KCache shape {k_cache_shape} and VCache shape {v_cache_shape}"
    assert v.size(-1) == v_cache_shape[-1], "Incompatible head dim"

    k = k.squeeze(1) if k.dim() == 4 else k
    assert k.dim() == 3, f"Incompatible k dim {k.dim()}"
    v = v.squeeze(1) if v.dim() == 4 else v
    assert v.dim() == 3, f"Incompatible v dim {v.dim()}"

    bsz, num_kv_heads, head_dim = k.shape
    assert kv_lengths.shape[0] == block_tables.shape[0] == bsz, (
        f"Got incompatible batch size (number of seqs):\n"
        f"  Past kv sequence lengths bsz {kv_lengths.shape[0]}; "
        f" block tables bsz {block_tables.shape[0]}, input k batch size {bsz}"
    )

    # Modify if the shape of kv cahce is changed.
    block_size = k_cache.size(-2)

    x = head_dim
    stride_kcsplit_x, stride_kcs, stride_kcd = 0, k_cache.stride(2), k_cache.stride(3)
    if use_new_kcache_layout:
        x = k_cache.size(-1)
        stride_kcsplit_x, stride_kcs, stride_kcd = k_cache.stride()[2:]

    num_warps = 8 if head_dim > 128 else 4
    grid = (bsz, num_kv_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        kv_lengths,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        stride_kcsplit_x,
        stride_kcs,
        stride_kcd,
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        block_size,
        HEAD_DIM=head_dim,
        KCACHE_X=x,
        num_warps=num_warps,
    )




##################################################################################################################################################


# Test for copy_kv_to_blocked_cache
def test_copy_kv_to_blocked_cache():
    # Parameters
    bsz = 2
    num_kv_heads = 4
    head_dim = 64
    block_size = 16
    max_blocks_per_sequence = 10

    # Inputs
    k = torch.randn(bsz, 1, num_kv_heads, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(bsz, 1, num_kv_heads, head_dim, dtype=torch.float32, device="cuda")
    k_cache = torch.zeros(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=torch.float32, device="cuda")
    v_cache = torch.zeros(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=torch.float32, device="cuda")
    kv_lengths = torch.tensor([5, 10], dtype=torch.int32, device="cuda")
    block_tables = torch.randint(0, max_blocks_per_sequence, (bsz, max_blocks_per_sequence), dtype=torch.int32, device="cuda")

    # Test with old kcache layout
    copy_kv_to_blocked_cache(k, v, k_cache, v_cache, kv_lengths, block_tables, use_new_kcache_layout=False)

    # Test with new kcache layout
    k_cache_new_layout = torch.zeros(max_blocks_per_sequence, num_kv_heads, head_dim // 8, block_size, 8, dtype=torch.float32, device="cuda")
    v_cache_new_layout = torch.zeros(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=torch.float32, device="cuda")
    copy_kv_to_blocked_cache(k, v, k_cache_new_layout, v_cache_new_layout, kv_lengths, block_tables, use_new_kcache_layout=True)

    # Collect results
    results = {
        "test_case_1": (k_cache.clone(), v_cache.clone()),
        "test_case_2": (k_cache_new_layout.clone(), v_cache_new_layout.clone())
    }
    return results

# Execute the test function
result_gold = test_copy_kv_to_blocked_cache()
