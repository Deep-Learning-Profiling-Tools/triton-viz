
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index)

    src_data = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (seq_len, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return




##################################################################################################################################################


import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index)

    src_data = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (seq_len, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


#######################################################################################################


import torch

def test_destindex_copy_quantize_kv():
    # Define the input tensors
    batch_size = 2
    head_num = 4
    head_dim = 16
    seq_len = 10
    quant_group_dim = 8

    # Ensure head_dim is divisible by quant_group_dim
    assert head_dim % quant_group_dim == 0

    # Create random input tensors
    K = torch.randn((seq_len, head_num, head_dim), dtype=torch.float32, device='cuda')
    DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32, device='cuda')
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty((seq_len, head_num, head_dim // quant_group_dim), dtype=torch.float32, device='cuda')

    # Case 1: Normal execution (no early exit conditions)
    destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale)
    result_case_1 = {
        "Out": Out,
        "Out_scale": Out_scale
    }

    # Case 2: Handle a small batch size, less than group_size
    batch_size_small = 1
    K_small = torch.randn((batch_size_small, head_num, head_dim), dtype=torch.float32, device='cuda')
    DestLoc_small = torch.randint(0, seq_len, (batch_size_small,), dtype=torch.int32, device='cuda')
    Out_small = torch.empty_like(K_small, dtype=torch.int8)
    Out_scale_small = torch.empty((batch_size_small, head_num, head_dim // quant_group_dim), dtype=torch.float32, device='cuda')

    destindex_copy_quantize_kv(K_small, DestLoc_small, Out_small, Out_scale_small)
    result_case_2 = {
        "Out": Out_small,
        "Out_scale": Out_scale_small
    }

    # Case 3: Modify DestLoc to contain different sequence lengths
    DestLoc_varied = torch.randint(0, seq_len, (seq_len // 2,), dtype=torch.int32, device='cuda')
    Out_varied = torch.empty_like(K, dtype=torch.int8)
    Out_scale_varied = torch.empty((seq_len // 2, head_num, head_dim // quant_group_dim), dtype=torch.float32, device='cuda')

    destindex_copy_quantize_kv(K, DestLoc_varied, Out_varied, Out_scale_varied)
    result_case_3 = {
        "Out": Out_varied,
        "Out_scale": Out_scale_varied
    }

    # Case 4: Head dimension not divisible by quant_group_dim (assert will trigger)
    try:
        head_dim_invalid = 15  # Invalid head_dim
        K_invalid = torch.randn((seq_len, head_num, head_dim_invalid), dtype=torch.float32, device='cuda')
        DestLoc_invalid = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32, device='cuda')
        Out_invalid = torch.empty_like(K_invalid, dtype=torch.int8)
        Out_scale_invalid = torch.empty((seq_len, head_num, head_dim_invalid // quant_group_dim), dtype=torch.float32, device='cuda')

        destindex_copy_quantize_kv(K_invalid, DestLoc_invalid, Out_invalid, Out_scale_invalid)
    except AssertionError as e:
        result_case_4 = str(e)

    return {
        "result_case_1": result_case_1,
        "result_case_2": result_case_2,
        "result_case_3": result_case_3,
        "result_case_4": result_case_4,
    }

result_gold = test_destindex_copy_quantize_kv()
