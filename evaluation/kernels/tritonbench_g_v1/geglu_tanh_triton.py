
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh


def calculate_settings(n_cols):
    # This function calculates the BLOCK_SIZE and num_warps based on n_cols.
    # For simplicity, let's assume BLOCK_SIZE is 128 and num_warps is 4.
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

@triton.jit
def _geglu_tanh_forward_kernel(
    a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)

@triton.jit
def _geglu_tanh_backward_kernel(
    dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)

    db_row = dc_row * geglu_a

    # Gradient w.r.t. a can be computed with:
    # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * a_row
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    )
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)

def geglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)

def geglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return a.view(*ori_shape), b.view(*ori_shape)




##################################################################################################################################################


import torch

# Test case
def test_geglu():
    results = {}
    
    # Test case 1
    a = torch.randn(2, 128, dtype=torch.float32, device='cuda')
    b = torch.randn(2, 128, dtype=torch.float32, device='cuda')
    dc = torch.randn(2, 128, dtype=torch.float32, device='cuda')
    a_out, b_out, c_out = geglu_forward(a, b)
    da_out, db_out = geglu_backward(a, b, dc)
    results['test_case_1'] = (a_out, b_out, c_out, da_out, db_out)

    # Test case 2: Different batch size
    a = torch.randn(3, 128, dtype=torch.float32, device='cuda')
    b = torch.randn(3, 128, dtype=torch.float32, device='cuda')
    dc = torch.randn(3, 128, dtype=torch.float32, device='cuda')
    a_out, b_out, c_out = geglu_forward(a, b)
    da_out, db_out = geglu_backward(a, b, dc)
    results['test_case_2'] = (a_out, b_out, c_out, da_out, db_out)

    # Test case 3: Different column size
    a = torch.randn(2, 256, dtype=torch.float32, device='cuda')
    b = torch.randn(2, 256, dtype=torch.float32, device='cuda')
    dc = torch.randn(2, 256, dtype=torch.float32, device='cuda')
    a_out, b_out, c_out = geglu_forward(a, b)
    da_out, db_out = geglu_backward(a, b, dc)
    results['test_case_3'] = (a_out, b_out, c_out, da_out, db_out)

    # Test case 4: Single row input
    a = torch.randn(1, 128, dtype=torch.float32, device='cuda')
    b = torch.randn(1, 128, dtype=torch.float32, device='cuda')
    dc = torch.randn(1, 128, dtype=torch.float32, device='cuda')
    a_out, b_out, c_out = geglu_forward(a, b)
    da_out, db_out = geglu_backward(a, b, dc)
    results['test_case_4'] = (a_out, b_out, c_out, da_out, db_out)

    return results

result_gold = test_geglu()

# [4/4] 分支覆盖率已实现。
