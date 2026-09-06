
import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)

def chunk_global_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32
    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z



##################################################################################################################################################


import torch

# Test for chunk_global_cumsum_vector with all possible branch coverage
def test_chunk_global_cumsum_vector():
    B, H, T, S = 2, 3, 4, 5  # Example dimensions
    BS = 32
    s = torch.rand((B, H, T, S), dtype=torch.float32).cuda()

    result_dict = {}
    
    # First case: Default configuration
    result_dict["test_case_1"] = chunk_global_cumsum_vector(s)
    
    # Third case: Testing with larger tensor dimensions
    s_large = torch.rand((B, H, T * 2, S * 2), dtype=torch.float32).cuda()
    result_dict["test_case_2"] = chunk_global_cumsum_vector(s_large)
    
    # Fourth case: Testing with smaller tensor dimensions
    s_small = torch.rand((B, H, 1, S), dtype=torch.float32).cuda()
    result_dict["test_case_3"] = chunk_global_cumsum_vector(s_small)
    
    return result_dict

# Run all tests
result_gold = test_chunk_global_cumsum_vector()
