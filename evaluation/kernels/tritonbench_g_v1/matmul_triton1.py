import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    m_size,
    k_size,
    n_size,
    m_block_size: tl.constexpr,
    k_block_size: tl.constexpr,
    n_block_size: tl.constexpr,
):
    # Calculate the m and n blocks to compute using the program id
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks

    m_offsets = tl.arange(0, m_block_size) + m_block * m_block_size
    n_offsets = tl.arange(0, n_block_size) + n_block * n_block_size
    k_offsets = tl.arange(0, k_block_size)

    x_ptrs = x_ptr + m_offsets[:, None] * k_size + k_offsets[None, :]
    y_ptrs = y_ptr + k_offsets[:, None] * n_size + n_offsets[None, :]
    z_ptrs = z_ptr + m_offsets[:, None] * n_size + n_offsets[None, :]

    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    for _ in range(0, k_size, k_block_size):
        x_sub = tl.load(x_ptrs)
        y_sub = tl.load(y_ptrs)
        z += tl.dot(x_sub, y_sub, allow_tf32=False)
        x_ptrs += k_block_size
        y_ptrs += k_block_size * n_size

    tl.store(z_ptrs, z)

def matmul(x, y):
    m_size, k_size = x.shape
    _, n_size = y.shape
    z = torch.empty(m_size, n_size, device='cuda:0')
    def grid(meta):
        return (triton.cdiv(m_size, meta['m_block_size']) * triton.cdiv(n_size, meta['n_block_size']),)
    
    matmul_kernel[grid](
        x,
        y,
        z,
        m_size,
        k_size,
        n_size,
        m_size,
        k_size,
        n_size
    )
    return z



##################################################################################################################################################


import torch

def test_matmul():
    # Test the matmul function with different block sizes
    x = torch.randn(16, 16, device='cuda:0')
    y = torch.randn(16, 16, device='cuda:0')

    # Test case 1
    output1 = matmul(x, y)

    return {
        "test_case_1": output1,
    }

result_gold = test_matmul()
