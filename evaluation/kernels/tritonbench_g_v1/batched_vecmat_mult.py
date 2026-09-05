import torch
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
        A,  # shape: [dim_m, dim_k]
        B,  # shape: [dim_m, dim_n, dim_k]
        dim_m, dim_n, dim_k,
        output,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

    tl.store(output + output_tile, vecmat)


def batched_vecmat(
    M, N, K, block_m, block_n, block_k, num_warps=4, num_stages=1
):

    A = torch.randn(M, K, device='cuda', dtype=torch.float32)  # shape: [M, K]
    B = torch.randn(M, N, K, device='cuda', dtype=torch.float32)  # shape: [M, N, K]
    output = torch.zeros(M, N, device='cuda', dtype=torch.float32)  # 输出张量，shape: [M, N]

    assert K % block_k == 0, ""
    assert M % block_m == 0, ""
    assert N % block_n == 0, ""

    grid = (M // block_m, N // block_n)

    # 调用 Triton Kernel
    batched_vecmat_kernel[grid](
        A,
        B,
        M, N, K,
        output,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages
    )

    return output



##################################################################################################################################################


# Function 3: Test the correctness of the Triton kernel against the reference implementation
def test_vecmat():
    M, N, K = 128, 128, 128
    block_m, block_n, block_k = 16, 32, 64

    results = {}
    output = batched_vecmat(M, N, K, block_m, block_n, block_k)
    results['test_case_1'] = output.clone()  # Store first result

    output2 = batched_vecmat(M, N, K, block_m, block_n, block_k, num_warps=2, num_stages=2)
    results['test_case_2'] = output2.clone()  # Store second result with different key

    return results

# Run the test
result_gold = test_vecmat()
