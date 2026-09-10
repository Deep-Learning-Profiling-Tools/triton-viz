import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(c_ptr, a_ptr, b_ptr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M, N, K = 4096, 4096, 4096
    stride_am = 4096
    stride_ak = 1
    stride_bk = 4096
    stride_bn = 1
    stride_cm = 4096
    stride_cn = 1

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    matmul_kernel[triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)](
        c, a, b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )



##################################################################################################################################################


def test_matmul():
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 64
    M, N, K = 4096, 4096, 4096

    # Initialize matrices on CUDA device
    c = torch.empty((M, N), device='cuda:0', dtype=torch.float16)
    a = torch.rand((M, K), device='cuda:0', dtype=torch.float16)
    b = torch.rand((K, N), device='cuda:0', dtype=torch.float16)

    # Call the matmul function multiple times
    test_case_1 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # Additional test cases to cover more branches
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 64, 128
    test_case_2 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 256, 64
    test_case_3 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    test_case_4 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    return {
        "test_case_1": test_case_1,
        "test_case_2": test_case_2,
        "test_case_3": test_case_3,
        "test_case_4": test_case_4
    }

result_gold = test_matmul()
