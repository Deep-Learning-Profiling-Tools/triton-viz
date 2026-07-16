import torch
import triton
import triton.language as tl


@triton.jit
def iv_dependent_matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                                M, N, K,  #
                                stride_am, stride_ak,  #
                                stride_bk, stride_bn,  #
                                stride_cm, stride_cn,  #
                                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                                type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_ptrs = a_ptr
    b_ptrs = b_ptr
    if type == "post_load_two_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
    elif type == "post_load_three_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
        a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
        b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if type == "pre_load":
            a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + k * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        if type == "post_load":
            a_ptrs = a_ptr + (k + 1) * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_two_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptr + (k + 2) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + (k + 2) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_three_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptrs_next_next
            b_ptrs_next = b_ptrs_next_next
            a_ptrs_next_next = a_ptr + (k + 3) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next_next = b_ptr + (k + 3) * BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def iv_dependent_matmul_wrapper(
    M: int, 
    K: int, 
    N: int, 
    BLOCK_SIZE_M: int, 
    BLOCK_SIZE_N: int, 
    BLOCK_SIZE_K: int, 
    type: str = "pre_load",  # Kernel type for scheduling ("pre_load", "post_load", etc.)
    device: torch.device = "cuda"  # Device to run the test (defaults to "cuda")
):
    # Ensure the device is correct
    device = torch.device(device)

    # Generate random input matrices a and b on the specified device
    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    # Create an empty tensor to store the Triton result
    triton_output = torch.empty((M, N), device=device)

    # Define Triton grid configuration
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # Set the number of stages based on the kernel type
    num_stages = 4 if type == "post_load_three_iters" else 3

    # Run the Triton kernel
    iv_dependent_matmul_kernel[grid](
        a, b, triton_output, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        triton_output.stride(0), triton_output.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, type=type,  #
        num_stages=num_stages
    )

    # Optionally print the result for inspection
    # print(triton_output)

    return triton_output




##################################################################################################################################################


import torch

# 封装 IV Dependent MatMul 测试的函数
def test_iv_dependent_matmul_kernel():
    # 定义矩阵维度和块大小
    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # 创建 CUDA 设备
    device = torch.device('cuda')

    # 定义所有类型的内核配置
    types = [
        "pre_load",
        "post_load",
        "post_pre_mixed",
        "post_load_two_iters",
        "post_load_three_iters"
    ]

    # 字典用于存储每个测试用例的结果
    results = {}

    # 遍历每种内核类型，进行测试
    for i, type in enumerate(types):
        # 调用封装函数运行 Triton 核心
        triton_output = iv_dependent_matmul_wrapper(
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            type=type,
            device=device
        )

        # 确保输出的大小正确
        assert triton_output.shape == (M, N), f"Expected output shape {(M, N)} but got {triton_output.shape} for type {type}"

        # 保存结果到字典
        results[f"test_case_{i+1}"] = triton_output

    return results

# 执行测试函数进行所有类型的验证
result_gold = test_iv_dependent_matmul_kernel()
