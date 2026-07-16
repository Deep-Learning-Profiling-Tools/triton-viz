import torch
import triton
import triton.language as tl
from torch.testing import assert_close


@triton.jit
def load_reduce_kernel(
    x_ptr,      # pointer to the input matrix
    y_ptr,      # pointer to the output vector
    stride_xm,  # stride of matrix x in leading dimension
    stride_xn,  # stride of matrix x in the second dimension
    stride_y,   # stride of output vector y
    BLOCK_M: tl.constexpr,  # block size in leading dimension
    BLOCK_N: tl.constexpr,  # block size in second dimension
):
    x_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)

# Test function for load_reduce_kernel
def load_reduce(BLOCK_M, BLOCK_N, dtype_str):
    dtype_mapping = {
        'float16': torch.float16,
        'float32': torch.float32,
    }
    dtype = dtype_mapping[dtype_str]
    x = torch.randn((BLOCK_M, BLOCK_N), device='cuda', dtype=dtype)
    y = torch.empty((BLOCK_M,), device='cuda', dtype=dtype)

    load_reduce_kernel[(1,)](x, y, x.stride(0), x.stride(1), y.stride(0), BLOCK_M, BLOCK_N)

    golden = x.max(dim=1)[0]
    torch.set_printoptions(profile='full')
    assert_close(y, golden, rtol=1e-2, atol=1e-3, check_dtype=False)



##################################################################################################################################################


import torch

def test_reduce():
    # 测试参数设置
    test_cases = [
        {"BLOCK_M": 16, "BLOCK_N": 16, "dtype_str": "float16"},
        {"BLOCK_M": 32, "BLOCK_N": 32, "dtype_str": "float16"},
        {"BLOCK_M": 64, "BLOCK_N": 64, "dtype_str": "float32"},
        {"BLOCK_M": 128, "BLOCK_N": 128, "dtype_str": "float32"},
    ]

    results = {}
    for i, case in enumerate(test_cases):
        BLOCK_M = case["BLOCK_M"]
        BLOCK_N = case["BLOCK_N"]
        dtype_str = case["dtype_str"]

        try:
            load_reduce(BLOCK_M, BLOCK_N, dtype_str)
            results[f"test_case_{i+1}"] = "passed"
        except Exception as e:
            results[f"test_case_{i+1}"] = f"failed: {e}"

    return results

result_gold = test_reduce()
