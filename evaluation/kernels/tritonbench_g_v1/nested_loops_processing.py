import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j in range(0, 2):
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)

            for k in range(0, 2):
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n

                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n

        a_ptrs += 2 * stride_n


def wrapper_nested3(n_rows, n_cols):
    x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)
    nested3[grid](x, output, x.stride(0), x.stride(1))
    print(output)




##################################################################################################################################################


import torch

def test_nested3():
    # Test dimensions
    results = {}
    
    # Test case 1
    n_rows = 8
    n_cols = 8
    results['test_case_1'] = wrapper_nested3(n_rows, n_cols)
    
    # Test case 2
    n_rows = 4
    n_cols = 4
    results['test_case_2'] = wrapper_nested3(n_rows, n_cols)
    
    # Test case 3
    n_rows = 16
    n_cols = 16
    results['test_case_3'] = wrapper_nested3(n_rows, n_cols)
    
    # Test case 4
    n_rows = 2
    n_cols = 2
    results['test_case_4'] = wrapper_nested3(n_rows, n_cols)
    
    return results

result_gold = test_nested3()
