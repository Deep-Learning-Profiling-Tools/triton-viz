import triton
from triton import language as tl
import torch

@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b

@triton.jit
def softmax_kernel_online_v2(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = tl.full((TILE_N,), value=-float("inf"), dtype=output_ptr.dtype.element_ty)
    z = tl.full((TILE_N,), value=0, dtype=output_ptr.dtype.element_ty)
    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
        m = new_m
        z = new_z
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
        m = new_m
        z = new_z
    final_m = tl.max(m, 0)
    z = tl.sum(tl.exp(m - final_m) * z)
    m = final_m

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out)
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)

def softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = min(4096, triton.next_power_of_2(N))
    grid = (M, 1, 1)
    softmax_kernel_online_v2[grid](out, x, M, N, TILE_N)
    return out



##################################################################################################################################################


# Comparison Test
def test_softmax():

    torch.manual_seed(0)
    
    result = {}
    
    # Case 1: M = 128, N = 512
    x1 = torch.randn(128, 512, device='cuda', dtype=torch.float32)
    result['test_case_1'] = softmax(x1)

    # Case 2: M = 64, N = 1024
    x2 = torch.randn(64, 1024, device='cuda', dtype=torch.float32)
    result['test_case_2'] = softmax(x2)

    # Case 3: M = 256, N = 128
    x3 = torch.randn(256, 128, device='cuda', dtype=torch.float32)
    result['test_case_3'] = softmax(x3)
    
    return result

# Execute test function
result_gold = test_softmax()
