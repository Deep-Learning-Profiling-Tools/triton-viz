
import triton
import triton.language as tl
import torch

@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
    tl.store(Y + offs, x, mask=mask)

def f8_to_f16(x, dtypes=tl.float8e5) -> torch.Tensor:
    assert x.dtype == torch.int8, f"torch.int8 expected but got {x.dtype}"
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"
    ret = torch.empty_like(x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = ret.untyped_storage().size() // ret.element_size()
    kernel_f8_to_f16[grid](ret, triton.reinterpret(x, dtypes), numel, BLOCK_SIZE=1024)
    return ret

@triton.jit
def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

def f16_to_f8(x: torch.Tensor, dtypes=tl.float8e5) -> torch.Tensor:
    assert x.dtype in [torch.float16, torch.float32]
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"
    ret = torch.empty_like(x, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = x.untyped_storage().size() // x.element_size()
    kernel_f16_to_f8[grid](triton.reinterpret(ret, dtypes), x, numel, BLOCK_SIZE=1024)
    return ret




##################################################################################################################################################


# Test code for f16_to_f8 and f8_to_f16 functions
def test_triton_kernels():
    results = {}
    # Test for f16_to_f8 and f8_to_f16 conversion
    for i in range(4):
        # Create a random tensor of shape (16, 128) with dtype float16 on CUDA
        a = torch.randn((16, 128), dtype=torch.float16, device="cuda")
        
        # Convert from float16 to float8 using f16_to_f8
        b = f16_to_f8(a, dtypes=tl.float8e5)
        
        # Convert back from float8 to float16 using f8_to_f16
        c = f8_to_f16(b, dtypes=tl.float8e5)
        
        # Store results
        results[f'test_case_{i+1}'] = c

    return results

# Run the test and store the result
result_gold = test_triton_kernels()
