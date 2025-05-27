import triton
import triton.language as tl
import torch


def test_vec_add():
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        output = tl.zeros(x.shape, dtype=x.dtype) + x + y
        tl.store(output_ptr + offsets, output)

    access_size = 24
    size = 17
    BLOCK_SIZE = 8
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    output = torch.empty_like(a, device='cuda')
    grid = lambda meta: (triton.cdiv(access_size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, output, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    test_vec_add()
    print("Test passed!")