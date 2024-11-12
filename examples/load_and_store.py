import triton
import triton.language as tl
import torch

import triton_viz
from triton_viz.clients import FakeExecutor

@triton_viz.trace(clients=FakeExecutor())
@triton.jit
def simple_kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X_ptr + idx)

    tl.store(Y_ptr + idx, x)

if __name__ == '__main__':
    BLOCK_SIZE = 1024
    n_elements = 512

    # Create input and output tensors
    X = torch.arange(n_elements, dtype=torch.float32, device='cuda')
    Y = torch.empty_like(X, device='cuda')

    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    simple_kernel[grid](X, Y, BLOCK_SIZE=BLOCK_SIZE)
 
    # Verify the results
    print("Input tensor X:", X)
    print("Output tensor Y:", Y)
