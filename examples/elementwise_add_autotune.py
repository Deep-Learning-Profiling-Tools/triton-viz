import triton
import triton.language as tl
import torch
import triton_viz
from triton_viz.clients import Sanitizer


# Custom kernel for element-wise addition with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements'],  # Key for selecting the optimal config based on input size
    warmup=5,
    rep=5,
)
@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def elementwise_add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Compute the index for this program (block of threads)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Prevent out-of-bounds access
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform element-wise addition
    output = x + y

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

# Create PyTorch tensors as input
n_elements = 100000
x = torch.randn(n_elements, device='cuda')
y = torch.randn(n_elements, device='cuda')
output = torch.empty_like(x)

# Launch the Triton kernel
grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
elementwise_add_kernel[grid](x, y, output, n_elements)

assert torch.allclose(output, x + y)
