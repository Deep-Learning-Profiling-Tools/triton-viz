import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer


@triton_viz.trace(clients=Tracer())
@triton.jit
def histogram_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Generate a wider range of outputs to showcase histograms
    perturbed = tl.sin(values) * 0.5 + values * 0.3
    tl.store(y_ptr + offsets, perturbed, mask=mask)


def run_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 131072
    BLOCK = 1024
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float32, device=device) * 3.0
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    histogram_kernel[grid](x, y, n, BLOCK)

    print("Histogram demo kernel executed.")
    print("Open the visualization, click any Load/Store op,")
    print("then use the 'Value Histogram' button to inspect distributions.")


if __name__ == "__main__":
    run_demo()
    try:
        triton_viz.launch(share=False)
    except Exception as exc:
        print("Failed to launch visualization:", exc)

