import torch
import triton
import triton.language as tl
import triton_viz


BLOCK = 1024
SHAPE = (2, 3, 4, 3, 4, 2, 4, 2, 3)


@triton_viz.trace("tracer")
@triton.jit
def copy_9d_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Copy a rank-9 tensor via linearized offsets."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values, mask=mask)


def run_demo() -> None:
    """Run a rank-9 copy example and launch Triton Viz."""
    torch.manual_seed(0)
    x = torch.randn(SHAPE, device="cpu")
    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = (triton.cdiv(n_elements, BLOCK),)
    copy_9d_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK=BLOCK,
    )

    assert torch.allclose(y, x), "copy kernel output mismatch"
    triton_viz.launch(share=False, port=5001)


if __name__ == "__main__":
    run_demo()
