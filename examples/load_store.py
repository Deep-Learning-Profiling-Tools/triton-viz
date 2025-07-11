import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core import config as cfg
from triton_viz.core.trace import launches


@triton_viz.trace(clients=Tracer())
@triton.jit
def simple_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


if __name__ == "__main__":
    cfg.reset()
    device = "cpu"
    size = 16
    BLOCK_SIZE = 8
    torch.manual_seed(0)
    x = torch.arange(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    simple_kernel[grid](x, output, size, BLOCK_SIZE)

    # Print records to see what's being captured
    print(f"Number of launches: {len(launches)}")
    if launches:
        launch = launches[-1]
        print(f"Number of records: {len(launch.records)}")
        for i, record in enumerate(launch.records):
            print(f"Record {i}: {type(record).__name__}")
            if hasattr(record, "ptr"):
                print(f"  ptr: {record.ptr}")
            if hasattr(record, "offsets"):
                print(f"  offsets shape: {record.offsets.shape}")
            if hasattr(record, "masks"):
                print(f"  masks shape: {record.masks.shape}")

    # Try to launch visualization
    try:
        triton_viz.launch()
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback

        traceback.print_exc()
