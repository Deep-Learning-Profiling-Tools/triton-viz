import cairocffi
cairocffi.install_as_pycairo()

import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core import config as cfg
from triton_viz.core.trace import launches
from triton_viz.visualizer import draw
import os


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
    
    # Manual visualization to bypass interface issues
    if launches:
        launch = launches[-1]
        print(f"Number of records: {len(launch.records)}")
        
        # Collect and visualize
        try:
            program_records, tensor_table, failures = draw.collect_grid()
            print(f"Grid records collected: {list(program_records.keys())}")
            
            # Try to draw for grid (0, 0, 0)
            if (0, 0, 0) in program_records:
                records = program_records[(0, 0, 0)]
                print(f"Number of records for grid (0,0,0): {len(records)}")
                for i, record in enumerate(records):
                    print(f"  Record {i}: {type(record).__name__}")
                
                # Visualize
                output_file = "test_visualization.png"
                w, h = draw.draw_record(records, tensor_table, output_file)
                print(f"Visualization saved to {output_file} with dimensions {w}x{h}")
                print(f"File exists: {os.path.exists(output_file)}")
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()