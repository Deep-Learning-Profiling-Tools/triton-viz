import os
import sys
import torch
import triton
from triton_viz.core import config as cfg
from triton_viz.core.trace import launches
from triton_viz.core.data import Load, Store, Dot

# Ensure project root is on sys.path so we can import examples
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.load_store import simple_kernel


def main():
    # Reset viz config to a clean state
    cfg.reset()

    # Prepare sample data and run the kernel once
    device = "cpu"
    size = 16
    BLOCK_SIZE = 8
    torch.manual_seed(0)
    x = torch.arange(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    # Run the traced kernel (decorated in examples/load_store.py)
    simple_kernel[grid](x, output, size, BLOCK_SIZE)

    # Inspect the last launch and print a sample op's call_path
    print(f"Number of launches: {len(launches)}")
    if not launches:
        return

    launch = launches[-1]
    print(f"Number of records: {len(launch.records)}")

    target = None
    for rec in launch.records:
        if isinstance(rec, (Load, Store, Dot)):
            target = rec
            break

    if target is None:
        print("No Load/Store/Dot record found to display call_path.")
        return

    print(f"=== {type(target).__name__} call_path ===")
    for i, f in enumerate(target.call_path):
        line = f.line if hasattr(f, "line") else ""
        print(f"{i:02d}  {f.filename}:{f.lineno}  [{f.name}]  |  {line}")


if __name__ == "__main__":
    main()
