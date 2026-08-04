"""Plot the DMA-transpose bandwidth surface."""

from __future__ import annotations

import argparse
from pathlib import Path

from microbench.inf2_nki.profile_parser.plot_dma_partition_surface import plot_surface


def plot_transpose_surface(csv_path: Path, output: Path):
    return plot_surface(
        csv_path,
        output,
        benchmark_name="dma_transpose_surface",
        title="DMA transpose",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.csv.with_name("dma_transpose_surface.png")
    points = plot_transpose_surface(args.csv, output)
    print(f"Plotted {len(points)} points to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
