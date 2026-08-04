"""Plot pointwise DMA-transpose bandwidth relative to DMA copy."""

from __future__ import annotations

import argparse
from pathlib import Path

from microbench.inf2_nki.profile_parser.plot_dma_partition_surface import load_surface


def plot_comparison(copy_csv: Path, transpose_csv: Path, output: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    copy_points = load_surface(copy_csv)
    transpose_points = load_surface(transpose_csv, benchmark_name="dma_transpose_surface")
    copies = {(point["partitions"], point["free_bytes"]): point["bandwidth_gbps"] for point in copy_points}
    transposes = {(point["partitions"], point["free_bytes"]): point["bandwidth_gbps"] for point in transpose_points}
    keys = sorted(copies.keys() & transposes.keys())
    if not keys:
        raise ValueError("No matching copy and transpose surface points")
    partitions = sorted({key[0] for key in keys})
    free_bytes = sorted({key[1] for key in keys})
    ratios = np.array([[transposes[p, f] / copies[p, f] for f in free_bytes] for p in partitions])

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = axis.imshow(ratios, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1.2)
    axis.set_xticks(range(len(free_bytes)), [f"{f/1024:g}K" if f >= 1024 else str(int(f)) for f in free_bytes], rotation=35)
    axis.set_yticks(range(len(partitions)), [str(p) for p in partitions])
    axis.set_xlabel("Free bytes / destination SBUF partition")
    axis.set_ylabel("Destination SBUF partitions")
    axis.set_title("DMA transpose / DMA copy payload bandwidth")
    fig.colorbar(image, ax=axis, label="Bandwidth ratio")
    for row, p in enumerate(partitions):
        for column, _ in enumerate(free_bytes):
            value = ratios[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                      color="black" if 0.35 < value < 1.05 else "white")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return ratios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("copy_csv", type=Path)
    parser.add_argument("transpose_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ratios = plot_comparison(args.copy_csv, args.transpose_csv, args.output)
    print(f"Plotted {ratios.size} matched points to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
