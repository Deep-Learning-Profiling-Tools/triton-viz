"""Plot the DMA bandwidth surface over active partitions and free bytes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def load_surface(csv_path: Path, benchmark_name: str = "dma_partition_surface") -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("row_type") != "benchmark" or row.get("status") != "ok":
            continue
        if row.get("spec.name") != benchmark_name:
            continue
        try:
            point = {
                "partitions": int(float(row["work.partition_count"])),
                "free_bytes": float(row["work.free_bytes_per_partition"]),
                "engines": min(16, int(float(row["work.partition_count"]))),
                "bandwidth_gbps": float(row["derived.read_gbps_dma_active"]),
                "run_id": row.get("run_id", ""),
            }
        except (KeyError, TypeError, ValueError):
            continue
        point["engine_utilization"] = point["bandwidth_gbps"] / (point["engines"] * 17.0)
        by_run.setdefault(point["run_id"], []).append(point)
    if not by_run:
        return []
    selected = max(by_run, key=lambda run_id: (len(by_run[run_id]), run_id))
    return sorted(by_run[selected], key=lambda point: (point["partitions"], point["free_bytes"]))


def plot_surface(csv_path: Path, output: Path, benchmark_name: str = "dma_partition_surface",
                 title: str = "DMA copy") -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required to render the DMA surface") from exc
    points = load_surface(csv_path, benchmark_name=benchmark_name)
    if not points:
        raise ValueError(f"No dma_partition_surface rows found in {csv_path}")
    partitions = sorted({point["partitions"] for point in points})
    free_bytes = sorted({point["free_bytes"] for point in points})
    lookup = {(point["partitions"], point["free_bytes"]): point for point in points}
    z_bandwidth = np.array([[lookup[(p, f)]["bandwidth_gbps"] for f in free_bytes] for p in partitions])
    z_utilization = np.array([[lookup[(p, f)]["engine_utilization"] for f in free_bytes] for p in partitions])
    x, y = np.meshgrid(np.log2(free_bytes), np.log2(partitions))

    fig = plt.figure(figsize=(16, 7))
    surface_axis = fig.add_subplot(1, 2, 1, projection="3d")
    surface = surface_axis.plot_surface(x, y, z_bandwidth, cmap="viridis", edgecolor="none", alpha=0.92)
    surface_axis.set_xlabel("Free bytes / partition")
    surface_axis.set_ylabel("Active partitions")
    surface_axis.set_zlabel(f"{title} read bandwidth (GB/s)")
    surface_axis.set_xticks(np.log2(free_bytes), [f"{f/1024:g}K" if f >= 1024 else str(int(f)) for f in free_bytes])
    surface_axis.set_yticks(np.log2(partitions), [str(p) for p in partitions])
    fig.colorbar(surface, ax=surface_axis, shrink=0.65, pad=0.12, label="GB/s")

    heat_axis = fig.add_subplot(1, 2, 2)
    heat = heat_axis.imshow(z_utilization, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    heat_axis.set_xticks(range(len(free_bytes)), [f"{f/1024:g}K" if f >= 1024 else str(int(f)) for f in free_bytes], rotation=35)
    heat_axis.set_yticks(range(len(partitions)), [str(p) for p in partitions])
    heat_axis.set_xlabel("Free bytes / partition")
    heat_axis.set_ylabel("Active partitions")
    heat_axis.set_title(f"{title}: utilization vs min(partitions,16) × 17 GB/s")
    fig.colorbar(heat, ax=heat_axis, label="DMA-engine utilization")
    fig.subplots_adjust(left=0.04, right=0.95, bottom=0.15, top=0.95, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.csv.with_name("dma_partition_surface.png")
    points = plot_surface(args.csv, output)
    print(f"Plotted {len(points)} points to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
