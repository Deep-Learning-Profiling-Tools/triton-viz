"""Plot HBM->SBUF DMA throughput versus free bytes per SBUF partition."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DMA_ENGINES = 16
PEAK_BYTES_PER_NS_PER_ENGINE = 17.0
PEAK_AGGREGATE_GBPS = DMA_ENGINES * PEAK_BYTES_PER_NS_PER_ENGINE


def _format_bytes(value: float) -> str:
    if value < 1024:
        return str(int(value))
    kib = value / 1024
    return f"{kib:g}K"


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def load_points(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    points_by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("row_type") != "benchmark" or row.get("status") != "ok":
            continue
        if row.get("spec.name") != "dma_free_dimension" or row.get("mode") != "hbm_to_sbuf_stream":
            continue
        free_bytes = _float(row, "work.free_bytes_per_partition")
        bandwidth = _float(row, "derived.read_gbps_dma_active")
        if free_bytes is None or bandwidth is None:
            continue
        run_id = row.get("run_id", "")
        points_by_run.setdefault(run_id, []).append({
            "free_bytes_per_partition": free_bytes,
            "aggregate_gbps": bandwidth,
            "per_engine_bytes_per_ns": bandwidth / DMA_ENGINES,
            "run_id": row.get("run_id", ""),
        })
    if not points_by_run:
        return []
    # A CSV may cover several historical runs. Plot one coherent sweep: prefer
    # the run with the most points, then the lexicographically latest run ID.
    selected = max(points_by_run, key=lambda run_id: (len(points_by_run[run_id]), run_id))
    return sorted(points_by_run[selected], key=lambda point: point["free_bytes_per_partition"])


def plot_csv(csv_path: Path, output: Path) -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError("matplotlib is required to render the DMA curve") from exc

    points = load_points(csv_path)
    if not points:
        raise ValueError(f"No dma_free_dimension rows with Explorer timing found in {csv_path}")
    xs = [point["free_bytes_per_partition"] for point in points]
    ys = [point["aggregate_gbps"] for point in points]

    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.plot(xs, ys, marker="o", linewidth=2, label="Inf2 measured HBM→SBUF payload")
    axis.axhline(PEAK_AGGREGATE_GBPS, color="black", linestyle="--", linewidth=1.2,
                 label="NCv2 theoretical: 16 × 17 B/ns = 272 GB/s")
    axis.axvline(2048, color="#e69f00", linestyle=":", label="2 KiB recommended minimum")
    axis.axvline(4096, color="#009e73", linestyle=":", label="4 KiB near-saturation target")
    axis.set_xscale("log", base=2)
    axis.set_xticks(xs)
    axis.set_xticklabels([_format_bytes(value) for value in xs], rotation=35)
    axis.set_xlabel("Free bytes per partition (p=128)")
    axis.set_ylabel("Aggregate read bandwidth (GB/s)")
    axis.set_ylim(bottom=0)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)

    secondary = axis.secondary_yaxis(
        "right", functions=(lambda value: value / DMA_ENGINES, lambda value: value * DMA_ENGINES)
    )
    secondary.set_ylabel("Average bandwidth per DMA engine (B/ns)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="all_results.csv produced by the suite")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.csv.with_name("dma_free_dimension.png")
    points = plot_csv(args.csv, output)
    print(f"Plotted {len(points)} points to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
