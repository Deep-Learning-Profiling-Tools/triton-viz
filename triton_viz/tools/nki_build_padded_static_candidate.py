"""Build a frozen structural Static-DMA calibration from control-only labels."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.base_calibration.open(encoding="utf-8", newline="") as file:
        base_rows = list(csv.DictReader(file))
    # Re-open only to obtain the header cleanly; no hardware or post-compile
    # artifact is inspected by this builder.
    with args.base_calibration.open(encoding="utf-8", newline="") as file:
        base_fields = list(csv.DictReader(file).fieldnames or [])

    samples: dict[tuple[int, int, int], list[float]] = {}
    for path in args.control_results:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    2 if row["dtype"] == "bfloat16" else 4,
                    int(row["p"]),
                    int(row["f"]),
                )
                samples.setdefault(key, []).append(float(row["static_dma_active_ns"]))

    extra_fields = ["calibration_mode", "logical_partition_count"]
    fields = base_fields + [name for name in extra_fields if name not in base_fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(base_rows)
        for (element_bytes, logical_p, free_dim), values in sorted(samples.items()):
            writer.writerow(
                {
                    "calibration_mode": "padded_partition_shape",
                    "element_bytes": element_bytes,
                    "logical_partition_count": logical_p,
                    "logical_free_dim": free_dim,
                    "static_dma_ns": statistics.median(values),
                }
            )
    print(f"Wrote {args.output} with {len(samples)} padded control points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
