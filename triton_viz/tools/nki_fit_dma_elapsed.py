"""Freeze the global DMA descriptor-issue interval from independent controls.

A DMA transfer's bandwidth model prices bytes; the DMA queue is occupied per
descriptor.  A contiguous run coalesces into one descriptor per partition,
while a fragmented free axis costs one descriptor per element.  The
independent strided-store controls isolate the second regime: their elapsed
time tracks descriptor count, not bytes, and exceeds measured DMA active time
by up to 12x.

The fitted quantity is a single global issue interval ``ns_per_descriptor``
in the physical relation

    ``elapsed_ns = max(active_ns, descriptor_count * ns_per_descriptor)``

which is validated against control NC-p50 as
``elapsed_ns + global completion offset`` under strict
leave-one-entire-free-dimension-out CV.  It is a rate, not a table: nothing
here is keyed by kernel structure, operator or grammar.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _control_samples(results_jsonl: Path) -> list[dict[str, float]]:
    rows = []
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        if result.get("status") != "ok":
            continue
        if result.get("spec", {}).get("kind") != "dma_strided_store":
            continue
        case = Path(result["dir"])
        summary = case / "explorer_summary.json"
        if not summary.is_file():
            continue
        profile = next(
            iter(json.loads(summary.read_text(encoding="utf-8")).values()), {}
        )
        spec = result["spec"]
        rows.append(
            {
                "case": case.name,
                "dtype": spec["dtype"],
                "free_dim": int(spec["f"]),
                # ``strided_store_pair_factory`` issues *two* stride-``stride``
                # stores of p*f elements into one interleaved output, so the
                # control's fragmented-descriptor count is 2*p*f.
                "descriptors": 2 * int(spec["p"]) * int(spec["f"]),
                "active_ns": float(profile.get("dma_active_time") or 0.0) * 1e9,
                "nc_ns": float(
                    result["latency_percentiles"]["nc_latency"]["p50_us"]
                )
                * 1000.0,
            }
        )
    return rows


def _predict_ns(sample: dict[str, float], rate: float, offset_ns: float) -> float:
    return max(sample["active_ns"], sample["descriptors"] * rate) + offset_ns


def _mape_pct(
    samples: list[dict[str, float]], rate: float, offset_ns: float
) -> float:
    return (
        100.0
        * sum(
            abs(_predict_ns(sample, rate, offset_ns) - sample["nc_ns"])
            / sample["nc_ns"]
            for sample in samples
        )
        / len(samples)
    )


def _fit_rate(samples: list[dict[str, float]], offset_ns: float) -> float:
    """Least-absolute-relative-error issue interval over a 0.05ns grid."""
    grid = [index / 20.0 for index in range(20, 1001)]
    return min(grid, key=lambda rate: _mape_pct(samples, rate, offset_ns))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", nargs="+", type=Path)
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument(
        "--global-completion-csv",
        required=True,
        type=Path,
        help="Frozen global completion surface; its offset is the launch cost "
        "subtracted before the descriptor rate is fitted.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    parser.add_argument("--max-mean-mape", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in DMA elapsed fit")

    with args.global_completion_csv.open(encoding="utf-8", newline="") as file:
        offset_ns = float(next(iter(csv.DictReader(file)))["completion_offset_ns"])

    samples = [
        sample for path in args.results_jsonl for sample in _control_samples(path)
    ]
    if len(samples) < 2:
        raise ValueError("DMA elapsed fit needs independent strided controls")

    folds = []
    for held in sorted({sample["free_dim"] for sample in samples}):
        train = [sample for sample in samples if sample["free_dim"] != held]
        test = [sample for sample in samples if sample["free_dim"] == held]
        rate = _fit_rate(train, offset_ns)
        folds.append(
            {
                "held_free_dim": held,
                "samples": len(test),
                "fold_ns_per_descriptor": rate,
                "nc_mape_pct": _mape_pct(test, rate, offset_ns),
            }
        )
    mean_mape = sum(fold["nc_mape_pct"] for fold in folds) / len(folds)
    rate = _fit_rate(samples, offset_ns)
    passed = mean_mape < args.max_mean_mape
    descriptors = [sample["descriptors"] for sample in samples]
    report = {
        "schema": "triton-viz.dma-elapsed-control-cv-v1",
        "protocol": "leave-one-entire-free-dimension-out; one global descriptor issue interval",
        "model": "elapsed_ns = max(active_ns, descriptor_count * ns_per_descriptor)",
        "metric": "NC-p50 MAPE",
        "control_samples": len(samples),
        "completion_offset_ns": offset_ns,
        "folds": folds,
        "mean_nc_mape_pct": mean_mape,
        "frozen_ns_per_descriptor": rate,
        "measured_descriptor_domain": [min(descriptors), max(descriptors)],
        "gate_pct": args.max_mean_mape,
        "passed": passed,
        "target_postcompile_prediction_reads": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "ns_per_descriptor",
                "measured_min_descriptors",
                "measured_max_descriptors",
                "control_samples",
                "control_cv_mean_mape_pct",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "ns_per_descriptor": rate,
                "measured_min_descriptors": min(descriptors),
                "measured_max_descriptors": max(descriptors),
                "control_samples": len(samples),
                "control_cv_mean_mape_pct": mean_mape,
            }
        )
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
