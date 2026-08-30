"""Freeze the single global NC completion model from independent controls.

The model has no structural key.  For every program, whatever its grammar,
end-to-end completion is bounded below by

    ``max(engine_busy) + beta * (sum(engine_busy) - max(engine_busy)) + offset``

where ``beta`` is one global fraction of non-critical engine work that the
compiler does *not* overlap with the critical engine, and ``offset`` is one
global fixed launch/drain cost.  Both are measured on independent control
programs as ``NC-p50 minus measured engine active time`` and are validated by
leave-one-entire-free-dimension-out CV before they may enter production.

Setting ``beta = 1`` reduces the model to the strict single-constant form
``sum(engine_busy) + offset``; ``--fixed-beta`` selects that variant.

A pipeline-depth refinement, in which each engine's residue is discounted by
how many independent work items the source DAG gives it, was fitted and
cross-validated against a control set spanning both this serial whole-program
family and a software-pipelined tiled-Dot family.  It won that CV decisively
(10.22% versus 13.99% leave-one-suite-out) and still failed to improve the
authoritative 254-point target metric, so it was rejected rather than promoted.
Its frozen candidate and CV report are retained under
``diagnostics/rejected_pipeline_overlap_v1/``.

This tool refuses target artifacts.  Control Explorer summaries are aggregate
compiler labels for control programs, which the project's calibration boundary
permits.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ENGINES = ("vector", "scalar", "gpsimd", "tensor")


def _control_samples(root: Path) -> list[dict[str, float]]:
    rows = []
    with (root / "operator_results.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok":
                continue
            case = f"{row['op']}__r{row['rows']}__c{row['cols']}__{row['dtype']}"
            summary = root / case / "hardware" / "explorer_summary.json"
            if not summary.is_file():
                continue
            profile = next(
                iter(json.loads(summary.read_text(encoding="utf-8")).values()), {}
            )
            busy = [
                float(profile.get(f"{engine}_engine_active_time") or 0.0) * 1e9
                for engine in ENGINES
            ]
            busy.append(float(profile.get("dma_active_time") or 0.0) * 1e9)
            rows.append(
                {
                    "case": case,
                    "free_dim": int(row["cols"]),
                    "max_busy_ns": max(busy),
                    "sum_busy_ns": sum(busy),
                    "completion_ns": float(row["hardware_nc_p50_us"]) * 1000.0,
                }
            )
    return rows


def _base_ns(sample: dict[str, float], beta: float) -> float:
    return sample["max_busy_ns"] + beta * (
        sample["sum_busy_ns"] - sample["max_busy_ns"]
    )


def _offset_ns(samples: list[dict[str, float]], beta: float) -> float:
    """Least-absolute-deviation constant: the median measured non-busy time."""
    return statistics.median(
        sample["completion_ns"] - _base_ns(sample, beta) for sample in samples
    )


def _mape_pct(samples: list[dict[str, float]], beta: float, offset: float) -> float:
    return (
        100.0
        * sum(
            abs(_base_ns(sample, beta) + offset - sample["completion_ns"])
            / sample["completion_ns"]
            for sample in samples
        )
        / len(samples)
    )


def _fit_beta(samples: list[dict[str, float]], fixed_beta: float | None) -> float:
    if fixed_beta is not None:
        return fixed_beta
    grid = [index / 100.0 for index in range(0, 101)]
    return min(grid, key=lambda beta: _mape_pct(samples, beta, _offset_ns(samples, beta)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_roots", nargs="+", type=Path)
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    parser.add_argument("--max-mean-mape", type=float, default=20.0)
    parser.add_argument(
        "--fixed-beta",
        type=float,
        help="Pin the overlap fraction instead of fitting it; 1.0 gives the "
        "strict single-constant sum(engine_busy)+offset model.",
    )
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in global completion fit")

    samples = [
        sample for root in args.control_roots for sample in _control_samples(root)
    ]
    if len(samples) < 2:
        raise ValueError("Global completion fit needs independent control programs")

    folds = []
    for held in sorted({sample["free_dim"] for sample in samples}):
        train = [sample for sample in samples if sample["free_dim"] != held]
        test = [sample for sample in samples if sample["free_dim"] == held]
        beta = _fit_beta(train, args.fixed_beta)
        offset = _offset_ns(train, beta)
        folds.append(
            {
                "held_free_dim": held,
                "samples": len(test),
                "fold_beta": beta,
                "fold_offset_ns": offset,
                "nc_mape_pct": _mape_pct(test, beta, offset),
            }
        )
    mean_mape = sum(fold["nc_mape_pct"] for fold in folds) / len(folds)
    beta = _fit_beta(samples, args.fixed_beta)
    offset = _offset_ns(samples, beta)
    passed = mean_mape < args.max_mean_mape
    report = {
        "schema": "triton-viz.global-completion-control-cv-v1",
        "protocol": "leave-one-entire-free-dimension-out; one global (beta, offset) pair, no structural key",
        "metric": "NC-p50 MAPE",
        "control_samples": len(samples),
        "folds": folds,
        "mean_nc_mape_pct": mean_mape,
        "frozen_beta": beta,
        "frozen_offset_ns": offset,
        "gate_pct": args.max_mean_mape,
        "passed": passed,
        "target_postcompile_prediction_reads": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "overlap_fraction",
                "completion_offset_ns",
                "control_samples",
                "control_cv_mean_mape_pct",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "overlap_fraction": beta,
                "completion_offset_ns": offset,
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
