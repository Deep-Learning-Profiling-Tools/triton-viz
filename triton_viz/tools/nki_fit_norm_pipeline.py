"""Freeze reduce-rsqrt-broadcast NC completion from independent controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KIND_TO_STRUCTURE = {
    "two_pass_reduce_multiply": "one_reduce_rsqrt_broadcast_multiply",
    "two_pass_reduce_affine": "two_reduce_rsqrt_broadcast_affine",
}


def _load(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            structure = KIND_TO_STRUCTURE.get(row.get("kind", ""))
            if structure:
                rows.append(
                    {
                        "dtype": row["dtype"],
                        "structure": structure,
                        "partition_count": int(row["p"]),
                        "free_dim": int(row["f"]),
                        "broadcast_instances": (
                            (2 if structure == "two_reduce_rsqrt_broadcast_affine" else 1)
                            * ((int(row["f"]) + 2047) // 2048)
                        ),
                        "nc_completion_ns": float(row["hardware_nc_p50_us"]) * 1000,
                    }
                )
    return rows


def _predict(train: list[dict[str, object]], sample: dict[str, object]) -> float | None:
    free_dim = int(sample["free_dim"])
    regime = 1 if free_dim <= 2048 else 2
    candidates = sorted(
        (int(row["free_dim"]), float(row["nc_completion_ns"]))
        for row in train
        if row["dtype"] == sample["dtype"]
        and row["structure"] == sample["structure"]
        and row["partition_count"] == sample["partition_count"]
        and row["broadcast_instances"] == sample["broadcast_instances"]
        and (1 if int(row["free_dim"]) <= 2048 else 2) == regime
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]
    upper_index = next((i for i, row in enumerate(candidates) if row[0] >= free_dim), len(candidates))
    if upper_index == 0:
        lower, upper = candidates[0], candidates[1]
    elif upper_index == len(candidates):
        lower, upper = candidates[-2], candidates[-1]
    else:
        lower, upper = candidates[upper_index - 1], candidates[upper_index]
    weight = (free_dim - lower[0]) / (upper[0] - lower[0])
    return lower[1] + weight * (upper[1] - lower[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--artifact-role", required=True, choices=("control", "target"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-output", required=True, type=Path)
    parser.add_argument("--max-mean-mape", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.artifact_role != "control":
        raise SystemExit("Refusing target artifacts in norm pipeline fit")
    suites = {f"{path.parent.name}/{path.name}": _load(path) for path in args.inputs}
    if len(suites) < 2 or any(not rows for rows in suites.values()):
        raise ValueError("Strict norm CV requires at least two independent suites")
    folds = []
    for held_name, test in suites.items():
        train = [row for name, rows in suites.items() if name != held_name for row in rows]
        pairs = [(_predict(train, row), float(row["nc_completion_ns"])) for row in test]
        covered = [(prediction, actual) for prediction, actual in pairs if prediction is not None]
        mape = 100 * sum(abs(prediction - actual) / actual for prediction, actual in covered) / len(covered)
        folds.append(
            {
                "held_suite": held_name,
                "samples": len(test),
                "covered_samples": len(covered),
                "coverage_pct": 100 * len(covered) / len(test),
                "nc_mape_pct": mape,
            }
        )
    mean_mape = sum(row["nc_mape_pct"] for row in folds) / len(folds)
    passed = all(row["coverage_pct"] == 100 for row in folds) and mean_mape < args.max_mean_mape
    report = {
        "schema": "triton-viz.norm-pipeline-control-cv-v1",
        "protocol": "leave-one-independent-suite-out; physical width regimes <=2048 and >2048",
        "metric": "NC-p50 MAPE",
        "folds": folds,
        "mean_nc_mape_pct": mean_mape,
        "gate_pct": args.max_mean_mape,
        "passed": passed,
        "target_postcompile_prediction_reads": False,
    }
    frozen = [row for rows in suites.values() for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dtype", "structure", "partition_count", "broadcast_instances",
        "free_dim", "nc_completion_ns",
    )
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(frozen, key=lambda row: tuple(row[field] for field in fields[:-1])))
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
