"""Audit structured completion-floor dependence on frozen NKI holdouts.

This tool never fits on holdout measurements.  It reloads the frozen
calibration bundle and replays saved traces while excluding one independent
completion-control dimension at a time:

* free dimension;
* partition count;
* structural grammar key;
* operator (implemented by excluding every completion key used by that
  operator, then evaluating only that operator's cases).

The output distinguishes covered-grammar reconstruction from prediction
without the structured completion floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from triton_viz.tools.nki_cost_model_pipeline import (
    _load_splits,
    _replay_args,
)
from triton_viz.tools.nki_region_ir import structural_calibration_key


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _case_keys(holdout: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for trace in holdout.glob("*/trace.jsonl"):
        keys = set()
        for line in trace.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            region = event.get("region_ir")
            if region and int(region.get("reduction_count") or 0) > 0:
                keys.add(structural_calibration_key(region))
        result[trace.parent.name] = keys
    return result


def _mape(rows: list[dict[str, str]], field: str = "nc_error_pct") -> float:
    values = [
        abs(float(row[field]))
        for row in rows
        if row.get(field) not in (None, "")
    ]
    return statistics.mean(values) if values else float("nan")


def _summary(label: str, rows: list[dict[str, str]]) -> dict:
    return {
        "audit": label,
        "cases": len(rows),
        "nc_mape_pct": _mape(rows),
        "without_completion_floor_mape_pct": _mape(
            rows, "without_completion_floor_error_pct"
        ),
        "completion_activated_cases": sum(
            int(row.get("completion_floor_activated") or 0) for row in rows
        ),
        "completion_exact_count": sum(
            int(row.get("completion_exact_count") or 0) for row in rows
        ),
        "completion_interpolated_count": sum(
            int(row.get("completion_interpolated_count") or 0) for row in rows
        ),
        "completion_ood_count": sum(
            int(row.get("completion_ood_count") or 0) for row in rows
        ),
    }


def _run_replays(
    root: Path,
    holdouts: list[tuple[str, Path, str]],
    output_dir: Path,
    label: str,
    extra_args: list[str],
) -> list[dict[str, str]]:
    result = []
    for split_name, holdout, dtype in holdouts:
        output = output_dir / f"{label}__{holdout.name}.csv"
        args = _replay_args(root, holdout, output, dtype)
        args[args.index("--output") : args.index("--output")] = [
            "--strict-calibration",
            *extra_args,
        ]
        if split_name in {"formal_fp32_v1", "full_fp32_v1", "full_bf16_v1"}:
            args[args.index("--output") : args.index("--output")] = [
                "--strided-dma-csv",
                str(root / "calibration/strided_dma.csv"),
            ]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
        result.extend(_rows(output))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "completion_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_splits()
    holdouts = []
    for split_name, split in splits.items():
        for holdout in sorted((root / "holdouts").glob(f"{split_name}*")):
            holdouts.append((split_name, holdout, split["dtype"]))

    baseline = [
        row
        for path in sorted((root / "evaluation").glob("surface_*.csv"))
        for row in _rows(path)
    ]
    report = [_summary("baseline_rows", baseline)]
    report.append(
        {
            **_summary("without_structured_completion_floor", baseline),
            "nc_mape_pct": _mape(
                baseline, "without_completion_floor_error_pct"
            ),
        }
    )

    case_keys: dict[str, set[str]] = {}
    for _split, holdout, _dtype in holdouts:
        case_keys.update(_case_keys(holdout))

    for free_dim in sorted(
        {int(row["cols"]) for row in baseline if row.get("cols")}
    ):
        rows = _run_replays(
            root,
            holdouts,
            output_dir,
            f"leave_f_{free_dim}",
            ["--completion-exclude-free-dim", str(free_dim)],
        )
        report.append(
            _summary(
                f"leave_one_F_out:{free_dim}",
                [row for row in rows if int(row["cols"]) == free_dim],
            )
        )

    for partition in sorted(
        {int(row["rows"]) for row in baseline if row.get("rows")}
    ):
        rows = _run_replays(
            root,
            holdouts,
            output_dir,
            f"leave_p_{partition}",
            ["--completion-exclude-partition", str(partition)],
        )
        report.append(
            _summary(
                f"leave_one_partition_out:{partition}",
                [row for row in rows if int(row["rows"]) == partition],
            )
        )

    cases_by_key: dict[str, set[str]] = defaultdict(set)
    for case, keys in case_keys.items():
        for key in keys:
            cases_by_key[key].add(case)
    for index, (key, cases) in enumerate(sorted(cases_by_key.items())):
        relevant = [
            item
            for item in holdouts
            if any((item[1] / case).is_dir() for case in cases)
        ]
        rows = _run_replays(
            root,
            relevant,
            output_dir,
            f"leave_grammar_{index}",
            ["--completion-exclude-calibration-key", key],
        )
        report.append(
            {
                **_summary(
                    f"leave_one_grammar_control_out:{index}",
                    [row for row in rows if row["case"] in cases],
                ),
                "calibration_key": key,
            }
        )

    operators = sorted({row["op"] for row in baseline})
    for operator in operators:
        cases = {case for case in case_keys if case.startswith(f"{operator}__")}
        keys = sorted({key for case in cases for key in case_keys.get(case, ())})
        if not keys:
            continue
        relevant = [
            item
            for item in holdouts
            if any((item[1] / case).is_dir() for case in cases)
        ]
        extra = [
            value
            for key in keys
            for value in ("--completion-exclude-calibration-key", key)
        ]
        rows = _run_replays(
            root, relevant, output_dir, f"leave_operator_{operator}", extra
        )
        report.append(
            _summary(
                f"leave_one_operator_out:{operator}",
                [row for row in rows if row["op"] == operator],
            )
        )

    fields = sorted({key for row in report for key in row})
    with (output_dir / "completion_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)
    (output_dir / "completion_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
