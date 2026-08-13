"""Three-stage Inf2 NKI cost-model paper pipeline.

The stages deliberately keep calibration controls and Tilebench holdouts in
separate directories: ``collect`` gathers both, ``fit`` reads controls only,
and ``evaluate`` replays the frozen model on holdouts and reports ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from triton_viz.tools.nki_provenance import (
    collect_compiler_fingerprint,
    validate_model_manifest,
    write_model_manifest,
)

FORMAL_SPLITS = Path("microbench/inf2_nki/configs/formal_holdouts.json")
TOOL_MODULES = {
    "microbench.inf2_nki.harness.run_microbench": "microbench.inf2_nki.harness.run_microbench",
    "microbench.inf2_nki.profile_parser.export_csv": "microbench.inf2_nki.profile_parser.export_csv",
    "microbench.inf2_nki.profile_parser.fit_compute_calibration": "microbench.inf2_nki.profile_parser.fit_compute_calibration",
    "triton_viz.tools.nki_region_control_experiments": "triton_viz.tools.nki_region_control_experiments",
    "triton_viz.tools.nki_operator_experiments": "triton_viz.tools.nki_operator_experiments",
    "triton_viz.tools.nki_fit_structured_controls": "triton_viz.tools.nki_fit_structured_controls",
    "triton_viz.tools.nki_fit_structural_static_dma": "triton_viz.tools.nki_fit_structural_static_dma",
    "triton_viz.tools.nki_fit_runtime_overhead": "triton_viz.tools.nki_fit_runtime_overhead",
    "triton_viz.tools.nki_fit_strided_dma": "triton_viz.tools.nki_fit_strided_dma",
    "triton_viz.tools.nki_replay_operator_predictions": "triton_viz.tools.nki_replay_operator_predictions",
}


def _run(args: list[str], dry_run: bool) -> None:
    print("+", " ".join(args), flush=True)
    _validate_command_contract(args)
    if not dry_run:
        subprocess.run(args, check=True)


def _module(name: str, *args: object) -> list[str]:
    return [sys.executable, "-m", name, *(str(arg) for arg in args)]


def _validate_command_contract(command: list[str]) -> None:
    """Parse a generated child command without executing its workload."""
    if len(command) < 4 or command[1] != "-m":
        raise ValueError(f"Not a Python module command: {command}")
    module_name = command[2]
    if module_name not in TOOL_MODULES:
        raise ValueError(f"Unknown pipeline child module: {module_name}")
    import importlib

    module = importlib.import_module(module_name)
    if hasattr(module, "build_parser"):
        module.build_parser().parse_args(command[3:])
        return

    # Capture the parser at the exact parse_args boundary. This validates the
    # real generated argv, then aborts before any hardware/filesystem work.
    from unittest.mock import patch

    original_parse_args = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def parse_and_stop(parser, args=None, namespace=None):
        original_parse_args(parser, command[3:], namespace)
        raise Parsed

    try:
        with patch.object(argparse.ArgumentParser, "parse_args", parse_and_stop):
            module.main(command[3:])
    except Parsed:
        return
    raise RuntimeError(f"{module_name}.main did not parse command-line arguments")


def _load_splits(path: Path = FORMAL_SPLITS) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "triton-viz.nki-formal-holdouts-v1":
        raise ValueError(f"Unsupported holdout split schema in {path}")
    return data["splits"]


def _split_case_count(split: dict) -> int:
    rows = split["rows"] if isinstance(split["rows"], list) else [split["rows"]]
    return len(rows) * sum(len(dims) for dims in split["operators"].values())


def _collect_split_commands(root: Path, tilebench: Path, name: str, split: dict):
    """Group operators by identical dimension lists without changing the split."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for op, dims in split["operators"].items():
        groups.setdefault(tuple(int(dim) for dim in dims), []).append(op)
    commands = []
    rows_values = split["rows"] if isinstance(split["rows"], list) else [split["rows"]]
    command_count = len(groups) * len(rows_values)
    index = 0
    for rows in rows_values:
        for dims, ops in sorted(groups.items()):
            output_name = name if command_count == 1 else f"{name}_{index}"
            commands.append(_module(
                "triton_viz.tools.nki_operator_experiments",
                "--output-dir",
                root / "holdouts" / output_name,
                "--tilebench-ops-dir",
                tilebench,
                "--ops",
                *ops,
                "--rows",
                int(rows),
                "--cols",
                *dims,
                "--dtype",
                split["dtype"],
                "--warmup",
                10,
                "--iters",
                100,
                "--resume",
            ))
            index += 1
    return commands


def collect(root: Path, tilebench: Path, dry_run: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    configs = [
        "engine_lowering_sweep.json",
        "runtime_overhead.json",
        "dma_partition_surface.json",
        "dma_partition_large_free.json",
        "dma_transpose_surface.json",
        "dma_directional_dtype_canary.json",
        "dma_write_partition_surface.json",
        "dma_write_bf16_steady.json",
        "dma_strided_store_surface.json",
    ]
    for config in configs:
        run_id = Path(config).stem
        _run(
            _module(
                "microbench.inf2_nki.harness.run_microbench",
                "--config",
                Path("microbench/inf2_nki/configs") / config,
                "--output-root",
                root / "microbench",
                "--run-id",
                run_id,
                "--profile-export",
                "summary-json",
                "--skip-existing",
            ),
            dry_run,
        )

    control_kinds = [
        "elementwise_maximum",
        "elementwise_multiply",
        "elementwise_sigmoid",
        "masked_log_reduction",
        "two_pass_reduce_affine",
        "two_pass_reduce_multiply",
        "softmax_reduction",
    ]
    _run(
        _module(
            "triton_viz.tools.nki_region_control_experiments",
            "--output-dir",
            root / "controls",
            "--kinds",
            *control_kinds,
            "--free-dims",
            128,
            512,
            1024,
            2048,
            4096,
            "--dtypes",
            "float32",
            "bfloat16",
            "--p",
            128,
            "--warmup",
            10,
            "--iters",
            100,
            "--resume",
        ),
        dry_run,
    )

    splits = _load_splits()
    for name, split in splits.items():
        for command in _collect_split_commands(root, tilebench, name, split):
            _run(command, dry_run)


def fit(root: Path, dry_run: bool) -> None:
    calibration = root / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    canonical = calibration / "microbench.csv"
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench",
            "--output",
            canonical,
        ),
        dry_run,
    )
    dma_exports = {
        "dma_directional.csv": "dma_directional_dtype_canary",
        "dma_read_surface.csv": "dma_partition_surface",
        "dma_transpose_surface.csv": "dma_transpose_surface",
        "dma_write_fp32.csv": "dma_write_partition_surface",
        "dma_write_bf16.csv": "dma_write_bf16_steady",
    }
    for output_name, run_name in dma_exports.items():
        _run(
            _module(
                "microbench.inf2_nki.profile_parser.export_csv",
                root / "microbench" / run_name,
                "--output",
                calibration / output_name,
            ),
            dry_run,
        )
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.export_csv",
            root / "microbench" / "dma_partition_large_free",
            "--output",
            calibration / "dma_read_large_free.csv",
        ),
        dry_run,
    )
    compute = calibration / "compute.csv"
    _run(
        _module(
            "microbench.inf2_nki.profile_parser.fit_compute_calibration",
            canonical,
            "--output",
            compute,
        ),
        dry_run,
    )
    structured = calibration / "structured_compute.csv"
    _run(
        _module(
            "triton_viz.tools.nki_fit_structured_controls",
            root / "controls",
            "--compute-calibration-csv",
            compute,
            "--output",
            structured,
        ),
        dry_run,
    )
    _run(
        _module(
            "triton_viz.tools.nki_fit_structural_static_dma",
            root / "controls",
            "--output",
            calibration / "static_dma.csv",
        ),
        dry_run,
    )
    _run(
        _module(
            "triton_viz.tools.nki_fit_runtime_overhead",
            root / "microbench" / "runtime_overhead" / "results.jsonl",
            "--dma-read-surface-csv",
            calibration / "microbench.csv",
            "--dma-read-bf16-surface-csv",
            calibration / "dma_directional.csv",
            "--dma-write-surface-csv",
            calibration / "dma_write_fp32.csv",
            "--dma-write-bf16-surface-csv",
            calibration / "dma_write_bf16.csv",
            "--compute-calibration-csv",
            compute,
            "--output",
            calibration / "runtime_overhead.csv",
        ),
        dry_run,
    )
    _run(
        _module(
            "triton_viz.tools.nki_fit_strided_dma",
            root / "microbench" / "dma_strided_store_surface" / "results.jsonl",
            "--output",
            calibration / "strided_dma.csv",
        ),
        dry_run,
    )
    if not dry_run:
        source_manifests = [
            root / "controls" / "experiment_manifest.json",
            *sorted((root / "microbench").glob("*/run_manifest.json")),
        ]
        write_model_manifest(
            calibration,
            calibration_files=[
                canonical,
                calibration / "dma_directional.csv",
                calibration / "dma_read_surface.csv",
                calibration / "dma_read_large_free.csv",
                calibration / "dma_transpose_surface.csv",
                calibration / "dma_write_fp32.csv",
                calibration / "dma_write_bf16.csv",
                compute,
                structured,
                calibration / "static_dma.csv",
                calibration / "runtime_overhead.csv",
                calibration / "strided_dma.csv",
            ],
            source_manifests=source_manifests,
            split_file=FORMAL_SPLITS,
        )


def _replay_args(root: Path, holdout: Path, output: Path, dtype: str) -> list[str]:
    calibration = root / "calibration"
    write_csv = (
        calibration / "dma_write_bf16.csv"
        if dtype == "bfloat16"
        else calibration / "dma_write_fp32.csv"
    )
    args = _module(
        "triton_viz.tools.nki_replay_operator_predictions",
        holdout,
        "--dma-read-surface-csv",
        (
            calibration / "dma_directional.csv"
            if dtype == "bfloat16"
            else calibration / "microbench.csv"
        ),
        "--dma-write-surface-csv",
        write_csv,
        "--compute-calibration-csv",
        calibration / "compute.csv",
        "--structured-control-csv",
        calibration / "structured_compute.csv",
        "--structural-static-dma-csv",
        calibration / "static_dma.csv",
        "--runtime-overhead-csv",
        calibration / "runtime_overhead.csv",
        "--output",
        output,
    )
    if dtype == "float32":
        args[args.index("--compute-calibration-csv"):args.index("--compute-calibration-csv")] = [
            "--dma-transpose-surface-csv",
            str(calibration / "dma_transpose_surface.csv"),
        ]
    return args


def _evaluate_replays(
    root: Path,
    replay_dir: Path,
    splits: dict,
    *,
    model_name: str,
    dry_run: bool,
) -> list[tuple[str, Path]]:
    outputs: list[tuple[str, Path]] = []
    for split_name, split in splits.items():
        holdout_dirs = sorted((root / "holdouts").glob(f"{split_name}*"))
        if dry_run and not holdout_dirs:
            holdout_dirs = [
                root / "holdouts" / split_name
                if len(_collect_split_commands(root, Path("."), split_name, split)) == 1
                else root / "holdouts" / f"{split_name}_{index}"
                for index in range(
                    len(_collect_split_commands(root, Path("."), split_name, split))
                )
            ]
        for holdout in holdout_dirs:
            output = replay_dir / f"{model_name}_{holdout.name}.csv"
            args = _replay_args(root, holdout, output, split["dtype"])
            args[args.index("--output"):args.index("--output")] = [
                "--strict-calibration",
            ]
            if split_name in {"formal_fp32_v1", "full_fp32_v1"}:
                args[args.index("--output"):args.index("--output")] = [
                    "--strided-dma-csv",
                    str(root / "calibration" / "strided_dma.csv"),
                ]
            _run(args, dry_run)
            outputs.append((split_name, output))
    return outputs


def evaluate(root: Path, dry_run: bool) -> None:
    replay_dir = root / "evaluation"
    replay_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        calibration = root / "calibration"
        validate_model_manifest(
            calibration / "model_manifest.json",
            calibration_files=[
                calibration / "microbench.csv",
                calibration / "dma_directional.csv",
                calibration / "dma_read_surface.csv",
                calibration / "dma_read_large_free.csv",
                calibration / "dma_transpose_surface.csv",
                calibration / "dma_write_fp32.csv",
                calibration / "dma_write_bf16.csv",
                calibration / "compute.csv",
                calibration / "structured_compute.csv",
                calibration / "static_dma.csv",
                calibration / "runtime_overhead.csv",
                calibration / "strided_dma.csv",
            ],
            current_fingerprint=collect_compiler_fingerprint(
                Path(__file__).resolve().parents[2]
            ),
        )
    splits = _load_splits()
    outputs = _evaluate_replays(
        root,
        replay_dir,
        splits,
        model_name="surface",
        dry_run=dry_run,
    )
    if dry_run:
        return

    rows_by_split = {
        split_name: [
            row
            for name, path in outputs
            if name == split_name
            for row in csv.DictReader(path.open())
        ]
        for split_name in splits
    }
    formal_rows = rows_by_split["formal_fp32_v1"]
    expected = _split_case_count(splits["formal_fp32_v1"])
    if len(formal_rows) != expected:
        raise ValueError(
            f"formal_fp32_v1 expected {expected} successful cases, got {len(formal_rows)}"
        )
    full_rows = rows_by_split["full_fp32_v1"]
    full_expected = _split_case_count(splits["full_fp32_v1"])
    if len(full_rows) != full_expected or full_expected != 120:
        raise ValueError(
            f"full_fp32_v1 expected exactly 120 successful cases, got {len(full_rows)}"
        )
    stages = {
        "compute_only_mape_pct": "compute_only_error_pct",
        "compute_plus_dma_mape_pct": "compute_dma_error_pct",
        "resource_overlap_mape_pct": "resource_overlap_error_pct",
        "final_nc_p50_mape_pct": "nc_error_pct",
    }
    auxiliary_rows = rows_by_split.get("auxiliary_bf16_v1", [])
    report = {
        "formal_fp32_cases": len(formal_rows),
        "full_fp32_cases": len(full_rows),
        "auxiliary_bf16_cases": len(auxiliary_rows),
    }
    report["full_fp32_nc_p50_mape_pct"] = statistics.mean(
        abs(float(row["nc_error_pct"])) for row in full_rows
    )
    report["full_fp32_rows_mape_pct"] = {
        str(rows): statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in full_rows
            if int(row["rows"]) == rows
        )
        for rows in (1, 16, 128)
    }
    report["full_fp32_dma_surface_event_counts"] = {
        match: sum(int(row[field]) for row in full_rows)
        for match, field in {
            "exact": "dma_surface_exact_count",
            "interpolated": "dma_surface_interpolated_count",
            "ood_clamped": "dma_surface_ood_count",
        }.items()
    }
    if auxiliary_rows:
        report["auxiliary_bf16_nc_p50_mape_pct"] = statistics.mean(
            abs(float(row["nc_error_pct"])) for row in auxiliary_rows
        )
    for name, field in stages.items():
        report[name] = statistics.mean(
            abs(float(row[field])) for row in formal_rows
        )
    report["formal_fp32_dma_surface_event_counts"] = {
        "exact": sum(
            int(row["dma_surface_exact_count"]) for row in formal_rows
        ),
        "interpolated": sum(
            int(row["dma_surface_interpolated_count"]) for row in formal_rows
        ),
        "ood_clamped": sum(
            int(row["dma_surface_ood_count"]) for row in formal_rows
        ),
    }
    report["formal_fp32_dma_surface_max_log_distance"] = max(
        float(row["dma_surface_max_log_distance"]) for row in formal_rows
    )
    report["formal_fp32_operator_mape_pct"] = {
        operator: statistics.mean(
            abs(float(row["nc_error_pct"]))
            for row in formal_rows
            if row["op"] == operator
        )
        for operator in sorted({row["op"] for row in formal_rows})
    }
    worst = max(formal_rows, key=lambda row: abs(float(row["nc_error_pct"])))
    report["worst_case"] = {
        "case": worst["case"],
        "error_pct": float(worst["nc_error_pct"]),
    }
    (replay_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "fit", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--tilebench-dir",
        type=Path,
        default=Path("/home/ubuntu/Tilebench/benchmarks/operators"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "collect":
        collect(args.root.resolve(), args.tilebench_dir.resolve(), args.dry_run)
    elif args.stage == "fit":
        fit(args.root.resolve(), args.dry_run)
    else:
        evaluate(args.root.resolve(), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
