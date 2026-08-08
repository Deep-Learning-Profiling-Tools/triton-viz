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


def _run(args: list[str], dry_run: bool) -> None:
    print("+", " ".join(args), flush=True)
    if not dry_run:
        subprocess.run(args, check=True)


def _module(name: str, *args: object) -> list[str]:
    return [sys.executable, "-m", name, *(str(arg) for arg in args)]


def collect(root: Path, tilebench: Path, dry_run: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    configs = [
        "engine_lowering_sweep.json",
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
        ),
        dry_run,
    )

    holdouts = [
        ("elementwise_fp32", ["interleave", "kl_divergence", "relu", "mul2", "sigmoid"], [128, 512, 2048], "float32"),
        ("norm_fp32", ["rmsnorm", "layernorm"], [128, 512, 1024, 2048, 4096], "float32"),
        ("norm_bf16", ["rmsnorm", "layernorm"], [512, 2048], "bfloat16"),
        ("softmax_fp32", ["softmax"], [128, 512, 1024, 2048], "float32"),
    ]
    for name, ops, dims, dtype in holdouts:
        _run(
            _module(
                "triton_viz.tools.nki_operator_experiments",
                "--output-dir",
                root / "holdouts" / name,
                "--tilebench-ops-dir",
                tilebench,
                "--ops",
                *ops,
                "--rows",
                128,
                "--cols",
                *dims,
                "--dtype",
                dtype,
                "--warmup",
                10,
                "--iters",
                100,
            ),
            dry_run,
        )


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
            "triton_viz.tools.nki_fit_nc_latency",
            root / "controls",
            "--output",
            calibration / "nc_latency.csv",
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
        "--dma-affine-read-csv",
        calibration / "dma_directional.csv",
        "--dma-affine-write-csv",
        write_csv,
        "--compute-calibration-csv",
        calibration / "compute.csv",
        "--structured-control-csv",
        calibration / "structured_compute.csv",
        "--structural-static-dma-csv",
        calibration / "static_dma.csv",
        "--nc-latency-csv",
        calibration / "nc_latency.csv",
        "--output",
        output,
    )
    return args


def evaluate(root: Path, dry_run: bool) -> None:
    replay_dir = root / "evaluation"
    replay_dir.mkdir(parents=True, exist_ok=True)
    holdouts = [
        ("elementwise_fp32", "float32", True),
        ("norm_fp32", "float32", False),
        ("norm_bf16", "bfloat16", False),
        ("softmax_fp32", "float32", False),
    ]
    outputs = []
    for name, dtype, strided in holdouts:
        output = replay_dir / f"{name}.csv"
        args = _replay_args(root, root / "holdouts" / name, output, dtype)
        if strided:
            args[args.index("--output"):args.index("--output")] = [
                "--strided-dma-csv",
                str(root / "calibration" / "strided_dma.csv"),
            ]
        _run(args, dry_run)
        outputs.append(output)
    if dry_run:
        return

    rows = [row for path in outputs for row in csv.DictReader(path.open())]
    stages = {
        "compute_only_mape_pct": "compute_only_error_pct",
        "compute_plus_dma_mape_pct": "compute_dma_error_pct",
        "resource_overlap_mape_pct": "resource_overlap_error_pct",
        "final_nc_p50_mape_pct": "nc_error_pct",
    }
    report = {"holdout_cases": len(rows)}
    for name, field in stages.items():
        report[name] = statistics.mean(abs(float(row[field])) for row in rows)
    worst = max(rows, key=lambda row: abs(float(row["nc_error_pct"])))
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
