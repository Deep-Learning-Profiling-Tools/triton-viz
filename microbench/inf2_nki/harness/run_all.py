"""Run the canonical Inf2 NKI validation suite and produce one CSV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from microbench.inf2_nki.profile_parser.export_csv import export_csv
from microbench.inf2_nki.profile_parser.plot_dma_free_dimension import plot_csv
from microbench.inf2_nki.profile_parser.plot_dma_partition_surface import plot_surface
from microbench.inf2_nki.profile_parser.plot_dma_transpose_surface import plot_transpose_surface
from microbench.inf2_nki.profile_parser.plot_dma_copy_vs_transpose import plot_comparison


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PACKAGE_DIR / "configs" / "all.json"


def _load_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("configs"), list) or not data["configs"]:
        raise ValueError(f"{path}: 'configs' must be a non-empty list")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-root", type=Path, default=PACKAGE_DIR / "results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--profile-export", choices=["none", "parquet", "summary-json"], default="summary-json")
    parser.add_argument("--explorer-timeout-s", type=int, default=120)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    suite_path = args.suite_config.resolve()
    suite = _load_suite(suite_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{stamp}__{suite.get('suite', 'inf2_nki_all')}"
    suite_dir = args.output_root.resolve() / run_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []

    for item in suite["configs"]:
        config = (suite_path.parent / str(item)).resolve()
        child_id = config.stem
        command = [
            sys.executable,
            "-m",
            "microbench.inf2_nki.harness.run_microbench",
            "--config",
            str(config),
            "--output-root",
            str(suite_dir),
            "--run-id",
            child_id,
            "--profile-export",
            args.profile_export,
            "--explorer-timeout-s",
            str(args.explorer_timeout_s),
        ]
        print(f"[{child_id}] running {config.name}", flush=True)
        completed = subprocess.run(command, check=False)
        records.append({"config": str(config), "run_id": child_id, "returncode": completed.returncode})
        if completed.returncode and not args.continue_on_error:
            break

    csv_path = suite_dir / "all_results.csv"
    rows = export_csv(suite_dir, csv_path)
    plot_path = suite_dir / "dma_free_dimension.png"
    try:
        plotted_points = len(plot_csv(csv_path, plot_path))
        plot_status: dict[str, Any] = {"status": "ok", "path": str(plot_path), "points": plotted_points}
    except (ImportError, RuntimeError, ValueError) as exc:
        plot_status = {"status": "error", "error": repr(exc)}
    surface_path = suite_dir / "dma_partition_surface.png"
    try:
        surface_points = len(plot_surface(csv_path, surface_path))
        surface_status: dict[str, Any] = {
            "status": "ok", "path": str(surface_path), "points": surface_points
        }
    except (ImportError, RuntimeError, ValueError) as exc:
        surface_status = {"status": "error", "error": repr(exc)}
    transpose_path = suite_dir / "dma_transpose_surface.png"
    comparison_path = suite_dir / "dma_copy_vs_transpose.png"
    try:
        transpose_points = len(plot_transpose_surface(csv_path, transpose_path))
        comparison_points = int(plot_comparison(csv_path, csv_path, comparison_path).size)
        transpose_status: dict[str, Any] = {
            "status": "ok", "path": str(transpose_path), "points": transpose_points,
            "comparison_path": str(comparison_path), "comparison_points": comparison_points,
        }
    except (ImportError, RuntimeError, ValueError) as exc:
        transpose_status = {"status": "error", "error": repr(exc)}
    failed = [record for record in records if record["returncode"]]
    suite_manifest = {
        "suite": suite,
        "suite_config": str(suite_path),
        "run_id": run_id,
        "runs": records,
        "csv": str(csv_path),
        "csv_rows": len(rows),
        "dma_free_dimension_plot": plot_status,
        "dma_partition_surface_plot": surface_status,
        "dma_transpose_surface_plot": transpose_status,
        "status": "failed" if failed or any(
            status["status"] != "ok" for status in (plot_status, surface_status, transpose_status)
        ) else "ok",
    }
    (suite_dir / "suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"CSV: {csv_path}", flush=True)
    if plot_status["status"] == "ok":
        print(f"DMA curve: {plot_path}", flush=True)
    else:
        print(f"DMA curve failed: {plot_status['error']}", flush=True)
    if surface_status["status"] == "ok":
        print(f"DMA surface: {surface_path}", flush=True)
    else:
        print(f"DMA surface failed: {surface_status['error']}", flush=True)
    if transpose_status["status"] == "ok":
        print(f"DMA transpose surface: {transpose_path}", flush=True)
        print(f"DMA transpose/copy comparison: {comparison_path}", flush=True)
    else:
        print(f"DMA transpose surface failed: {transpose_status['error']}", flush=True)
    return 1 if failed or any(
        status["status"] != "ok" for status in (plot_status, surface_status, transpose_status)
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
