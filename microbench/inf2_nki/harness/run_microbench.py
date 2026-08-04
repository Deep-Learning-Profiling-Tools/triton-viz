"""Run Inf2/NKI microbenchmarks and save reproducible artifact bundles."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import itertools
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neuronxcc import nki

from microbench.inf2_nki.common.inputs import make_input, make_pointer_ring
from microbench.inf2_nki.tests.bandwidth_dma.kernels import (
    bulk_copy_factory,
    transpose_copy_factory,
    transpose_pipeline_factory,
    transpose_pipeline_work_bytes,
    transpose_work_bytes,
    work_bytes as dma_work_bytes,
)
from microbench.inf2_nki.tests.engine_ops.kernels import (
    scalar_exp_factory,
    tensor_matmul_factory,
    vector_add_factory,
    work_units as engine_work_units,
)
from microbench.inf2_nki.tests.latency_pointer_chase.kernels import (
    dma_roundtrip_factory,
    pointer_chase_factory,
    work_units as latency_work_units,
)
from microbench.inf2_nki.tests.overlap.kernels import tensor_dma_overlap_factory
from microbench.inf2_nki.tests.program_mapping.kernels import program_mapping_factory
from microbench.inf2_nki.tests.static_dma.kernels import (
    static_dma_scatter_factory,
    work_units as static_dma_work_units,
)


BENCHMARKS = {
    # True latency microbenchmarks: loop-carried dependencies; analyze by
    # slope vs repeat/ring length, not by aggregate bandwidth.
    "pointer_chase": {
        "folder": "latency_pointer_chase",
        "factory": pointer_chase_factory,
        "work": latency_work_units,
        "input": "pointer_ring",
    },
    "dma_roundtrip_latency": {
        "folder": "latency_pointer_chase",
        "factory": dma_roundtrip_factory,
        "work": latency_work_units,
        "input": "default",
    },
    # Bandwidth microbenchmarks: bulk transfers; analyze bytes/time and DMA
    # packet throughput.
    "dma_bandwidth": {
        "folder": "bandwidth_dma",
        "factory": bulk_copy_factory,
        "work": dma_work_bytes,
        "input": "default",
    },
    # Backward-compatible alias for the initial simple suite name.
    "dma_copy": {
        "folder": "bandwidth_dma",
        "factory": bulk_copy_factory,
        "work": dma_work_bytes,
        "input": "default",
    },
    "dma_transpose": {
        "folder": "bandwidth_dma",
        "factory": transpose_copy_factory,
        "work": transpose_work_bytes,
        "input": "default",
    },
    "dma_transpose_pipeline": {
        "folder": "bandwidth_dma",
        "factory": transpose_pipeline_factory,
        "work": transpose_pipeline_work_bytes,
        "input": "default",
    },
    # Engine-local instruction microbenchmarks.
    "vector_add": {
        "folder": "engine_ops",
        "factory": vector_add_factory,
        "work": engine_work_units,
        "input": "default",
    },
    "scalar_exp": {
        "folder": "engine_ops",
        "factory": scalar_exp_factory,
        "work": engine_work_units,
        "input": "default",
    },
    "tensor_matmul": {
        "folder": "engine_ops",
        "factory": tensor_matmul_factory,
        "work": engine_work_units,
        "input": "default",
    },
    # Cross-resource interaction and SPMD mapping probes.
    "tensor_dma_overlap": {
        "folder": "overlap",
        "factory": tensor_dma_overlap_factory,
        "work": None,
        "input": "default",
    },
    "program_mapping": {
        "folder": "program_mapping",
        "factory": program_mapping_factory,
        "work": None,
        "input": "default",
    },
    "static_dma_scatter": {
        "folder": "static_dma",
        "factory": static_dma_scatter_factory,
        "work": static_dma_work_units,
        "input": "default",
    },
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_s": round(time.time() - start, 3),
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": 127, "stdout": "", "stderr": str(exc), "elapsed_s": round(time.time() - start, 3)}
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": _text(exc.stdout),
            "stderr": _text(exc.stderr) or f"Timed out after {timeout}s",
            "elapsed_s": round(time.time() - start, 3),
        }


def _latency_percentiles_from_metric(metric: Any) -> dict[str, float] | None:
    if metric is None:
        return None
    out: dict[str, float] = {}
    for p in (0, 1, 10, 25, 50, 90, 99, 100):
        try:
            out[f"p{p}_us"] = float(metric.get_latency_percentile(p))
        except Exception:
            pass
    return out or None


def _latency_percentiles(bench_func: Any) -> dict[str, Any] | None:
    result = getattr(bench_func, "benchmark_result", None)
    if result is None:
        return None
    out: dict[str, Any] = {}
    for name in ("nc_latency", "latency"):
        metric = getattr(result, name, None)
        values = _latency_percentiles_from_metric(metric)
        if values is not None:
            out[name] = values
    full_results = getattr(result, "full_results", None)
    if isinstance(full_results, dict):
        for name in ("nc_latency", "latency", "cc_latency"):
            raw = full_results.get(name)
            if isinstance(raw, dict):
                out[name] = {f"p{k}_us": float(v) for k, v in raw.items() if v is not None}
        for name in ("throughput", "inference_count", "input_type", "tensor_placement"):
            if name in full_results:
                out[name] = full_results[name]
    return out or None


def _collect_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for mod_name in ("nki", "neuronxcc", "torch", "torch_neuronx", "torch_xla"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[mod_name] = f"unavailable: {exc}"
    for tool in ("neuron-ls", "neuron-explorer", "neuron-profile", "neuronx-cc"):
        versions[f"{tool}_path"] = shutil.which(tool)
    neuron_ls = _run(["neuron-ls"], timeout=10)
    versions["neuron_ls"] = {
        "returncode": neuron_ls["returncode"],
        "stdout": neuron_ls["stdout"],
        "stderr": neuron_ls["stderr"],
    }
    return versions


def _export_profile(bench_dir: Path, export: str, timeout_s: int) -> dict[str, Any] | None:
    if export == "none":
        return None
    neff = bench_dir / "file.neff"
    ntff = bench_dir / "profile.ntff"
    if not neff.exists() or not ntff.exists():
        return {"error": "missing file.neff or profile.ntff"}
    if export == "summary-json":
        output = bench_dir / "explorer_summary.json"
        cmd = [
            "neuron-explorer",
            "view",
            "-n",
            str(neff),
            "-s",
            str(ntff),
            "--output-format",
            "summary-json",
            "--disable-ui",
            "--ignore-event-trace",
        ]
    elif export == "json":
        output = bench_dir / "explorer.json"
        cmd = [
            "neuron-explorer",
            "view",
            "-n",
            str(neff),
            "-s",
            str(ntff),
            "--output-format",
            "json",
            "--output-file",
            str(output),
            "--disable-ui",
            "--ignore-event-trace",
        ]
    elif export == "parquet":
        output = bench_dir / "explorer_parquet"
        output.mkdir(exist_ok=True)
        cmd = [
            "neuron-explorer",
            "view",
            "-n",
            str(neff),
            "-s",
            str(ntff),
            "--output-format",
            "parquet",
            "--output-file",
            str(output),
            "--disable-ui",
            "--ignore-event-trace",
        ]
    else:
        raise ValueError(f"Unknown profile export {export!r}")
    res = _run(cmd, cwd=bench_dir, timeout=timeout_s)
    (bench_dir / "explorer_view_stdout.txt").write_text(res["stdout"], encoding="utf-8")
    (bench_dir / "explorer_view_stderr.txt").write_text(res["stderr"], encoding="utf-8")
    if export == "summary-json" and res["returncode"] == 0:
        output.write_text(res["stdout"], encoding="utf-8")
    res["output"] = str(output)
    if output.exists():
        if output.is_dir():
            res["output_files"] = [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()]
        else:
            res["output_size_bytes"] = output.stat().st_size
    return res


def _instantiate(spec: dict[str, Any]):
    kind = spec["kind"]
    info = BENCHMARKS[kind]
    factory = info["factory"]
    params = {k: v for k, v in spec.items() if k not in {"name", "kind", "modes", "mode"}}
    if "dtype" in params:
        params["dtype_name"] = params.pop("dtype")
    mode = spec["mode"]
    result = factory(mode=mode, **params)
    if len(result) == 2:  # legacy factory form
        kernel, shapes = result
        grid = (1,)
    else:
        kernel, shapes, grid = result
    return kernel, shapes, grid


def _bench_folder(spec: dict[str, Any]) -> str:
    return str(BENCHMARKS[spec["kind"]]["folder"])


def _bench_id(spec: dict[str, Any]) -> str:
    name = spec.get("name", spec["kind"])
    mode = spec["mode"]
    parts = [name, mode]
    for key in ("dtype", "p", "f", "x", "y", "m", "k", "n", "ring_length", "stride", "repeat", "programs", "placement", "dge_mode"):
        if key in spec:
            parts.append(f"{key}{spec[key]}")
    return "__".join(str(p).replace("/", "_") for p in parts)


def _work_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    work_fn = BENCHMARKS[spec["kind"]].get("work")
    if work_fn is None:
        return {}
    params = {k: v for k, v in spec.items() if k not in {"name", "kind", "modes"}}
    if "dtype" in params:
        params["dtype_name"] = params.pop("dtype")
    try:
        return dict(work_fn(**params))
    except Exception as exc:
        return {"work_metadata_error": repr(exc)}


def _make_inputs(spec: dict[str, Any], shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    input_kind = BENCHMARKS[spec["kind"]].get("input", "default")
    if input_kind == "pointer_ring":
        return [make_pointer_ring(int(spec["ring_length"]), int(spec.get("stride", 1)))]
    dtype_name = spec.get("dtype", "float32")
    return [make_input(tuple(shape), dtype_name, seed=i) for i, shape in enumerate(shapes)]


LEGACY_MODE_ALIASES = {
    ("dma_copy", "independent"): ("dma_bandwidth", "hbm_to_sbuf_stream"),
    ("dma_copy", "dependent"): ("dma_roundtrip_latency", "serialized_roundtrip"),
    ("vector_add", "dependent"): ("vector_add", "dependent_chain"),
    ("vector_add", "independent"): ("vector_add", "independent_stream"),
    ("scalar_exp", "dependent"): ("scalar_exp", "dependent_chain"),
    ("scalar_exp", "independent"): ("scalar_exp", "independent_stream"),
    ("tensor_matmul", "dependent"): ("tensor_matmul", "dependent_accumulate"),
    ("tensor_matmul", "independent"): ("tensor_matmul", "independent_stream"),
    ("tensor_dma_overlap", "independent"): ("tensor_dma_overlap", "independent_overlap"),
}


def expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for spec in config.get("benchmarks", []):
        matrix = spec.get("matrix") or {}
        matrix_keys = list(matrix)
        combinations = itertools.product(*(matrix[key] for key in matrix_keys)) if matrix_keys else [()]
        for combination in combinations:
            base = {key: value for key, value in spec.items() if key != "matrix"}
            base.update(dict(zip(matrix_keys, combination)))
            modes = base.get("modes", [base.get("mode", "independent")])
            for mode in modes:
                row = dict(base)
                row.pop("modes", None)
                kind = row["kind"]
                alias = LEGACY_MODE_ALIASES.get((kind, mode))
                resolved_mode = mode
                if alias is not None:
                    row["kind"], resolved_mode = alias
                row["mode"] = resolved_mode
                expanded.append(row)
    return expanded


def run_one(spec: dict[str, Any], run_dir: Path, warmup: int, iters: int, profile_export: str, explorer_timeout_s: int, skip_existing: bool = False) -> dict[str, Any]:
    # The runner changes cwd below because nki.benchmark writes some artifacts
    # relative to it.  Resolve first so profile post-processing does not look
    # for ``<bench_dir>/<relative bench_dir>/file.neff`` afterwards.
    run_dir = run_dir.resolve()
    bench_id = _bench_id(spec)
    bench_dir = run_dir / _bench_folder(spec) / bench_id
    if skip_existing and (bench_dir / "manifest.json").exists():
        return {"id": bench_id, "status": "skipped_existing", "dir": str(bench_dir)}
    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    bench_dir.mkdir(parents=True)
    artifacts_dir = bench_dir / "compiler_artifacts"

    row: dict[str, Any] = {
        "id": bench_id,
        "status": "unknown",
        "spec": spec,
        "dir": str(bench_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    old_cwd = Path.cwd()
    try:
        os.chdir(bench_dir)
        kernel, shapes, grid = _instantiate(spec)
        inputs = _make_inputs(spec, [tuple(shape) for shape in shapes])
        bench = nki.benchmark(
            warmup=warmup,
            iters=iters,
            save_neff_name="file.neff",
            save_trace_name="profile.ntff",
            artifacts_dir=str(artifacts_dir),
        )(kernel)
        stdout = io.StringIO()
        stderr = io.StringIO()
        t0 = time.time()
        invoked = bench
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if grid is None:
                invoked(*inputs)
            else:
                invoked = bench[grid]
                invoked(*inputs)
        elapsed = time.time() - t0
        row.update(
            {
                "status": "ok",
                "elapsed_s": round(elapsed, 3),
                "latency_percentiles": _latency_percentiles(invoked) or _latency_percentiles(bench),
                "grid": grid,
                "input_shapes": [list(a.shape) for a in inputs],
                "input_dtypes": [str(a.dtype) for a in inputs],
                "microbench_class": _bench_folder(spec),
                "work": _work_metadata(spec),
            }
        )
        (bench_dir / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
        (bench_dir / "stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
    except Exception as exc:
        row.update({"status": "error", "error": repr(exc)})
        try:
            (bench_dir / "stdout.txt").write_text(locals().get("stdout", io.StringIO()).getvalue(), encoding="utf-8")
            (bench_dir / "stderr.txt").write_text(locals().get("stderr", io.StringIO()).getvalue(), encoding="utf-8")
        except Exception:
            pass
    else:
        # Profile export is a post-processing step: a profiler failure (e.g. a
        # parquet-export timeout) must not flip a successful kernel run to
        # "error", otherwise we would discard valid latency/artifact data.
        try:
            row["profile_export"] = _export_profile(bench_dir, profile_export, explorer_timeout_s)
        except Exception as exc:
            row["profile_export"] = {"status": "error", "error": repr(exc)}
    finally:
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = {
            **row,
            "paths": {
                "neff": str(bench_dir / "file.neff"),
                "ntff": str(bench_dir / "profile.ntff"),
                "compiler_artifacts": str(artifacts_dir),
            },
        }
        (bench_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.chdir(old_cwd)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("microbench/inf2_nki/configs/quick.json"))
    parser.add_argument("--output-root", type=Path, default=Path("microbench/inf2_nki/results"))
    parser.add_argument("--run-id", default=None, help="Default: UTC timestamp plus config suite name")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-kind", default=None)
    parser.add_argument("--profile-export", choices=["none", "summary-json", "json", "parquet"], default="none")
    parser.add_argument("--explorer-timeout-s", type=int, default=180)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    specs = expand_config(config)
    if args.filter_kind:
        specs = [s for s in specs if s.get("kind") == args.filter_kind]
    if args.limit is not None:
        specs = specs[: args.limit]

    warmup = args.warmup if args.warmup is not None else int(config.get("warmup", 2))
    iters = args.iters if args.iters is not None else int(config.get("iters", 10))
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}__{config.get('suite', 'inf2_nki')}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "config_path": str(args.config),
        "config": config,
        "warmup": warmup,
        "iters": iters,
        "profile_export": args.profile_export,
        "versions": _collect_versions(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")

    results_path = run_dir / "results.jsonl"
    ok = 0
    with results_path.open("a", encoding="utf-8") as f:
        for i, spec in enumerate(specs, start=1):
            print(f"[{i}/{len(specs)}] {spec['kind']} mode={spec['mode']} spec={spec}", flush=True)
            row = run_one(spec, run_dir, warmup, iters, args.profile_export, args.explorer_timeout_s, args.skip_existing)
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f.flush()
            print(f"  -> {row['status']} {row.get('latency_percentiles')}", flush=True)
            ok += row.get("status") == "ok"

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["num_benchmarks"] = len(specs)
    metadata["num_ok"] = ok
    (run_dir / "run_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Wrote {results_path}")
    return 0 if ok == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
