"""Export benchmark manifests, Explorer counters, and latency fits to one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from microbench.inf2_nki.profile_parser.fit_latency import fit_results


IDENTITY_COLUMNS = [
    "row_type",
    "run_id",
    "id",
    "status",
    "kind",
    "mode",
    "microbench_class",
    "dir",
    "error",
]


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, target)
    elif isinstance(value, (list, tuple)):
        target[prefix] = json.dumps(value, separators=(",", ":"), sort_keys=True)
    elif value is not None:
        target[prefix] = value


def _run_id(path: Path, root: Path) -> str:
    for parent in (path.parent, *path.parents):
        manifest = parent / "run_manifest.json"
        if manifest.is_file():
            try:
                return str(json.loads(manifest.read_text(encoding="utf-8")).get("run_id") or parent.name)
            except (OSError, json.JSONDecodeError):
                return parent.name
        if parent == root:
            break
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else root.name


def _profile_values(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = list(data.values()) if isinstance(data, dict) else []
    models = [model for model in models if isinstance(model, dict)]
    if not models:
        return {}
    # A compiled microbenchmark currently has one model. If Explorer emits more,
    # keep the first model intact and expose the count rather than silently adding
    # counters whose aggregation semantics are metric-dependent.
    values = dict(models[0])
    values["model_count"] = len(models)
    return values


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _add_derived(row: dict[str, Any]) -> None:
    read = _number(row, "profile.hbm_read_bytes")
    write = _number(row, "profile.hbm_write_bytes")
    dma_time = _number(row, "profile.dma_active_time")
    exec_time = _number(row, "profile.total_exec_time")
    if read is not None and write is not None:
        total = read + write
        row["derived.total_hbm_bytes_actual"] = total
        if dma_time and dma_time > 0:
            row["derived.hbm_gbps_dma_active"] = total / dma_time / 1e9
            row["derived.read_gbps_dma_active"] = read / dma_time / 1e9
            row["derived.write_gbps_dma_active"] = write / dma_time / 1e9
            partitions = _number(row, "work.partition_count")
            engines = min(16.0, partitions) if partitions and partitions > 0 else None
            if engines and engines > 0:
                row["derived.dma_engines_active"] = int(engines)
                row["derived.partitions_per_dma_engine"] = int((partitions + engines - 1) // engines)
                peak = engines * 17.0
                row["derived.dma_engine_peak_gbps"] = peak
                row["derived.read_dma_engine_utilization"] = (read / dma_time / 1e9) / peak
        if exec_time and exec_time > 0:
            row["derived.hbm_gbps_total_exec"] = total / exec_time / 1e9
    for direction in ("read", "write"):
        expected = _number(row, f"work.hbm_{direction}_bytes")
        actual = read if direction == "read" else write
        # Explorer summary may omit a counter whose value is exactly zero.
        # Only infer the omitted value when the manifest also declares zero;
        # never turn a missing non-zero counter into a false match.
        if actual is None and expected == 0:
            actual = 0.0
        if expected is not None and actual is not None:
            row[f"derived.{direction}_byte_delta"] = actual - expected
            row[f"derived.{direction}_byte_count_match"] = actual == expected
    flops = _number(row, "work.matmul_flops")
    tensor_time = _number(row, "profile.tensor_engine_active_time")
    if flops is not None and tensor_time and tensor_time > 0:
        row["derived.tensor_tflops_active"] = flops / tensor_time / 1e12
    static_bytes = _number(row, "work.static_dma_bytes")
    static_time = _number(row, "profile.static_dma_active_time")
    if static_bytes is not None and static_time and static_time > 0:
        row["derived.static_dma_gbps_active"] = static_bytes / static_time / 1e9
    elements = _number(row, "work.elements")
    engine = {"scalar_exp": "scalar", "vector_add": "vector"}.get(str(row.get("kind")))
    engine_time = _number(row, f"profile.{engine}_engine_active_time") if engine else None
    if elements is not None and engine_time and engine_time > 0:
        row[f"derived.{engine}_gelem_s_active"] = elements / engine_time / 1e9


def collect_rows(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    manifests_by_run: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or "spec" not in manifest:
            continue
        spec = manifest.get("spec") or {}
        run_id = _run_id(path, root)
        row: dict[str, Any] = {
            "row_type": "benchmark",
            "run_id": run_id,
            "id": manifest.get("id", path.parent.name),
            "status": manifest.get("status", "unknown"),
            "kind": spec.get("kind", ""),
            "mode": spec.get("mode", ""),
            "microbench_class": manifest.get("microbench_class", ""),
            "dir": str(path.parent),
            "error": manifest.get("error", ""),
        }
        _flatten("spec", spec, row)
        _flatten("work", manifest.get("work") or {}, row)
        _flatten("latency", manifest.get("latency_percentiles") or {}, row)
        profile_export = manifest.get("profile_export") or {}
        _flatten(
            "profile_export",
            {key: profile_export[key] for key in ("returncode", "elapsed_s", "output", "output_size_bytes")
             if key in profile_export},
            row,
        )
        _flatten("profile", _profile_values(path.parent / "explorer_summary.json"), row)
        _add_derived(row)
        rows.append(row)
        manifests_by_run.setdefault(run_id, []).append(manifest)

    for run_id, manifests in sorted(manifests_by_run.items()):
        fit_data = fit_results(manifests)
        for index, fit in enumerate(fit_data["fits"]):
            group = fit.get("group") or {}
            row = {
                "row_type": "latency_fit", "run_id": run_id,
                "id": f"{run_id}__latency_fit_{index}",
                "status": "error" if "error" in fit else "ok",
                "kind": group.get("kind", ""), "mode": group.get("mode", ""),
                "microbench_class": "latency", "error": fit.get("error", ""),
                "fit.source": fit_data["source"], "fit.percentile": fit_data["percentile"],
                "fit.slope_ns": float(fit["slope_us"]) * 1000 if "slope_us" in fit else "",
            }
            _flatten("spec", group, row)
            for key in ("points", "intercept_us", "r2", "x", "y_us"):
                if key in fit:
                    _flatten(f"fit.{key}", fit[key], row)
            rows.append(row)
    return rows


def export_csv(root: Path, output: Path) -> list[dict[str, Any]]:
    rows = collect_rows(root)
    extra = sorted({key for row in rows for key in row} - set(IDENTITY_COLUMNS))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=IDENTITY_COLUMNS + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run or suite directory to scan recursively")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.root / "all_results.csv"
    rows = export_csv(args.root, output)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
