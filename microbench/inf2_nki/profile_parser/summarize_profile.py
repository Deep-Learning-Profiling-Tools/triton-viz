"""Schema-tolerant Neuron Explorer parquet/profile summary helper.

Neuron Explorer profile schemas evolve quickly.  This parser does not assume one
exact layout; it inventories parquet tables and extracts duration/opcode/engine
aggregates when columns with familiar names are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_columns(path: Path) -> tuple[dict[str, list[Any]] | None, str | None]:
    """Load a parquet file into a column->list dict using any available engine.

    Returns ``(columns, error)``. Neuron Explorer emits parquet, but the Neuron
    venv does not always ship a parquet engine, so we try pyarrow first and then
    pandas (which itself needs pyarrow/fastparquet). We surface the failure
    reason instead of silently returning empty tables, so a missing dependency
    is obvious in the summary output.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return {name: table[name].to_pylist() for name in table.column_names}, None
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - engine specific
        return None, f"pyarrow read failed: {exc!r}"
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path)
        return {str(col): frame[col].tolist() for col in frame.columns}, None
    except Exception as exc:
        return None, f"no usable parquet engine (install pyarrow): {exc!r}"

def _first_column(names: set[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {n.lower(): n for n in names}
    for cand in candidates:
        if cand in names:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _numeric_summary(values: list[Any]) -> dict[str, float] | None:
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return None

    def pct(p: float) -> float:
        if len(nums) == 1:
            return nums[0]
        pos = (len(nums) - 1) * p / 100.0
        lo = int(pos)
        hi = min(lo + 1, len(nums) - 1)
        frac = pos - lo
        return nums[lo] * (1 - frac) + nums[hi] * frac

    return {
        "count": float(len(nums)),
        "min": nums[0],
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": nums[-1],
        "sum": float(sum(nums)),
    }


def _value_counts(values: list[Any], limit: int = 40) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])


def summarize_parquet_file(path: Path) -> dict[str, Any]:
    columns, error = _load_columns(path)
    if columns is None:
        return {"path": str(path), "error": error or "could not read parquet"}
    column_names = list(columns.keys())
    names = set(column_names)
    num_rows = len(next(iter(columns.values()))) if columns else 0
    out: dict[str, Any] = {
        "path": str(path),
        "rows": num_rows,
        "columns": column_names,
    }
    duration_col = _first_column(names, ("duration_ns", "duration", "latency_ns", "elapsed_ns"))
    start_col = _first_column(names, ("start_ts", "start_time_ns", "timestamp_ns"))
    end_col = _first_column(names, ("end_ts", "end_time_ns"))
    engine_col = _first_column(names, ("engine", "engine_name", "Engine", "hw_engine"))
    opcode_col = _first_column(names, ("opcode", "instruction", "instruction_name", "op_name", "name"))
    source_col = _first_column(names, ("nki_source_location", "source_location", "source", "line"))

    if duration_col:
        out["duration_ns"] = _numeric_summary(columns[duration_col])
    elif start_col and end_col:
        starts = columns[start_col]
        ends = columns[end_col]
        out["duration_ns"] = _numeric_summary([e - s for s, e in zip(starts, ends) if s is not None and e is not None])
    if engine_col:
        out["engine_counts"] = _value_counts(columns[engine_col])
    if opcode_col:
        out["opcode_counts"] = _value_counts(columns[opcode_col])
    if source_col:
        out["source_location_counts"] = _value_counts(columns[source_col], limit=80)

    # Capture common byte/flop/wait metrics if present.
    metric_candidates = (
        "hbm_read_bytes",
        "hbm_write_bytes",
        "sbuf_read_bytes",
        "sbuf_write_bytes",
        "psum_read_bytes",
        "psum_write_bytes",
        "flops",
        "evt_wait_time_ns",
        "dma_wait_time_ns",
        "spill_save_bytes",
        "spill_reload_bytes",
    )
    metrics: dict[str, Any] = {}
    for col in column_names:
        if col in metric_candidates or col.lower() in metric_candidates:
            metrics[col] = _numeric_summary(columns[col])
    if metrics:
        out["metrics"] = metrics
    return out


def summarize_run(root: Path) -> dict[str, Any]:
    manifests = sorted(root.rglob("manifest.json"))
    result: dict[str, Any] = {"root": str(root), "benchmarks": {}}
    for manifest_path in manifests:
        if manifest_path.name == "run_manifest.json":
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        bench_dir = manifest_path.parent
        bench_id = manifest.get("id", bench_dir.name)
        parquet_root = bench_dir / "explorer_parquet"
        tables = []
        if parquet_root.exists():
            for pq_file in sorted(parquet_root.rglob("*.parquet")):
                tables.append(summarize_parquet_file(pq_file))
        result["benchmarks"][bench_id] = {
            "status": manifest.get("status"),
            "spec": manifest.get("spec"),
            "latency_percentiles": manifest.get("latency_percentiles"),
            "dir": str(bench_dir),
            "parquet_tables": tables,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run directory or a single explorer_parquet directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.root.is_dir() and any(args.root.glob("*.parquet")):
        data: Any = {"root": str(args.root), "parquet_tables": [summarize_parquet_file(p) for p in sorted(args.root.glob("*.parquet"))]}
    else:
        data = summarize_run(args.root)
    text = json.dumps(data, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
