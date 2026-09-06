"""REHEARSAL ONLY: reproducible durable-checkpoint overhead measurements.

``micro`` replays every row from the explicitly supplied surviving JSONL
files, once per round, without running a detector. ``paired`` executes the
predeclared roster in ABBA order. A omits the experiment ledger; B uses
the real RunStore. Both save identical diagnostic records after each row,
outside its wall timer. No output of this module is a pinned dataset.

Example (run from the isolated checkout with its imports selected):
    python -m evaluation.checkpoint_overhead --phase micro --output-dir PATH \
        --input-jsonl OLD_FLA.jsonl --input-jsonl OLD_TRITONBENCH.jsonl
    python -m evaluation.checkpoint_overhead --phase paired --output-dir PATH
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid


# Fixed before measurement; historical times select workloads, never controls.
# The last official-budget candidate previously took 156.24 / 200 seconds.
SELECTION = (
    ("golden_smoke", "smoke_add_no"),
    ("golden_smoke", "smoke_bcast_store_yes"),
    ("golden_smoke", "smoke_gather_no"),
    ("rmw_sync", "lbd_no"),
    ("rmw_sync", "lbd_relaxed_yes"),
    ("fla", "fla_delta_rule_chunk__chunk_scaled_dot_kkt_fwd_kernel"),
    ("tritonbench_g", "tb_matmul_triton2"),
    ("tritonbench_g", "tb_kv_cache_copy"),
    ("tritonbench_g", "tb_chunk_retention__chunk_retention_bwd_kernel_dqkv"),
)
TIMEOUT_PROBE = (("tritonbench_g", "tb_chunk_gate_recurrence___bwd_recurrence"),)
PROTOCOL = "checkpoint-overhead-rehearsal-v1"
LIMITATIONS = (
    "REHEARSAL ONLY; never a pinned dataset. Two observations per mode in "
    "ABBA cannot establish equivalence, stable tail distributions, or absence "
    "of indirect timing effects. The 15-second timeout probe does not establish "
    "behavior at the official 200/320-second limits. Both modes durably save "
    "diagnostics outside the row timer; A disables only the experiment ledger. "
    "Caches are left in their naturally evolving state, with no explicit "
    "warmup or flushing; balanced order reduces but cannot remove drift."
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    """Exclusive, durable diagnostic output; failed runs are not overwritten."""
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def snapshot():
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_s": time.perf_counter(),
        "loadavg": list(os.getloadavg()),
    }


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def filesystem(path):
    """Record the containing mount, without assuming /tmp shares its device."""
    target = str(path.resolve())
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        mount = fields[4].replace("\\040", " ")
        if target == mount or target.startswith(mount.rstrip("/") + "/"):
            matches.append(
                (
                    len(mount),
                    {
                        "mount": mount,
                        "filesystem": right.split()[0],
                        "device": fields[2],
                        "mount_options": fields[5],
                        "st_dev": path.stat().st_dev,
                    },
                )
            )
    if not matches:
        raise RuntimeError(f"cannot identify filesystem for {path}")
    return max(matches, key=lambda item: item[0])[1]


def provenance(output_dir):
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    packages = {}
    for name in (
        "triton",
        "torch",
        "numpy",
        "z3-solver",
        "fla-core",
        "liger-kernel",
        "flag_attn",
        "flag_gems",
        "torchao",
        "tritonbench",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    hashes = {}
    for path in sorted((root / "evaluation" / "kernels").rglob("*")):
        if path.suffix in (".py", ".json", ".npz") and path.is_file():
            hashes[str(path.relative_to(root))] = sha256(path.read_bytes())
    for name in ("runner.py", "pinned_state.py", "checkpoint_overhead.py"):
        path = root / "evaluation" / name
        if path.exists():
            hashes[str(path.relative_to(root))] = sha256(path.read_bytes())
    allowlist = (
        "PYTHONPATH",
        "TRITON_INTERPRET",
        "TRITON_VIZ_FENCE_ORDER",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "TRITON_CACHE_DIR",
        "PYTHONHASHSEED",
    )
    return {
        "recorded_at": snapshot(),
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_status": git("status", "--porcelain", "--untracked-files=no"),
        "python": sys.version,
        "executable": sys.executable,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "packages": packages,
        "environment": {key: os.environ.get(key) for key in allowlist},
        "filesystem": filesystem(output_dir),
        "source_input_sha256": hashes,
    }


def manifest(rows, budget, label):
    return {
        "protocol_version": PROTOCOL,
        "run_id": str(uuid.uuid4()),
        "rehearsal": True,
        "label": label,
        "fingerprints": {},
        "rows": rows,
        "config": {
            "ladder_level": "L2",
            "row_timeout_s": budget,
            "retry_timeout_s": 320,
            "seed": 0,
            "fence_order": True,
        },
    }


class Samples:
    def __init__(self, path):
        self.stream = path.open("x")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        fsync_directory(path.parent)
        self.records = []

    def append(self, record):
        # This common diagnostic sink is deliberately outside both row timers
        # and ledger timings; A is not a no-disk-I/O machine experiment.
        self.stream.write(canonical(record) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records.append(record)

    def close(self):
        self.stream.close()


def measured_begin(store, corpus, name, session, budget):
    started = time.perf_counter()
    attempt = store.begin_attempt(corpus, name, "main", session, budget)
    elapsed = time.perf_counter() - started
    return attempt, elapsed, dict(store.last_begin_metrics)


def measured_commit(store, attempt, row):
    started = time.perf_counter()
    metrics = store.commit_result(attempt, row)
    return time.perf_counter() - started, metrics


def micro(output_dir, input_paths, rounds, samples):
    from evaluation.pinned_state import RunStore

    source = []
    inputs = []
    for path in input_paths:
        raw = path.read_bytes()
        input_info = {
            "path": str(path.resolve()),
            "sha256": sha256(raw),
            "bytes": len(raw),
            "filesystem": filesystem(path),
        }
        if path.stat().st_dev != output_dir.stat().st_dev:
            raise ValueError(
                "micro inputs and output must use the same filesystem device"
            )
        header = None
        for number, line in enumerate(raw.splitlines(), 1):
            row = json.loads(line)
            if row.get("header"):
                if header is not None or number != 1:
                    raise ValueError(f"misplaced or duplicate header: {path}:{number}")
                header = row
                continue
            if header is None:
                raise ValueError(f"missing header: {path}")
            if header.get("ladder_level") != "L2" or header.get("row_timeout_s") != 200:
                raise ValueError(
                    "micro input must be the original L2/200-second raw rows"
                )
            source.append((path, number, row, len(line) + 1))
        input_info["header"] = header
        inputs.append(input_info)
    if not source:
        raise ValueError("micro input contains no rows")
    plan = {
        "kind": "REHEARSAL",
        "phase": "micro",
        "rounds": rounds,
        "inputs": inputs,
        "rows_per_round": len(source),
        "source_row_bytes": distribution([item[3] for item in source]),
        "limitations": "Payload replay measures persistence cost, not detector timing effects.",
    }
    write_json(output_dir / "plan.json", plan)
    for round_index in range(rounds):
        prepared = []
        for index, (path, number, original, line_bytes) in enumerate(source):
            row = dict(
                original, name=f"replay_{round_index}_{index}_{original['name']}"
            )
            prepared.append((path, number, row, line_bytes))
        roster = [
            {
                "corpus": row["corpus"],
                "name": row["name"],
                "spec_hash": sha256(canonical(row).encode()),
            }
            for _, _, row, _ in prepared
        ]
        store = RunStore.create(
            output_dir / f"micro-{round_index}",
            manifest(roster, 200, f"micro-{round_index}"),
        )
        session = store.new_session({"rehearsal": True, "phase": "micro"})
        try:
            for index, (path, number, row, line_bytes) in enumerate(prepared):
                before = snapshot()
                t0 = time.perf_counter()
                encoded = canonical(row).encode()
                encoding_s = time.perf_counter() - t0
                attempt, begin_s, begin_metrics = measured_begin(
                    store, row["corpus"], row["name"], session, 200
                )
                commit_s, commit_metrics = measured_commit(store, attempt, row)
                samples.append(
                    {
                        "kind": "REHEARSAL",
                        "phase": "micro",
                        "round": round_index,
                        "index": index,
                        "source": str(path),
                        "source_line": number,
                        "source_line_bytes": line_bytes,
                        "canonical_row_bytes": len(encoded),
                        "encoding_probe_s": encoding_s,
                        "begin_external_s": begin_s,
                        "commit_external_s": commit_s,
                        "ledger_external_s": begin_s + commit_s,
                        "begin_metrics": begin_metrics,
                        "commit_metrics": commit_metrics,
                        "before": before,
                        "after": snapshot(),
                        "row": row,
                    }
                )
                if (index + 1) % 100 == 0 or index + 1 == len(prepared):
                    print(
                        f"[micro] round {round_index + 1}/{rounds}: "
                        f"{index + 1}/{len(prepared)} rows saved",
                        flush=True,
                    )
            store.end_session(session, "measurement-complete")
        finally:
            store.close()
    return {
        "source_row_bytes": plan["source_row_bytes"],
        **{
            key: distribution([record[key] for record in samples.records])
            for key in (
                "canonical_row_bytes",
                "encoding_probe_s",
                "begin_external_s",
                "commit_external_s",
                "ledger_external_s",
            )
        },
        "result_serialization_s": distribution(
            [r["commit_metrics"]["serialization_s"] for r in samples.records]
        ),
        "result_transaction_s": distribution(
            [r["commit_metrics"]["commit_s"] for r in samples.records]
        ),
    }


def paired_summary(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["group"], record["corpus"], record["name"])].append(record)
    rows = []
    for (group, corpus, name), values in groups.items():
        by_mode = {mode: [v for v in values if v["mode"] == mode] for mode in "AB"}
        a = [v["row"]["wall_s"] for v in by_mode["A"]]
        b = [v["row"]["wall_s"] for v in by_mode["B"]]
        mean_a, mean_b = statistics.mean(a), statistics.mean(b)
        semantic = [
            {
                "block": v["block"],
                "mode": v["mode"],
                "verdict": v["row"].get("verdict"),
                "terminal": v["row"].get("terminal"),
                "reasons": semantic_reasons(v["row"]),
            }
            for v in values
        ]
        pairs = []
        for offset in range(0, len(values), 2):
            adjacent = {v["mode"]: v for v in values[offset : offset + 2]}
            wa, wb = (adjacent[m]["row"]["wall_s"] for m in "AB")
            pairs.append(
                {
                    "blocks": [v["block"] for v in values[offset : offset + 2]],
                    "B_minus_A_s": wb - wa,
                    "B_over_A": wb / wa if wa else None,
                }
            )
        rows.append(
            {
                "group": group,
                "corpus": corpus,
                "name": name,
                "A_wall_s": a,
                "B_wall_s": b,
                "A_range_s": max(a) - min(a),
                "B_range_s": max(b) - min(b),
                "B_minus_A_mean_s": mean_b - mean_a,
                "B_over_A_mean": mean_b / mean_a if mean_a else None,
                "adjacent_pairs": pairs,
                "semantics": semantic,
                "semantic_outcomes_identical": len(
                    {
                        canonical({k: v[k] for k in ("verdict", "terminal", "reasons")})
                        for v in semantic
                    }
                )
                == 1,
            }
        )
    persistent = [v for v in records if v["mode"] == "B"]
    wall_sum = sum(v["row"]["wall_s"] for v in persistent)
    cost_sum = sum(v["ledger_external_s"] for v in persistent)
    return {
        "rows": rows,
        "B_ledger_external_s": distribution(
            [v["ledger_external_s"] for v in persistent]
        ),
        "B_accepted_wall_s": wall_sum,
        "direct_checkpoint_fraction_of_B_wall": cost_sum / wall_sum
        if wall_sum
        else None,
        "timeout_counts": {
            mode: sum(
                v["row"].get("terminal") == "timeout"
                for v in records
                if v["mode"] == mode
            )
            for mode in "AB"
        },
        "timeout_counts_by_group": {
            group: {
                mode: sum(
                    v["row"].get("terminal") == "timeout"
                    for v in records
                    if v["mode"] == mode and v["group"] == group
                )
                for mode in "AB"
            }
            for group in sorted({v["group"] for v in records})
        },
        "limitations": LIMITATIONS,
    }


def semantic_reasons(row):
    """Retain named reasons without comparing solver time or witness choices."""
    reasons = {}

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                current = f"{path}.{key}" if path else key
                if "reason" in key:
                    reasons[current] = child
                elif isinstance(child, (dict, list)):
                    visit(child, current)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(row, "")
    return reasons


def paired(output_dir, blocks, samples):
    from evaluation.kernels import load
    from evaluation.pinned_run import load_guard
    from evaluation.pinned_state import RunStore
    from evaluation.runner import _run_one
    from triton_viz.clients.race_detector.ladder import LadderLevel
    from triton_viz.core.config import config as cfg
    import evaluation.runner
    import triton_viz

    root = Path(__file__).resolve().parent.parent
    for module in (evaluation.runner, triton_viz):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(
                f"import escaped measurement checkout: {module.__file__}"
            )
    if not cfg.race_detector_fence_order:
        raise ValueError("paired measurement requires fence order enabled")
    plan = {
        "kind": "REHEARSAL",
        "phase": "paired",
        "blocks": blocks,
        "selection": [{"corpus": c, "name": n} for c, n in SELECTION],
        "budget_s": 200,
        "timeout_probe": [{"corpus": c, "name": n} for c, n in TIMEOUT_PROBE],
        "timeout_probe_budget_s": 15,
        "level": "L2",
        "seed": 0,
        "fresh_subprocess": True,
        "load_guard": "before each block",
        "A": "no experiment ledger",
        "B": "RunStore durable start/result transactions",
        "limitations": LIMITATIONS,
    }
    write_json(output_dir / "plan.json", plan)
    corpora = {name: load(name) for name in dict(SELECTION + TIMEOUT_PROBE)}
    specs = {
        (corpus, spec.name): spec
        for corpus, value in corpora.items()
        for spec in value.specs
    }
    for key in SELECTION + TIMEOUT_PROBE:
        if key not in specs:
            raise ValueError(f"predeclared measurement row missing: {key}")
    for group, selection, budget in (
        ("representative", SELECTION, 200),
        ("timeout_stress", TIMEOUT_PROBE, 15),
    ):
        roster = [
            {
                "corpus": c,
                "name": n,
                "spec_hash": sha256(
                    canonical({"corpus": c, "name": n, "protocol": PROTOCOL}).encode()
                ),
            }
            for c, n in selection
        ]
        for block_index, mode in enumerate(blocks):
            label = f"{group}-{block_index}-{mode}"
            load_guard(True, sys.stderr, label)
            block_before = snapshot()
            store = (
                RunStore.create(
                    output_dir / f"ledger-{label}", manifest(roster, budget, label)
                )
                if mode == "B"
                else None
            )
            session = (
                store.new_session({"rehearsal": True, "block": label})
                if store
                else None
            )
            try:
                for index, (corpus, name) in enumerate(selection):
                    child_output = output_dir / "child-output" / label / str(index)
                    child_output.mkdir(parents=True)
                    before = snapshot()
                    begin_s, begin_metrics, attempt = 0.0, {}, None
                    if store:
                        attempt, begin_s, begin_metrics = measured_begin(
                            store, corpus, name, session, budget
                        )
                    print(
                        f"[paired] {label} {index + 1}/{len(selection)} "
                        f"{corpus}/{name}",
                        flush=True,
                    )
                    row = _run_one(
                        specs[(corpus, name)],
                        corpus,
                        0,
                        budget,
                        False,
                        LadderLevel.L2,
                        cancel_requested=lambda: False,
                        output_dir=child_output,
                    )
                    commit_s, commit_metrics = (
                        measured_commit(store, attempt, row) if store else (0.0, {})
                    )
                    samples.append(
                        {
                            "kind": "REHEARSAL",
                            "phase": "paired",
                            "group": group,
                            "block": block_index,
                            "mode": mode,
                            "index": index,
                            "corpus": corpus,
                            "name": name,
                            "budget_s": budget,
                            "before": before,
                            "after": snapshot(),
                            "begin_external_s": begin_s,
                            "commit_external_s": commit_s,
                            "ledger_external_s": begin_s + commit_s,
                            "begin_metrics": begin_metrics,
                            "commit_metrics": commit_metrics,
                            "canonical_row_bytes": len(canonical(row).encode()),
                            "row": row,
                        }
                    )
                if store:
                    store.end_session(session, "measurement-complete")
            finally:
                if store:
                    store.close()
            write_json(
                output_dir / f"block-{label}.json",
                {"before": block_before, "after": snapshot(), "kind": "REHEARSAL"},
            )
    return paired_summary(samples.records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("micro", "paired"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--blocks", choices=("ABBA", "BAAB"), default="ABBA")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.phase == "micro" and not args.input_jsonl:
        parser.error("micro requires explicit --input-jsonl paths")
    if args.phase == "paired" and args.input_jsonl:
        parser.error("paired uses its predeclared selection, not input result files")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fsync_directory(args.output_dir.parent)
    before = snapshot()
    write_json(args.output_dir / "provenance.json", provenance(args.output_dir))
    samples = Samples(args.output_dir / "samples.jsonl")
    try:
        if args.phase == "micro":
            summary = micro(args.output_dir, args.input_jsonl, args.rounds, samples)
        else:
            summary = paired(args.output_dir, args.blocks, samples)
    finally:
        samples.close()
    summary.update(
        {
            "kind": "REHEARSAL",
            "phase": args.phase,
            "before": before,
            "after": snapshot(),
            "complete": True,
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"[checkpoint-overhead] complete: {args.output_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
