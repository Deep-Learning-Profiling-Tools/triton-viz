"""Durable pinned reruns: one immutable experiment across driver sessions.

The public CLI uses evaluation.pinned_resume for per-row main/retry commits,
owned service execution, pause/resume, identity checks and atomic publication.
This module retains the shared retry predicate, merge rules and statistics.
See evaluation/PINNED_RESUME.md for the protocol and operational instructions.

Usage:
    python -m evaluation.pinned_run start --ladder-level L2
    python -m evaluation.pinned_run pause --run-dir RUN_DIRECTORY
    python -m evaluation.pinned_run resume --run-dir RUN_DIRECTORY
    python -m evaluation.pinned_run start --ladder-level L0 --purpose attribution
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.runner import (
    _run_one,
    assert_protocol_dataset,
    run_corpus,
)
from triton_viz.clients.race_detector.ladder import (
    LadderLevel,
)

# The 16 corpora of the pinned run, longest first (a problem with fla
# surfaces early); the four suite/litmus files are last.
REAL_CODE_CORPORA = (
    "fla",
    "tritonbench_g",
    "aiter_ops",
    "flaggems",
    "torchao",
    "tilebench_cutile",
    "tilebench",
    "tritonbench_meta",
    "flagattn",
    "liger",
    "tutorials",
    "aiter_originals",
)
LITMUS_CORPORA = (
    "tritonracebench",
    "tritonracebench_cutile",  # the cuda.tile twins (results.md E9)
    "await_sync",
    "rmw_sync",
    "golden_smoke",
)
ALL_CORPORA = REAL_CODE_CORPORA + LITMUS_CORPORA

RETRY_TIMEOUT_S = 320
LOAD_MAX = 2.0  # 1-minute load average the guard waits under
LOAD_POLL_S = 15
PINNED_SUFFIX = "_pinned"

# extent of a proof, from its terminal (evaluation.md section 12; the
# L1 rung's proved@enum is the analyzed launch, like proved@interp)
EXTENT_OF = {
    "proved@T0": "any",
    "proved@T0+assumes-termination": "any",
    "proved@T1": "input",
    "proved@T1+assumes-termination": "input",
    "proved@T1-launch": "launch",
    "proved@T1-launch+assumes-termination": "launch",
    "proved@interp": "analyzed",
    "proved@enum": "analyzed",
    # Route 2 (L2): the proof went through a snapshot Select, so it is
    # qualified by this launch's contents, one rung below the plain one
    "proved@T1+content": "input+content",
    "proved@T1+assumes-termination+content": "input+content",
    "proved@T1-launch+content": "launch+content",
    "proved@T1-launch+assumes-termination+content": "launch+content",
}
EXTENTS = ("any", "input", "input+content", "launch", "launch+content", "analyzed")


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=root,
    ).stdout.strip()


def tree_is_clean(root: Path) -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        cwd=root,
    ).stdout
    return out.strip() == ""


# ── the load guard ──────────────────────────────────────────────────


def _foreign_evaluation_processes() -> list[str]:
    """Command lines of evaluation runner/harness processes that are not
    ours (not descendants of this process)."""
    me = os.getpid()
    out = subprocess.run(
        ["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True
    ).stdout.splitlines()
    parent_of: dict[int, int] = {}
    args_of: dict[int, str] = {}
    for line in out[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = int(parts[0]), int(parts[1]), parts[2]
        parent_of[pid] = ppid
        args_of[pid] = args

    def ours(pid: int) -> bool:
        seen = 0
        while pid in parent_of and seen < 64:
            if pid == me:
                return True
            pid = parent_of[pid]
            seen += 1
        return pid == me

    return [
        a
        for pid, a in args_of.items()
        if ("evaluation.runner" in a or "evaluation.harness" in a) and not ours(pid)
    ]


def load_guard(enabled: bool, log: Any, what: str) -> None:
    """Block until no foreign evaluation process runs and the 1-minute
    load average is under LOAD_MAX; log every wait."""
    if not enabled:
        return
    waited = 0
    while True:
        foreign = _foreign_evaluation_processes()
        load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        if not foreign and load1 < LOAD_MAX:
            if waited:
                print(
                    f"[pinned] load guard released after {waited}s before {what}",
                    file=log,
                    flush=True,
                )
            return
        if waited == 0:
            why = (
                f"{len(foreign)} foreign evaluation process(es)"
                if foreign
                else f"load {load1:.1f} >= {LOAD_MAX}"
            )
            print(
                f"[pinned] load guard: waiting before {what} ({why})",
                file=log,
                flush=True,
            )
        time.sleep(LOAD_POLL_S)
        waited += LOAD_POLL_S


# ── passes ─────────────────────────────────────────────────────────


def _read_rows(path: Path) -> tuple[dict, list[dict]]:
    header: dict = {}
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d.get("header"):
            header = d
        else:
            rows.append(d)
    return header, rows


def budget_reached(row: dict, budget: float) -> bool:
    return row.get("terminal") == "timeout" or float(row.get("wall_s", 0.0)) >= budget


def main_pass(
    corpora: tuple[str, ...],
    level: LadderLevel,
    seed: int,
    row_timeout: int,
    guard: bool,
    log: Any,
    suffix: str,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for corpus in corpora:
        load_guard(guard, log, f"corpus {corpus}")
        t0 = time.perf_counter()
        files[corpus] = run_corpus(
            corpus,
            None,
            seed,
            row_timeout,
            mutate=False,
            jobs=1,
            ladder_level=level,
            out_suffix=suffix,
        )
        print(
            f"[pinned] {corpus}: main pass done in {time.perf_counter() - t0:.0f}s -> {files[corpus].name}",
            file=log,
            flush=True,
        )
    return files


def retry_pass(
    files: dict[str, Path],
    level: LadderLevel,
    seed: int,
    row_timeout: int,
    retry_timeout: int,
    guard: bool,
    log: Any,
) -> dict[tuple[str, str], dict]:
    """Rerun every budget-reaching row once at the retry budget; returns
    the retry rows keyed by (corpus, name)."""
    from evaluation.kernels import load

    retried: dict[tuple[str, str], dict] = {}
    for corpus, path in files.items():
        _, rows = _read_rows(path)
        names = [r["name"] for r in rows if budget_reached(r, row_timeout)]
        if not names:
            continue
        specs = {s.name: s for s in load(corpus).specs}
        for name in names:
            load_guard(guard, log, f"retry {corpus}/{name}")
            row = _run_one(specs[name], corpus, seed, retry_timeout, False, level)
            row["corpus"] = corpus
            retried[(corpus, name)] = row
            print(
                f"[pinned] retry {corpus}/{name}: {row.get('terminal')} in {row.get('wall_s')}s",
                file=log,
                flush=True,
            )
    return retried


def merge(
    files: dict[str, Path],
    retried: dict[tuple[str, str], dict],
    commit: str,
    level: LadderLevel,
    row_timeout: int,
    retry_timeout: int,
    seed: int,
    fence_order: bool = True,
) -> tuple[dict, list[dict]]:
    """The merged, stamped dataset. Every per-corpus file must be a
    protocol dataset (never the debugging worker-reuse kind) produced
    under the run's memory-model switch (``fence_order``)."""
    header: dict = {
        "header": True,
        "pinned_commit": commit,
        "commit": commit,
        "ladder_level": level.name,
        "row_timeout_s": row_timeout,
        "retry_timeout_s": retry_timeout,
        "seed": seed,
        "jobs": 1,
        "worker_reuse": False,
        # the memory model the run was produced under (design-fence-order.md
        # stage 5: a pinned run is fence-ordered; the commit's config default
        # says so, and the header and every row repeat it)
        "fence_order": fence_order,
        "fence_order_env": os.environ.get("TRITON_VIZ_FENCE_ORDER"),
        "corpora": {},
    }
    merged: list[dict] = []
    for corpus, path in files.items():
        per_corpus_header = assert_protocol_dataset(path)
        if per_corpus_header.get("ladder_level") != level.name:
            raise ValueError(
                f"{path}: ladder level {per_corpus_header.get('ladder_level')} "
                f"in a {level.name} pinned run"
            )
        if per_corpus_header.get("fence_order") is not fence_order:
            raise ValueError(
                f"{path}: fence_order={per_corpus_header.get('fence_order')!r} "
                f"in a fence_order={fence_order} pinned run"
            )
        header["corpora"][corpus] = {
            k: v
            for k, v in per_corpus_header.items()
            if k not in ("header", "corpus", "seed", "jobs", "worker_reuse")
        }
        _, rows = _read_rows(path)
        for row in rows:
            row = dict(row)
            row["corpus"] = corpus
            if row.get("fence_order") is not fence_order:
                raise ValueError(
                    f"{path}: row {row.get('name')!r} ran under "
                    f"fence_order={row.get('fence_order')!r} in a "
                    f"fence_order={fence_order} pinned run"
                )
            retry = retried.get((corpus, row["name"]))
            if retry is not None and retry.get("verdict") not in (None, "error"):
                row = dict(retry)
                row["corpus"] = corpus
                row["pinned_pass"] = "retry"
                row["pinned_wall_s"] = float(row.pop("wall_s", 0.0))
            elif retry is not None:
                # the retry also failed: the row stands as a budget
                # timeout, no verdict (the T/o column), the wall of the
                # longest attempt
                for k in ("verdict", "terminal"):
                    row.pop(k, None)
                row["pinned_error"] = True
                row["pinned_pass"] = "retry"
                row["pinned_wall_s"] = float(
                    retry.get("wall_s", row.pop("wall_s", 0.0))
                )
                row.pop("wall_s", None)
                row["harness_error"] = retry.get("harness_error") or row.get(
                    "harness_error"
                )
            else:
                row["pinned_pass"] = "main"
                row["pinned_wall_s"] = float(row.pop("wall_s", 0.0))
            row["pinned_commit"] = commit
            merged.append(row)
    return header, merged


# ── statistics (evaluation.md sections 6 and 12) ───────────────────


def overhead_stats(files: dict[str, Path], budget: float) -> dict:
    """Section 6 recipe over the real-code per-corpus files: pooled
    per-row wall_s, timeouts = status/terminal timeout or wall >= budget,
    median / p95 (linear interpolation) / max over within-budget rows."""
    walls: list[float] = []
    timeouts = n = 0
    for corpus, path in files.items():
        if corpus not in REAL_CODE_CORPORA:
            continue
        _, rows = _read_rows(path)
        for r in rows:
            if "wall_s" not in r:
                continue
            n += 1
            if r.get("terminal") == "timeout" or r["wall_s"] >= budget:
                timeouts += 1
            else:
                walls.append(r["wall_s"])
    walls.sort()
    if walls:
        k = (len(walls) - 1) * 0.95
        f = math.floor(k)
        p95 = walls[f] + (walls[math.ceil(k)] - walls[f]) * (k - f)
        median, mx = statistics.median(walls), walls[-1]
    else:
        p95 = median = mx = 0.0
    return {
        "rows": n,
        "within_budget": len(walls),
        "median_s": median,
        "p95_s": p95,
        "max_s": mx,
        "timeouts": timeouts,
        "budget_s": budget,
    }


def verdict_table(merged: list[dict]) -> dict[str, dict]:
    """Section 12 counting per corpus: proofs by extent, races,
    abstentions (error rows fold in), budget timeouts (pinned_error, no
    verdict), median pinned_wall_s over rows with a verdict."""
    table: dict[str, dict] = {}
    by_corpus: dict[str, list[dict]] = defaultdict(list)
    for r in merged:
        by_corpus[r["corpus"]].append(r)
    for corpus, rows in by_corpus.items():
        c: Counter = Counter()
        walls = []
        for r in rows:
            if r.get("pinned_error") and "verdict" not in r:
                c["timeout"] += 1
                continue
            v = r.get("verdict")
            if v == "race-free":
                c["proof"] += 1
                c[f"extent_{EXTENT_OF.get(r.get('terminal', ''), 'other')}"] += 1
            elif v == "race":
                c["race"] += 1
            else:  # abstain, error (capture/compile failures fold in)
                c["abstain"] += 1
            if v is not None:
                walls.append(r["pinned_wall_s"])
        table[corpus] = {
            "rows": len(rows),
            "proof": c["proof"],
            **{e: c[f"extent_{e}"] for e in EXTENTS},
            "race": c["race"],
            "abstain": c["abstain"],
            "timeout": c["timeout"],
            "median_s": round(statistics.median(walls), 1) if walls else None,
        }
    return table


def summary_markdown(
    header: dict, merged: list[dict], stats: dict, table: dict[str, dict]
) -> str:
    lines = [
        f"# Pinned run {header['pinned_commit']} at {header['ladder_level']}",
        "",
        f"Rows {len(merged)}, seed {header['seed']}, jobs 1, row budget "
        f"{header['row_timeout_s']} s, retry budget {header['retry_timeout_s']} s, "
        f"fence order {'ON' if header.get('fence_order', True) else 'OFF (legacy)'}, "
        "one subprocess per row (no worker reuse).",
        "",
        "## Overhead (evaluation.md section 6 recipe, real-code corpora)",
        "",
        f"- rows {stats['rows']}, within budget {stats['within_budget']}, "
        f"timeouts {stats['timeouts']} (budget {stats['budget_s']} s)",
        f"- median {stats['median_s']:.2f} s, p95 {stats['p95_s']:.1f} s, max {stats['max_s']:.1f} s",
        "",
        "## Verdicts per corpus (evaluation.md section 12 counting)",
        "",
        "| corpus | rows | proofs | any | input | input+content | launch | "
        "launch+content | analyzed | races | abstain | T/o | median s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in list(REAL_CODE_CORPORA) + list(LITMUS_CORPORA):
        t = table.get(corpus)
        if t is None:
            continue
        ext = " | ".join(str(t[e]) for e in EXTENTS)
        lines.append(
            f"| {corpus} | {t['rows']} | {t['proof']} | {ext} | {t['race']} | "
            f"{t['abstain']} | {t['timeout']} | "
            f"{t['median_s'] if t['median_s'] is not None else '-'} |"
        )
    real = [table[c] for c in REAL_CODE_CORPORA if c in table]
    if real:
        keys = ("rows", "proof", *EXTENTS, "race", "abstain", "timeout")
        tot = {k: sum(t[k] for t in real) for k in keys}
        ext = " / ".join(f"{tot[e]} {e}" for e in EXTENTS)
        lines += [
            "",
            f"Real-code totals: {tot['rows']} rows, {tot['proof']} proofs ({ext}), "
            f"{tot['race']} race rows, {tot['abstain']} abstentions, "
            f"{tot['timeout']} timeouts.",
        ]
    lines.append("")
    return "\n".join(lines)


# ── the driver ─────────────────────────────────────────────────────


def run_pinned(
    level: LadderLevel,
    corpora: tuple[str, ...] = ALL_CORPORA,
    seed: int = 0,
    row_timeout: int | None = None,
    retry_timeout: int = RETRY_TIMEOUT_S,
    rehearsal: bool = False,
    guard: bool = True,
    log: Any = None,
) -> Path:
    """Start a durable run; direct rehearsal calls execute synchronously.

    Formal calls dispatch an owned service and return its run directory.
    Completed rehearsal calls return the published dataset path.
    """
    from evaluation.pinned_resume import start_run

    return start_run(
        level,
        corpora,
        seed,
        row_timeout,
        retry_timeout,
        rehearsal,
        guard,
        purpose="definitive" if level.name == "L2" else "attribution",
        foreground=rehearsal,
    )


def main() -> None:
    # Every public invocation now uses the durable scheduler. Legacy flags
    # remain a spelling of "start"; pure merge/statistic helpers stay shared.
    from evaluation.pinned_resume import main as resumable_main

    argv = sys.argv[1:]
    commands = {"start", "resume", "pause", "status", "verify", "_execute"}
    resumable_main(argv if argv and argv[0] in commands else ["start", *argv])
    return


if __name__ == "__main__":
    main()
