"""The pinned rerun driver: ONE run of ONE detector commit over every corpus.

The paper's evaluation numbers must be restated from a single run at a
single commit (paper repo ``pre-submission/pinned-rerun.md``). The first
pinned run (PINNED_fb91fc0.jsonl, 2026-09-02) was driven from a session
scratchpad that was never committed; this module is that driver,
committed so the run is reproducible from the repo alone. Protocol:

  1. preconditions: a clean tracked tree (the commit hash is the
     dataset's identity), the level decided, the row budget the level's
     (180 s at L0, 200 s at L1+; ``runner.row_timeout_s``), never the
     debugging-only process reuse, and the memory-model switch ON: a
     pinned run is fence-ordered (paper repo ``design-fence-order.md``,
     stage 5; ``config.race_detector_fence_order`` defaults to True since
     the flip commit, so the pinned hash itself carries the semantics),
     the driver refuses ``TRITON_VIZ_FENCE_ORDER=0`` outside
     ``--rehearsal``, and the header and every row stamp ``fence_order``
     (the runner and harness stamp the value their own process ran
     under; the merge cross-checks every stamp against the run's);
  2. main pass: every corpus through ``runner.run_corpus`` at jobs=1,
     seed 0, one subprocess per row, under a LOAD GUARD (no other
     evaluation process of ours running, load average below a bound)
     so wall times are those of an idle machine;
  3. retry pass: every row whose main-pass wall reached the budget
     (``terminal == "timeout"`` or ``wall_s >= budget``) reruns once at
     the retry budget (320 s);
  4. merge: the retry row replaces the main row when it produced a
     verdict; otherwise the row keeps the main row's fields with
     ``pinned_error`` set and no verdict (the T/o column); every row
     carries ``pinned_commit``, ``pinned_wall_s`` (the wall of the
     attempt that stands) and ``pinned_pass``; ``wall_s`` is dropped so
     no per-row number can be mistaken for a raw runner wall. Every
     per-corpus file is checked with ``runner.assert_protocol_dataset``
     before merging (a debugging dataset is refused);
  5. statistics: the paper repo's evaluation.md section 6 recipe (real-
     code rows: median / p95 / max within budget, timeouts) and section
     12 counting (per-corpus verdicts, extents, medians) into
     ``<merged>_SUMMARY.md``.

Output: ``evaluation/results/PINNED_<commit>[_L<n>].jsonl`` plus the
per-corpus files ``<corpus>[_L<n>]_pinned.jsonl`` the runner wrote.
``--rehearsal`` (a dry run on small corpora) names its outputs
``REHEARSAL_...`` and is the only mode in which the row budget may be
overridden (to exercise the retry pass) or the tree may be dirty.

Usage:
    python -m evaluation.pinned_run --ladder-level L2
    python -m evaluation.pinned_run --ladder-level L0
    python -m evaluation.pinned_run --rehearsal --corpora golden_smoke --row-timeout 2
"""

from __future__ import annotations

import argparse
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
    RESULTS_DIR,
    _run_one,
    assert_protocol_dataset,
    row_timeout_s,
    run_corpus,
)
from triton_viz.core.config import config as cfg
from triton_viz.clients.race_detector.ladder import (
    LADDER_LEVEL_NAMES,
    LadderLevel,
    parse_ladder_level,
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
    log = log or sys.stderr
    root = Path(__file__).resolve().parent.parent
    protocol_timeout = row_timeout_s(level)
    if row_timeout is None:
        row_timeout = protocol_timeout
    if not rehearsal:
        if row_timeout != protocol_timeout:
            raise SystemExit(
                f"a pinned run uses the level's budget ({protocol_timeout} s); "
                "--row-timeout is a rehearsal-only override"
            )
        if not tree_is_clean(root):
            raise SystemExit(
                "the tracked tree is dirty: a pinned run's identity is its "
                "commit hash (commit or stash first, or use --rehearsal)"
            )
        if not cfg.race_detector_fence_order:
            raise SystemExit(
                "a pinned run is fence-ordered (design-fence-order.md stage 5): "
                "TRITON_VIZ_FENCE_ORDER=0 is set in this environment; unset it "
                "(legacy-order attribution runs are --rehearsal only)"
            )
    fence_order = bool(cfg.race_detector_fence_order)
    commit = git_commit(root)
    tag = "REHEARSAL" if rehearsal else "PINNED"
    level_suffix = "" if level == LadderLevel.L0 else f"_{level.name}"
    suffix = PINNED_SUFFIX + ("-rehearsal" if rehearsal else "")
    out = RESULTS_DIR / f"{tag}_{commit}{level_suffix}.jsonl"
    print(
        f"[pinned] {tag} run at commit {commit}, level {level.name}, "
        f"{len(corpora)} corpora, row budget {row_timeout} s, retry {retry_timeout} s"
        + (", fence order ON" if fence_order else ", fence order OFF (legacy)")
        + ("" if guard else ", load guard OFF"),
        file=log,
        flush=True,
    )
    t_all = time.perf_counter()
    files = main_pass(corpora, level, seed, row_timeout, guard, log, suffix)
    retried = retry_pass(files, level, seed, row_timeout, retry_timeout, guard, log)
    header, merged = merge(
        files, retried, commit, level, row_timeout, retry_timeout, seed, fence_order
    )
    header["rehearsal"] = rehearsal
    header["retried_rows"] = len(retried)
    header["total_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in merged:
            f.write(json.dumps(row) + "\n")
    stats = overhead_stats(files, float(row_timeout))
    table = verdict_table(merged)
    summary = out.with_name(out.stem + "_SUMMARY.md")
    summary.write_text(summary_markdown(header, merged, stats, table))
    print(
        f"[pinned] merged {len(merged)} rows ({len(retried)} retried) -> {out.name}; "
        f"summary -> {summary.name}; total {header['total_s']}s",
        file=log,
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ladder-level", choices=LADDER_LEVEL_NAMES, default=LadderLevel.L0.name
    )
    ap.add_argument(
        "--corpora", nargs="+", default=list(ALL_CORPORA), help="default: all 16"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--row-timeout",
        type=int,
        default=None,
        help="rehearsal-only override of the level's row budget (to exercise the retry pass)",
    )
    ap.add_argument("--retry-timeout", type=int, default=RETRY_TIMEOUT_S)
    ap.add_argument(
        "--rehearsal",
        action="store_true",
        help="a dry run: outputs are named REHEARSAL_..., the budget may be overridden, "
        "the tree may be dirty; never a pinned dataset",
    )
    ap.add_argument(
        "--no-load-guard",
        action="store_true",
        help="(rehearsal) skip the idle-machine wait",
    )
    ns = ap.parse_args()
    unknown = [c for c in ns.corpora if c not in ALL_CORPORA]
    if unknown:
        ap.error(f"unknown corpora {unknown}; known: {ALL_CORPORA}")
    if ns.no_load_guard and not ns.rehearsal:
        ap.error("--no-load-guard is a rehearsal-only option")
    run_pinned(
        parse_ladder_level(ns.ladder_level),
        tuple(ns.corpora),
        ns.seed,
        ns.row_timeout,
        ns.retry_timeout,
        rehearsal=ns.rehearsal,
        guard=not ns.no_load_guard,
    )


if __name__ == "__main__":
    main()
