"""Sweep driver: one subprocess per LaunchSpec, hard timeout, JSONL out.

Usage:
    uv run python -m evaluation.runner --corpus golden_smoke
    uv run python -m evaluation.runner --corpus golden_smoke --only smoke_add_no
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from triton_viz.clients.race_detector.ladder import (
    LADDER_LEVEL_NAMES,
    LadderLevel,
    parse_ladder_level,
)

RESULTS_DIR = Path(__file__).parent / "results"
# The per-row subprocess budget is part of the evaluation protocol and
# therefore level-dependent (provenance: stamped into the header as
# row_timeout_s). L0 keeps the paper's 180 s; L1 (Hao, 2026-09-04) runs
# the concrete-enumeration rung after both symbolic tracks and gets 200 s.
PER_SPEC_TIMEOUT_S = 180
PER_SPEC_TIMEOUT_L1_S = 200


def row_timeout_s(ladder_level: LadderLevel = LadderLevel.L0) -> int:
    return (
        PER_SPEC_TIMEOUT_L1_S if ladder_level >= LadderLevel.L1 else PER_SPEC_TIMEOUT_S
    )


# Upstream commits of the liger-kernel PyPI releases we evaluate against
# (PyPI wheels embed no VCS info). Each entry is the commit the GitHub
# release tag points to, resolved via
# api.github.com/repos/linkedin/Liger-Kernel/git/refs/tags/v<version>.
_LIGER_RELEASE_COMMITS = {
    "0.8.0": "c4b16d43f9d8f69068e6a15bd879dfc6a63b2449",  # tag v0.8.0
}


# Same for fla-core (the flash-linear-attention kernel package); tags at
# github.com/fla-org/flash-linear-attention.
_FLA_RELEASE_COMMITS = {
    "0.5.1": "2e38c1fab332174d056928feaf29f8c5fd5ac550",  # tag v0.5.1
}


def _package_provenance(package: str, key: str, release_commits: dict) -> dict:
    """Version + best-effort git commit of an AS-INSTALLED corpus package
    for the results fingerprint (the corpus analyzes whatever is
    installed, so the artifact record must pin exactly which source that
    was). The commit comes from pip's direct_url.json for git installs,
    else from the release→tag-commit table; unknown releases record None
    — extend the table rather than guess."""
    from importlib import metadata

    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return {}
    commit = None
    raw = dist.read_text("direct_url.json")
    if raw:
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    if commit is None:
        commit = release_commits.get(dist.version)
    return {key: dist.version, f"{key}_commit": commit}


def _liger_provenance() -> dict:
    return _package_provenance("liger-kernel", "liger_kernel", _LIGER_RELEASE_COMMITS)


def _fla_provenance() -> dict:
    return _package_provenance("fla-core", "fla_core", _FLA_RELEASE_COMMITS)


def _flagattn_provenance() -> dict:
    # installed from git (no PyPI release), so the commit always comes
    # from pip's direct_url.json — no release table needed
    return _package_provenance("flag_attn", "flag_attn", {})


def _flaggems_provenance() -> dict:
    # git-pinned (PyPI lags upstream by months); direct_url.json carries
    # the commit
    return _package_provenance("flag_gems", "flag_gems", {})


def _torchao_provenance() -> dict:
    # git-pinned USE_CPP=0 install (the Triton kernels are pure Python,
    # so the C++ extension is skipped); direct_url.json carries the
    # commit and the version string itself embeds it (0.18.0+git<sha>)
    return _package_provenance("torchao", "torchao", {})


def _tritonbench_meta_provenance() -> dict:
    # meta-pytorch/tritonbench, git-pinned; dist version is a constant
    # 0.0.1, so the direct_url.json commit is the real pin (the corpus
    # module hard-checks it)
    return _package_provenance("tritonbench", "tritonbench_meta", {})


def _tilebench_provenance() -> dict:
    # local git checkout (TileBench has no packaging metadata); the
    # corpus module pins the HEAD commit and refuses tracked-dirty trees
    from evaluation.tilebench_capture import TILEBENCH_ROOT

    if not TILEBENCH_ROOT.is_dir():
        return {}
    head = subprocess.run(
        ["git", "-C", str(TILEBENCH_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"tilebench": head, "tilebench_commit": head} if head else {}


def _versions() -> dict:
    import numpy
    import torch
    import triton
    import z3

    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    ).stdout.strip()
    return {
        "triton": triton.__version__,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "z3": z3.get_version_string(),
        "commit": git,
        **_liger_provenance(),
        **_fla_provenance(),
        **_flagattn_provenance(),
        **_flaggems_provenance(),
        **_torchao_provenance(),
        **_tritonbench_meta_provenance(),
        **_tilebench_provenance(),
    }


# ── process reuse: DEBUGGING ONLY (Hao, 2026-09-05) ─────────────────
#
# One served worker runs many rows, saving the 2-3 s of interpreter and
# corpus start-up per row. It is FORBIDDEN for a pinned rerun and for any
# dataset the paper quotes: the paper's per-row wall times are per-row
# subprocess walls (start-up included), so a reused-worker dataset is on
# a different basis and its rows must never be merged with, or compared
# against, protocol rows. The runner makes a debugging dataset
# unmistakable: the output file carries the ``_debug-reuse`` suffix, the
# header carries ``worker_reuse.debugging_only``, a banner is printed at
# start, and ``assert_protocol_dataset`` (for the pinned driver) refuses
# such a file.
WORKER_ROWS = 50  # recycle a worker after this many rows (leak/memory bound)
WORKER_RSS_MB = 8192  # ...or when its resident set exceeds this
DEBUG_REUSE_SUFFIX = "_debug-reuse"
DEBUG_REUSE_BANNER = (
    "[runner] DEBUGGING ONLY: rows are served by reused worker processes; "
    "wall times exclude process start-up. This dataset is NOT on the "
    "paper's protocol and must never enter a pinned rerun."
)


def is_protocol_dataset(header: dict) -> bool:
    """True when a results header describes a per-row-subprocess run
    (the paper's protocol); False for a debugging (worker-reuse) run."""
    return not header.get("worker_reuse")


def assert_protocol_dataset(path: Path) -> dict:
    """Read a results file's header and refuse a debugging dataset. The
    pinned-rerun driver calls this on every input before merging."""
    with open(path) as f:
        first = f.readline()
    header = json.loads(first) if first.strip() else {}
    if not header.get("header"):
        raise ValueError(f"{path}: no results header")
    if not is_protocol_dataset(header):
        raise ValueError(
            f"{path}: a DEBUGGING dataset (worker reuse, wall times exclude "
            "start-up); forbidden in a pinned rerun or any quoted number"
        )
    return header


class _Worker:
    """A ``python -m evaluation.harness --serve`` process. ``run`` sends
    one request and waits for the sentinel line under the per-row
    budget; a silent worker is killed (the row is a timeout), a dead
    one is reported as a crash; both are respawned by the caller.
    stderr goes to a log file (never a pipe: no deadlock, and the tail
    is readable for a crash report)."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.log: Any = None
        self.rows = 0
        self.rss_mb = 0.0

    def start(self) -> None:
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "evaluation.harness", "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent.parent,
        )
        self.rows = 0
        self.rss_mb = 0.0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            self.proc = None
        if self.log is not None:
            try:
                self.log.close()
            except Exception:  # noqa: BLE001
                pass
            self.log = None

    def stderr_tail(self, n: int = 500) -> str:
        try:
            if self.log is not None:
                self.log.flush()
            return self.log_path.read_text()[-n:]
        except OSError:
            return ""

    def run(self, request: dict, timeout: float) -> tuple[str, str]:
        """('ok', status) / ('error', message) / ('timeout', '') /
        ('crash', stderr tail); the worker is stopped on the last two."""
        import select

        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            tail = self.stderr_tail()
            self.stop()
            return ("crash", tail)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                return ("timeout", "")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                if self.proc.poll() is not None:
                    tail = self.stderr_tail()
                    self.stop()
                    return ("crash", tail)
                continue
            line = self.proc.stdout.readline()
            if line == "":  # EOF: the worker died
                tail = self.stderr_tail()
                self.stop()
                return ("crash", tail)
            if not line.startswith("@@ROW@@"):
                continue  # a stray print from a kernel or a client
            self.rows += 1
            body = line[len("@@ROW@@") :].strip()
            if body.startswith("ok"):
                for tok in body.split():
                    if tok.startswith("rss_mb="):
                        self.rss_mb = float(tok[len("rss_mb=") :])
                return ("ok", body)
            return ("error", body[len("error") :].strip())

    def should_recycle(self, worker_rows: int, rss_limit_mb: float) -> bool:
        return self.rows >= worker_rows or self.rss_mb > rss_limit_mb


def _run_one_reused(
    worker: _Worker,
    spec,
    corpus_name: str,
    seed: int,
    timeout: int,
    mutate: bool,
    ladder_level: LadderLevel,
    probe: str | None = None,
) -> dict:
    """The served-worker counterpart of ``_run_one``: same row shapes and
    the same terminals for a dead or silent worker (``crash`` /
    ``timeout``); ``wall_s`` excludes process start-up by construction."""
    t0 = time.perf_counter()
    if not worker.alive():
        worker.start()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    request = {
        "corpus": corpus_name,
        "spec": spec.name,
        "seed": seed,
        "mutate": mutate,
        "ladder_level": ladder_level.name,
        "out": tmp,
    }
    if probe:
        request["probe"] = probe
    base = {
        "name": spec.name,
        "corpus": corpus_name,
        "expected": spec.expected,
        "pattern": spec.pattern,
    }
    try:
        kind, detail = worker.run(request, timeout)
        if kind == "ok" and os.path.getsize(tmp) > 0:
            with open(tmp) as f:
                row = json.load(f)
        elif kind == "timeout":
            row = {
                **base,
                "verdict": "error",
                "terminal": "timeout",
                "harness_error": f"exceeded {timeout}s",
            }
        elif kind == "crash":
            row = {
                **base,
                "verdict": "error",
                "terminal": "crash",
                "harness_error": detail[-500:],
            }
        else:
            row = {
                **base,
                "verdict": "error",
                "terminal": "harness-error",
                "harness_error": detail[-500:],
            }
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    row.setdefault("ladder_level", ladder_level.name)
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    print(f"  {spec.name:40s} {row.get('terminal', '?'):20s} {row['wall_s']}s")
    return row


def _run_one(
    spec,
    corpus_name: str,
    seed: int,
    timeout: int,
    mutate: bool,
    ladder_level: LadderLevel = LadderLevel.L0,
) -> dict:
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    cmd = [
        sys.executable, "-m", "evaluation.harness",
        "--corpus", corpus_name, "--spec", spec.name,
        "--seed", str(seed), "--out", tmp,
        "--ladder-level", ladder_level.name,
    ]  # fmt: skip
    if mutate:
        cmd.append("--mutate")
    row: dict
    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if os.path.getsize(tmp) > 0:
            with open(tmp) as f:
                row = json.load(f)
            if proc.returncode != 0:
                row.setdefault("harness_error", proc.stderr[-500:])
        else:
            row = {
                "name": spec.name,
                "corpus": corpus_name,
                "expected": spec.expected,
                "pattern": spec.pattern,
                "verdict": "error",
                "terminal": "crash",
                "harness_error": (proc.stderr or proc.stdout)[-500:],
            }
    except subprocess.TimeoutExpired:
        row = {
            "name": spec.name,
            "corpus": corpus_name,
            "expected": spec.expected,
            "pattern": spec.pattern,
            "verdict": "error",
            "terminal": "timeout",
            "harness_error": f"exceeded {timeout}s",
        }
    finally:
        os.unlink(tmp)
    row.setdefault("ladder_level", ladder_level.name)
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    print(f"  {spec.name:40s} {row.get('terminal', '?'):20s} {row['wall_s']}s")
    return row


def results_header(
    corpus_name: str,
    seed: int,
    provenance: dict,
    ladder_level: LadderLevel = LadderLevel.L0,
    timeout: int | None = None,
    worker_reuse: dict | None = None,
) -> dict:
    """The JSONL header: detector commit, package versions, corpus
    provenance, the ladder-depth stamp and the per-row budget (no
    dataset may mix levels or budgets unnoticed: a paper or CI
    deployment quotes one level of one run)."""
    return {
        "header": True,
        "corpus": corpus_name,
        "seed": seed,
        "ladder_level": ladder_level.name,
        "row_timeout_s": timeout
        if timeout is not None
        else row_timeout_s(ladder_level),
        # process reuse (opt-in): rows served by long-lived workers, whose
        # wall_s excludes process start-up; the two protocols are stamped
        "worker_reuse": worker_reuse or False,
        **_versions(),
        **provenance,
    }


def run_corpus(
    corpus_name: str,
    only: str | None,
    seed: int,
    timeout: int | None = None,
    mutate: bool = False,
    jobs: int = 1,
    ladder_level: LadderLevel = LadderLevel.L0,
    only_names: "set[str] | None" = None,
    out_suffix: "str | None" = None,
    reuse_workers: bool = False,
    worker_rows: int = WORKER_ROWS,
    worker_rss_mb: float = WORKER_RSS_MB,
) -> Path:
    from evaluation.kernels import load

    if timeout is None:
        timeout = row_timeout_s(ladder_level)
    corpus = load(corpus_name)
    specs = [
        s
        for s in corpus.specs
        if (only is None or s.name == only)
        and (only_names is None or s.name in only_names)
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    # One dataset per level: the L0 files keep their names (the paper's
    # numbers), deeper levels get a suffix so a rerun can never overwrite
    # the other level's rows unnoticed.
    # ``out_suffix`` names a subset run (a change-surface slice); it is
    # APPENDED to the level suffix, so no subset run can overwrite a recorded
    # dataset of another level, and no run at all can overwrite the paper's
    # L0 file unless it is a full L0 run.
    level_suffix = "" if ladder_level == LadderLevel.L0 else f"_{ladder_level.name}"
    suffix = level_suffix + (out_suffix or "")
    out_path = RESULTS_DIR / f"{corpus_name}{suffix}.jsonl"

    reuse = (
        {
            "debugging_only": True,
            "rows_per_worker": worker_rows,
            "rss_limit_mb": worker_rss_mb,
        }
        if reuse_workers
        else None
    )
    if reuse_workers:
        # a debugging dataset names itself: never the protocol file
        suffix = suffix + DEBUG_REUSE_SUFFIX
        out_path = RESULTS_DIR / f"{corpus_name}{suffix}.jsonl"
        print(DEBUG_REUSE_BANNER, file=sys.stderr, flush=True)
    header = results_header(
        corpus_name, seed, corpus.provenance, ladder_level, timeout, reuse
    )
    print(
        f"[runner] {corpus_name}: {len(specs)} specs -> {out_path} "
        f"(jobs={jobs}, ladder {ladder_level.name}, {timeout}s per row"
        + (
            f", DEBUGGING: workers reused for {worker_rows} rows)"
            if reuse_workers
            else ")"
        )
    )

    workers: list[_Worker] = []
    if reuse_workers:
        # one served worker per job thread; recycled on the row/RSS bound
        import threading

        local = threading.local()
        log_path = RESULTS_DIR / f"{corpus_name}{suffix}_worker.log"
        log_path.write_text("")
        lock = threading.Lock()

        def _worker() -> _Worker:
            w = getattr(local, "worker", None)
            if w is None or w.should_recycle(worker_rows, worker_rss_mb):
                if w is not None:
                    w.stop()
                w = local.worker = _Worker(log_path)
                with lock:
                    workers.append(w)
            return w

        def _one(s):
            return _run_one_reused(
                _worker(), s, corpus_name, seed, timeout, mutate, ladder_level
            )

    else:

        def _one(s):
            return _run_one(s, corpus_name, seed, timeout, mutate, ladder_level)

    try:
        if jobs == 1:
            rows = [_one(s) for s in specs]
        else:
            # rows are process-isolated, so concurrency only affects wall_s
            # (near-watchdog rows can flip to timeout under load — keep the
            # definitive paper sweeps at jobs=1); output order stays spec order
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=jobs) as ex:
                rows = list(ex.map(_one, specs))
    finally:
        for w in workers:
            w.stop()

    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--only")
    ap.add_argument(
        "--only-file",
        help="run only the specs whose names are listed in this file (one "
        "per line; a change-surface subset)",
    )
    ap.add_argument(
        "--out-suffix",
        help="extra output-name suffix, appended after the level suffix "
        "(<corpus>[_L<n>]<suffix>.jsonl): a subset run names itself so no "
        "recorded dataset is overwritten",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="per-row subprocess budget in seconds (default: 180 at L0, "
        "200 at L1+; stamped into the header)",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument(
        "--mutate",
        action="store_true",
        help="mutation-sensitivity mode: pid-pin / sem-relax / atomic-to-"
        "store mutants on every proved row (static solver only)",
    )
    ap.add_argument(
        "--ladder-level",
        choices=LADDER_LEVEL_NAMES,
        default=LadderLevel.L0.name,
        help="ladder depth for every row of this run: L0 = shipped rungs "
        "only (default; the paper's numbers), L1 = + the concrete "
        "per-instance enumeration rung, L2 = + forked capture (future). "
        "Stamped into the JSONL header and every row.",
    )
    ap.add_argument(
        "--debug-reuse-workers",
        action="store_true",
        help="DEBUGGING ONLY, never for a pinned rerun or a quoted number: "
        "serve rows from long-lived worker processes instead of one "
        "subprocess per row (saves the 2-3 s import + corpus load per row; "
        "the per-row budget and crash containment are kept by the parent). "
        "Wall times then exclude process start-up, so the output is written "
        "to <name>_debug-reuse.jsonl with a debugging_only header stamp and "
        "is refused by the pinned-rerun merge.",
    )
    ap.add_argument(
        "--debug-worker-rows",
        type=int,
        default=WORKER_ROWS,
        help="(debugging) recycle a reused worker after this many rows",
    )
    ns = ap.parse_args()

    only_names = None
    if ns.only_file:
        only_names = {
            ln.strip()
            for ln in Path(ns.only_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        from evaluation.kernels import load as _load

        known = {s.name for s in _load(ns.corpus).specs}
        missing = sorted(only_names - known)
        if missing:
            # A subset run must never silently shrink: a misspelled or
            # renamed spec would otherwise vanish from the change surface.
            raise SystemExit(
                f"--only-file names {len(missing)} spec(s) not in {ns.corpus}: "
                + ", ".join(missing[:5])
            )
    out = run_corpus(
        ns.corpus,
        ns.only,
        ns.seed,
        ns.timeout,
        mutate=ns.mutate,
        jobs=ns.jobs,
        ladder_level=parse_ladder_level(ns.ladder_level),
        only_names=only_names,
        out_suffix=ns.out_suffix,
        reuse_workers=ns.debug_reuse_workers,
        worker_rows=ns.debug_worker_rows,
    )
    if not ns.no_report:
        from evaluation.report import render

        md = render([out])
        md_path = RESULTS_DIR / "RESULTS.md"
        md_path.write_text(md)
        print(f"[runner] report -> {md_path}")


if __name__ == "__main__":
    main()
