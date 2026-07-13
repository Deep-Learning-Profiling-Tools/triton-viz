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

RESULTS_DIR = Path(__file__).parent / "results"
PER_SPEC_TIMEOUT_S = 180


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
    }


def _run_one(spec, corpus_name: str, seed: int, timeout: int, mutate: bool) -> dict:
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    cmd = [
        sys.executable, "-m", "evaluation.harness",
        "--corpus", corpus_name, "--spec", spec.name,
        "--seed", str(seed), "--out", tmp,
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
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    print(f"  {spec.name:40s} {row.get('terminal', '?'):20s} {row['wall_s']}s")
    return row


def run_corpus(
    corpus_name: str,
    only: str | None,
    seed: int,
    timeout: int,
    mutate: bool = False,
    jobs: int = 1,
) -> Path:
    from evaluation.kernels import load

    corpus = load(corpus_name)
    specs = [s for s in corpus.specs if only is None or s.name == only]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{corpus_name}.jsonl"

    header = {
        "header": True,
        "corpus": corpus_name,
        "seed": seed,
        **_versions(),
        **corpus.provenance,
    }
    print(f"[runner] {corpus_name}: {len(specs)} specs -> {out_path} (jobs={jobs})")

    if jobs == 1:
        rows = [_run_one(s, corpus_name, seed, timeout, mutate) for s in specs]
    else:
        # rows are subprocess-isolated, so concurrency only affects wall_s
        # (near-watchdog rows can flip to timeout under load — keep the
        # definitive paper sweeps at jobs=1); output order stays spec order
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            rows = list(
                ex.map(lambda s: _run_one(s, corpus_name, seed, timeout, mutate), specs)
            )

    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=PER_SPEC_TIMEOUT_S)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument(
        "--mutate",
        action="store_true",
        help="mutation-sensitivity mode: pid-pin / sem-relax / atomic-to-"
        "store mutants on every proved row (static solver only)",
    )
    ns = ap.parse_args()

    out = run_corpus(
        ns.corpus, ns.only, ns.seed, ns.timeout, mutate=ns.mutate, jobs=ns.jobs
    )
    if not ns.no_report:
        from evaluation.report import render

        md = render([out])
        md_path = RESULTS_DIR / "RESULTS.md"
        md_path.write_text(md)
        print(f"[runner] report -> {md_path}")


if __name__ == "__main__":
    main()
