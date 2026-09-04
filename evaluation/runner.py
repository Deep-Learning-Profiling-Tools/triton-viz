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

from triton_viz.clients.race_detector.ladder import (
    LADDER_LEVEL_NAMES,
    LadderLevel,
    parse_ladder_level,
)

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
) -> dict:
    """The JSONL header: detector commit, package versions, corpus
    provenance, and the ladder-depth stamp (no dataset may mix levels
    unnoticed: a paper or CI deployment quotes one level of one run)."""
    return {
        "header": True,
        "corpus": corpus_name,
        "seed": seed,
        "ladder_level": ladder_level.name,
        **_versions(),
        **provenance,
    }


def run_corpus(
    corpus_name: str,
    only: str | None,
    seed: int,
    timeout: int,
    mutate: bool = False,
    jobs: int = 1,
    ladder_level: LadderLevel = LadderLevel.L0,
    only_names: "set[str] | None" = None,
    out_suffix: "str | None" = None,
) -> Path:
    from evaluation.kernels import load

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

    header = results_header(corpus_name, seed, corpus.provenance, ladder_level)
    print(
        f"[runner] {corpus_name}: {len(specs)} specs -> {out_path} "
        f"(jobs={jobs}, ladder {ladder_level.name})"
    )

    def _one(s):
        return _run_one(s, corpus_name, seed, timeout, mutate, ladder_level)

    if jobs == 1:
        rows = [_one(s) for s in specs]
    else:
        # rows are subprocess-isolated, so concurrency only affects wall_s
        # (near-watchdog rows can flip to timeout under load — keep the
        # definitive paper sweeps at jobs=1); output order stays spec order
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(_one, specs))

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
    ap.add_argument("--timeout", type=int, default=PER_SPEC_TIMEOUT_S)
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
    )
    if not ns.no_report:
        from evaluation.report import render

        md = render([out])
        md_path = RESULTS_DIR / "RESULTS.md"
        md_path.write_text(md)
        print(f"[runner] report -> {md_path}")


if __name__ == "__main__":
    main()
