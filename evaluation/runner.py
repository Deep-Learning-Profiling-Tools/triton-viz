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
    }


def run_corpus(
    corpus_name: str, only: str | None, seed: int, timeout: int, mutate: bool = False
) -> Path:
    from evaluation.kernels import load

    corpus = load(corpus_name)
    specs = [s for s in corpus.specs if only is None or s.name == only]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{corpus_name}.jsonl"

    rows: list[dict] = []
    header = {"header": True, "corpus": corpus_name, "seed": seed, **_versions()}
    print(f"[runner] {corpus_name}: {len(specs)} specs -> {out_path}")

    for spec in specs:
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
        rows.append(row)
        print(f"  {spec.name:40s} {row.get('terminal', '?'):20s} {row['wall_s']}s")

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
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument(
        "--mutate",
        action="store_true",
        help="mutation-sensitivity mode: pid-pin / sem-relax / atomic-to-"
        "store mutants on every proved row (static solver only)",
    )
    ns = ap.parse_args()

    out = run_corpus(ns.corpus, ns.only, ns.seed, ns.timeout, mutate=ns.mutate)
    if not ns.no_report:
        from evaluation.report import render

        md = render([out])
        md_path = RESULTS_DIR / "RESULTS.md"
        md_path.write_text(md)
        print(f"[runner] report -> {md_path}")


if __name__ == "__main__":
    main()
