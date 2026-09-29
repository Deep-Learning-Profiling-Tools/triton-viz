"""Bulk-run the TTIR walker over a directory of ``.ttir`` files and summarise.

A D10b conformance aid, not a test: every file the MLIR parser accepts must
walk with 0 misalignments. Point it at a Triton cache (the default,
``~/.triton/cache``) to cover real compiled kernels, not only the curated
goldens. Files are only read.

    python tools/ir_bulk_conformance.py [ROOT ...] [--jobs N] [--any-version]
        [--jsonl OUT] [--limit N] [--show N]

The walk uses the installed Triton's printer table
(``tilelens.ir._mlir_walk.PRINTERS``); a release without one is reported and
nothing is walked. By default only cache entries whose metadata names the
installed Triton version are walked (the table is that release's printer);
files with no metadata at all (a plain directory of .ttir files) are always
walked. Texts are deduplicated by sha256. Walks run in worker subprocesses,
so a crash inside the MLIR bindings loses one chunk, which the summary
reports.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cache_version(ttir_path: str) -> str | None | bool:
    """The Triton version recorded next to a cache entry: a version string,
    None when the metadata has no version, False when there is no metadata."""
    found = False
    for j in glob.glob(os.path.join(os.path.dirname(ttir_path), "*.json")):
        if os.path.basename(j).startswith("__grp__"):
            continue
        found = True
        try:
            with open(j) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict) and meta.get("triton_version"):
            return str(meta["triton_version"])
    return None if found else False


def collect(
    roots: list[str], version: str | None
) -> tuple[list[str], collections.Counter]:
    counts: collections.Counter = collections.Counter()
    seen: set[bytes] = set()
    paths: list[str] = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith(".ttir"):
                    continue
                p = os.path.join(dirpath, fn)
                counts["files"] += 1
                if version is not None:
                    v = _cache_version(p)
                    if v is not False and v != version:
                        counts["skipped: other or unrecorded Triton version"] += 1
                        continue
                try:
                    with open(p, "rb") as f:
                        h = hashlib.sha256(f.read()).digest()
                except OSError:
                    counts["unreadable"] += 1
                    continue
                if h in seen:
                    counts["duplicate texts"] += 1
                    continue
                seen.add(h)
                paths.append(p)
    return paths, counts


def worker() -> None:
    """Walk each path read from stdin; print one JSON line per file."""
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, REPO)
    from tilelens.ir import _mlir_walk as W

    for p in sys.stdin.read().splitlines():
        if not p:
            continue
        row: dict = {"path": p}
        t0 = time.perf_counter()
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
            m = W._walk(text)  # uncached: every file is distinct anyway
            row.update(status="ok", stats=dict(m.stats))
        except W.MisalignedModule as e:
            row.update(
                status="misaligned", problems=list(e.problems[:20]), line_no=e.line_no
            )
        except W.ModuleParseError as e:
            row.update(
                status="parse-error", problems=[e.diagnostic[:300]], line_no=e.line_no
            )
        except W.UnknownTritonRelease as e:
            row.update(status="unknown-release", problems=[e.message])
        except Exception as e:  # noqa: BLE001  (an escape is a walker bug: report it)
            row.update(
                status="exception", problems=[f"{type(e).__name__}: {str(e)[:300]}"]
            )
        row["ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        print(json.dumps(row), flush=True)


def run_chunk(chunk: list[str], timeout: float) -> list[dict]:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker"]
    try:
        r = subprocess.run(
            cmd, input="\n".join(chunk), capture_output=True, text=True, timeout=timeout
        )
        out, rc, err = r.stdout, str(r.returncode), r.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = "timeout", ""
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    done = {r["path"] for r in rows}
    for p in chunk:
        if p not in done:
            rows.append(
                {
                    "path": p,
                    "status": "worker-lost",
                    "problems": [f"worker rc={rc}: {err.strip()[-300:]}"],
                }
            )
    return rows


def _category(problem: str) -> str:
    q = re.sub(r"^line \d+: ", "", problem)
    q = re.sub(r"\(\[.*|\(\['.*", "(...)", q)
    q = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", q)
    q = re.sub(r"\d+", "N", q)
    return q[:160]


def summarize(rows: list[dict], counts: collections.Counter, show: int) -> None:
    by = collections.Counter(r["status"] for r in rows)
    print("input:", dict(counts))
    print("walked:", len(rows), dict(by))
    parsed = by["ok"] + by["misaligned"]
    if parsed:
        rate = by["misaligned"] / parsed
        print(
            f"misalignment rate: {by['misaligned']}/{parsed} parsed texts = {rate:.4%}"
        )
    for status in (
        "misaligned",
        "parse-error",
        "unknown-release",
        "exception",
        "worker-lost",
    ):
        bad = [r for r in rows if r["status"] == status]
        if not bad:
            continue
        cats: collections.Counter[str] = collections.Counter()
        example: dict[str, str] = {}
        for r in bad:
            c = _category(r["problems"][0]) if r.get("problems") else "?"
            cats[c] += 1
            example.setdefault(c, r["path"])
        print(f"\n-- {status}: {len(bad)} files, first problem by category")
        for c, n in cats.most_common(show):
            print(f"{n:7d}  {c}\n         e.g. {example[c]}")
    tot: collections.Counter = collections.Counter()
    for r in rows:
        if r["status"] == "ok":
            tot.update(r["stats"])
    if tot:
        print("\nchecked over aligned texts:", dict(tot))
    ms = sorted(r["ms"] for r in rows if "ms" in r)
    if ms:
        q = lambda f: ms[min(len(ms) - 1, int(f * len(ms)))]  # noqa: E731
        print(
            f"walk time ms: median {q(0.5):.2f}  p99 {q(0.99):.2f}  max {ms[-1]:.2f}  total {sum(ms) / 1e3:.1f}s"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="*", default=[os.path.expanduser("~/.triton/cache")])
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument(
        "--chunk", type=int, default=200, help="files per worker subprocess"
    )
    ap.add_argument(
        "--timeout", type=float, default=900.0, help="seconds per worker subprocess"
    )
    ap.add_argument(
        "--any-version",
        action="store_true",
        help="walk cache entries of every Triton version",
    )
    ap.add_argument("--limit", type=int, default=0, help="walk at most N texts")
    ap.add_argument("--jsonl", help="write one JSON row per walked file here")
    ap.add_argument("--show", type=int, default=15, help="categories listed per status")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker()
        return 0
    sys.path.insert(0, REPO)
    from tilelens.ir import _mlir_walk as W

    try:
        table = W.printer()
    except W.UnknownTritonRelease as e:
        print(e.message, file=sys.stderr)
        return 2
    version = None
    if not args.any_version:
        import triton

        version = triton.__version__
    paths, counts = collect(args.roots, version)
    if args.limit:
        paths = paths[: args.limit]
    print(
        f"{len(paths)} distinct TTIR texts (Triton {version or 'any version'}; "
        f"the Triton {table.release} printer table)",
        file=sys.stderr,
    )
    chunks = [paths[i : i + args.chunk] for i in range(0, len(paths), args.chunk)]
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for got in ex.map(lambda c: run_chunk(c, args.timeout), chunks):
            rows += got
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    summarize(rows, counts, args.show)
    return 0 if all(r["status"] in ("ok", "parse-error") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
