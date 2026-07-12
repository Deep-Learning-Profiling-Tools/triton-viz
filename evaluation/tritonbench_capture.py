"""One-time launch capture for the vendored TritonBench_G_v1 corpus.

Each vendored file is a standalone operator whose test block executes AT
IMPORT TIME on CUDA, so this step needs a GPU machine — it runs every
file in a subprocess with a ``JITFunction.run`` hook that records, per
(file, kernel), the FIRST real launch: the full name→value binding split
into runtime args and constexprs, tensor descriptors (shape / dtype /
init class / contiguity / alias group), exact scalars, and the resolved
grid. The result is ``kernels/tritonbench_g_specs.json``; the corpus
module rebuilds CPU launches from it on any machine (no GPU, no test
blocks — only each file's pre-separator kernel section is executed).

Reconstruction is by-descriptor, not by-value (capture_common.py): float
tensors are seeded randn/rand/zeros, int tensors randint over the
OBSERVED value range (so index tensors stay in-bounds), and small
int/bool tensors carry an exact VALUE SNAPSHOT so value-coupled inputs
(monotone offsets, permutation tables, disjointness-keeping masks)
rebuild faithfully. Aliased pointer args (in-place ops) are rebuilt from
one tensor and the spec is marked ``aliased``. Non-contiguous tensors
are recorded and the file is SKIPPED with a reason — stride scalars
captured from a strided layout would misdescribe a contiguous rebuild.

Usage (GPU machine):
    uv run python -m evaluation.tritonbench_capture            # all files
    uv run python -m evaluation.tritonbench_capture --one <path> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "kernels" / "tritonbench_g_v1"
SPECS_PATH = Path(__file__).parent / "kernels" / "tritonbench_g_specs.json"
SEPARATOR = "#" * 100  # files use a ~146-char run; prefix match is enough
PER_FILE_TIMEOUT_S = 300


def _capture_one(path: Path) -> dict:
    import triton

    from evaluation.capture_common import LaunchRecorder

    recorder = LaunchRecorder()
    src = path.read_text()
    error = None
    with recorder.hooked():
        try:
            exec(  # noqa: S102 — trusted vendored corpus
                compile(src, str(path), "exec"), {"__name__": f"tb_{path.stem}"}
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    return {
        "file": path.name,
        "error": error,
        "kernels": recorder.captured,
        "skipped_kernels": recorder.skipped,
        "triton": triton.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        result = _capture_one(args.one)
        args.out.write_text(json.dumps(result, indent=1))
        return

    files = sorted(VENDOR_DIR.glob("*.py"))
    merged: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for i, f in enumerate(files, 1):
        # private per-run temp file: /tmp is shared and sticky, a fixed
        # path can collide with a concurrent sweep or another user's
        # stale file and merge records under the wrong run's provenance
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix=f"tb_capture_{f.stem}_")
        os.close(fd)
        out = Path(tmp)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.tritonbench_capture",
                    "--one",
                    str(f),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=PER_FILE_TIMEOUT_S,
                cwd=Path(__file__).parent.parent,
            )
            if proc.returncode != 0:
                failures[f.name] = (proc.stderr or "").strip()[-300:]
                print(f"[{i}/{len(files)}] {f.name}: CRASH")
                continue
            result = json.loads(out.read_text())
        except subprocess.TimeoutExpired:
            failures[f.name] = f"timeout after {PER_FILE_TIMEOUT_S}s"
            print(f"[{i}/{len(files)}] {f.name}: TIMEOUT")
            continue
        except (OSError, json.JSONDecodeError) as exc:
            failures[f.name] = f"capture output unreadable: {exc}"
            print(f"[{i}/{len(files)}] {f.name}: UNREADABLE")
            continue
        finally:
            out.unlink(missing_ok=True)
        if result["error"] and not result["kernels"]:
            failures[f.name] = result["error"][:300]
            print(f"[{i}/{len(files)}] {f.name}: ERROR ({result['error'][:80]})")
            continue
        merged[f.name] = result
        n = len(result["kernels"])
        note = (
            f" (+error after capture: {result['error'][:60]})"
            if result["error"]
            else ""
        )
        print(f"[{i}/{len(files)}] {f.name}: {n} kernel(s){note}")

    payload = {
        "upstream": "https://github.com/thunlp/TritonBench data/TritonBench_G_v1",
        "upstream_commit": "603e28a5050e8c268f6883a69709d477a272d49a",
        "files": merged,
        "capture_failures": failures,
    }
    SPECS_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    total = sum(len(r["kernels"]) for r in merged.values())
    print(
        f"\ncaptured {total} launches from {len(merged)}/{len(files)} files "
        f"({len(failures)} failures) -> {SPECS_PATH}"
    )


if __name__ == "__main__":
    main()
