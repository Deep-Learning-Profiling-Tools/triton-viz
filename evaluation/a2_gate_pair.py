#!/usr/bin/env python
"""Run the A2 gate against one compiler's litmus artifacts (experiment S4).

Compiles the tests/golden/a2gate litmus kernels with the GIVEN python's
triton (a subprocess, so any venv works), then runs the barrier-coverage
gate in-process and prints one verdict line per kernel.

The regression pair for triton PR #10816 ("[BACKEND] Insert CTA
barriers for atomic memory semantics", merged 2026-07-10):

  * pre-fix compiler (the PR's parent 7aab98ee, or the corpus pin
    3.6.0, which also predates the fix): every non-relaxed atomic is
    uncovered -> VIOLATION.
  * post-fix compiler (the merge commit c57bbbd8): every obligation is
    barrier-covered -> verified.

Usage:
  a2_gate_pair.py --python /path/to/venv/bin/python [--keep DIR]

Exit code: 0 when every kernel verdict matches --expect (default: just
print), 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GEN = os.path.join(REPO, "tests", "golden", "a2gate", "generate_golden.py")

sys.path.insert(0, REPO)

from triton_viz.clients.race_detector.compiled.ptx_gate import check_gate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", required=True, help="venv python whose triton compiles")
    ap.add_argument("--keep", default="", help="keep artifacts in this dir")
    ap.add_argument(
        "--expect",
        default="",
        choices=["", "verified", "violation"],
        help="assert every kernel verdict equals this",
    )
    ns = ap.parse_args()

    outdir = ns.keep or tempfile.mkdtemp(prefix="a2gate_")
    r = subprocess.run([ns.python, GEN, outdir], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        print(f"FAILED to compile litmus kernels with {ns.python}")
        return 1
    version = [ln for ln in r.stdout.splitlines() if ln.startswith("triton ")]
    print(version[0] if version else "triton ?", f"({ns.python})")

    ok = True
    for tag in ("a2_sems", "a2_cas"):
        with open(os.path.join(outdir, f"{tag}.ttir")) as f:
            ttir = f.read()
        with open(os.path.join(outdir, f"{tag}.ptx")) as f:
            ptx = f.read()
        res = check_gate(ttir, ptx, tag)
        print(
            f"  {tag}: {res.status}"
            + (f" — {res.reason}" if res.reason else "")
            + (f" ({len(res.reports)} uncovered side(s))" if res.reports else "")
        )
        for rep in res.reports:
            print(f"    {rep}")
        if ns.expect and res.status != ns.expect:
            ok = False
    if ns.expect:
        print("PAIR_CHECK", "OK" if ok else "MISMATCH", f"(expected {ns.expect})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
