"""Regenerate the A2-gate golden artifacts (ttir + ptx per kernel).

Compiles the two litmus kernels below for sm_89 through the ASTSource
path (torch-free) and writes ``<tag>.ttir`` / ``<tag>.ptx`` next to this
file. The kernels live HERE so the ``.loc``/``.file`` entries in the
goldens point at this checked-in file rather than a temp path.

The goldens were generated under the corpus pin triton 3.6.0, which
PREDATES triton-lang/triton PR #10816 ("[BACKEND] Insert CTA barriers
for atomic memory semantics", merged 2026-07-10): their PTX carries NO
ordering barriers around the non-relaxed atomics, which is exactly the
A2-class defect the gate exists to catch. Regenerating under a post-fix
triton produces barrier-covered PTX and flips the expectations in
tests/unit/test_ptx_gate.py — regenerate only together with those.

Usage: .venv/bin/python tests/golden/a2gate/generate_golden.py [outdir]

With an ``outdir`` argument the artifacts land there instead of next to
this file: that is how ``evaluation/a2_gate_pair.py`` reuses these
kernels to compare compilers without touching the goldens.
"""

import os
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def a2_sems_kernel(p_ptr, q_ptr):
    tl.atomic_add(p_ptr, 1, sem="relaxed", scope="gpu")
    tl.atomic_add(p_ptr, 1, sem="acquire", scope="gpu")
    tl.atomic_add(q_ptr, 1, sem="release", scope="gpu")
    tl.atomic_add(q_ptr, 1, sem="acq_rel", scope="gpu")


@triton.jit
def a2_cas_kernel(p_ptr, out_ptr):
    old = tl.atomic_cas(p_ptr, 0, 1, sem="acq_rel", scope="gpu")
    tl.store(out_ptr, old)


def main() -> int:
    here = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(here, exist_ok=True)
    target = GPUTarget("cuda", 89, 32)
    for tag, fn, sig in [
        ("a2_sems", a2_sems_kernel, {"p_ptr": "*i32", "q_ptr": "*i32"}),
        ("a2_cas", a2_cas_kernel, {"p_ptr": "*i32", "out_ptr": "*i32"}),
    ]:
        src = ASTSource(fn=fn, signature=sig, constexprs={}, attrs={})
        k = triton.compile(src, target=target, options={"num_warps": 4})
        for ext in ("ttir", "ptx"):
            path = os.path.join(here, f"{tag}.{ext}")
            with open(path, "w") as f:
                f.write(k.asm[ext])
            print(f"wrote {path}")
        ptx = k.asm["ptx"]
        print(f"[{tag}] bar.sync count: {ptx.count('bar.sync')}")
    print("triton", triton.__version__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
