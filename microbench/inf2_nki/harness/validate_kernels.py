"""Correctness validation for Inf2 NKI microbench kernels via ``nki.simulate_kernel``.

Why this exists: ``nki.benchmark`` runs kernels with zeroed inputs (its result
rows report ``input_type: ZERO``) and only reports timing, so a kernel that
compiles and "runs ok" can still be numerically wrong. In addition, on some
hosts device->host readback of ``nki.baremetal`` outputs returns zeros, so
baremetal return values are not a reliable oracle either. ``nki.simulate_kernel``
executes the traced kernel on CPU with real inputs and returns real outputs,
which makes it the correct place to assert kernel *logic* is right.

This module validates data-dependent kernels whose correctness is not obvious
from "it ran": pointer chasing and the Static DMA SBUF transpose. Pure
bandwidth/streaming kernels have trivial dataflow and are covered by shape/work
metadata tests.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from typing import Any

import numpy as np
from neuronxcc import nki

from microbench.inf2_nki.common.inputs import make_pointer_ring, pointer_ring_walk
from microbench.inf2_nki.tests.latency_pointer_chase.kernels import pointer_chase_factory
from microbench.inf2_nki.tests.static_dma.kernels import static_dma_scatter_factory


def validate_pointer_chase(ring_length: int = 256, stride: int = 17, repeats: tuple[int, ...] = (1, 2, 4, 8, 16)) -> dict[str, Any]:
    """Simulate the pointer-chase kernel and compare against a CPU ring walk."""
    ring = make_pointer_ring(ring_length, stride)
    cases: list[dict[str, Any]] = []
    all_ok = True
    for repeat in repeats:
        kernel, _shapes, _grid = pointer_chase_factory(
            ring_length=ring_length, repeat=repeat, stride=stride, mode="hbm_index_chain", dtype_name="uint32"
        )
        # nki.simulate_kernel prints a "Neuron NKI - Kernel call" banner to
        # stdout; silence it so --json output stays machine-parseable.
        with contextlib.redirect_stdout(io.StringIO()):
            out = nki.simulate_kernel(kernel, ring)
        device = int(np.asarray(out).reshape(-1)[0])
        expected = pointer_ring_walk(ring, repeat)
        ok = device == expected
        all_ok = all_ok and ok
        cases.append({"repeat": repeat, "device": device, "expected": expected, "match": ok})
    return {"kernel": "pointer_chase", "ring_length": ring_length, "stride": stride, "ok": all_ok, "cases": cases}


def validate_static_dma_scatter(p: int = 8, x: int = 4, y: int = 8) -> dict[str, Any]:
    """Simulate the scalar SBUF scatter and compare with a NumPy transpose."""
    kernel, _shapes, _grid = static_dma_scatter_factory(
        p=p, x=x, y=y, mode="sbuf_transpose_scatter", dtype_name="float32"
    )
    src = np.arange(p * x * y, dtype=np.float32).reshape(p, x * y)
    expected = src.reshape(p, x, y).transpose(0, 2, 1).reshape(p, x * y)
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(nki.simulate_kernel(kernel, src))
    return {
        "kernel": "static_dma_scatter",
        "p": p,
        "x": x,
        "y": y,
        "ok": bool(np.array_equal(out, expected)),
    }


def run_all() -> dict[str, Any]:
    results = [validate_pointer_chase(), validate_static_dma_scatter()]
    return {"ok": all(r["ok"] for r in results), "validations": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = run_all()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for validation in report["validations"]:
            status = "OK" if validation["ok"] else "FAIL"
            details = ", ".join(
                f"{key}={value}"
                for key, value in validation.items()
                if key not in {"kernel", "ok", "cases"}
            )
            print(f"[{status}] {validation['kernel']} ({details})")
            for case in validation.get("cases", []):
                mark = "ok" if case["match"] else "MISMATCH"
                print(f"    repeat={case['repeat']:>3}  device={case['device']:>6}  expected={case['expected']:>6}  {mark}")
        print("PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
