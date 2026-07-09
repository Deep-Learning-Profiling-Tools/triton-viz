"""Per-kernel worker: one LaunchSpec in, one JSONL row out.

Runs INSIDE the per-spec subprocess (see runner.py). Phase order is fixed
and load-bearing: the REAL host compile happens before anything engages the
interpreter (static C2/C3 replay, then the dynamic-mode comparison) — the
reverse order trips the interpreter-patching hazard documented in
core/trace.py.

Verdict mapping for DRB-style scoring (plan S5):
  static ok            -> "race-free"  (terminal = provenance rung)
  static races         -> "race"       (terminal = race-confirmed | races-unclassified)
  static unsupported   -> "abstain"    (terminal = race-unconfirmed | unsupported)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

from evaluation.spec import LaunchSpec


def _host_compile_ttir(spec: LaunchSpec) -> str:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(
        fn=spec.kernel_fn, signature=spec.signature, constexprs=spec.constexprs
    )
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
    return k.asm["ttir"]


def _static_track(spec: LaunchSpec, ttir: str, seed: int) -> dict[str, Any]:
    from types import SimpleNamespace

    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    det = CompiledRaceDetector(confirm_races=True, differential_check=True)
    args = spec.make_args(seed)
    t0 = time.perf_counter()
    det.pre_warmup_callback(spec.kernel_fn, *args, grid=spec.grid, **spec.constexprs)
    det.post_warmup_callback(spec.kernel_fn, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()
    elapsed = time.perf_counter() - t0

    witnesses = [
        {
            "first": rep.first_record.source_location,
            "second": rep.second_record.source_location,
            "race_type": rep.race_type.name,
            "pids": [list(rep.witness_grid_a or ()), list(rep.witness_grid_b or ())],
        }
        for rep in det.last_global_reports
    ]
    # tier-selector detail, recomputed via the public gate (the client does
    # not publish it): lets the T0 stretch show up as a re-run diff.
    t0_gate = None
    try:
        from triton_viz.clients.common.ttir_reader import parse_ttir
        from triton_viz.clients.race_detector.compiled.global_records import (
            t0_linearity_gate,
        )

        t0_gate = bool(t0_linearity_gate(parse_ttir(ttir)))
    except Exception:  # noqa: BLE001
        pass

    return {
        "status": det.last_global_status,
        "provenance": det.last_global_provenance,
        "confirmation": det.last_global_confirmation,
        "reason": det.last_global_reason,
        "n_reports": len(det.last_global_reports),
        "witnesses": witnesses,
        "parse_unsupported": [r for r in det.last_ttir_unsupported if r],
        "differential": det.last_differential,
        "t0_gate": t0_gate,
        "time_s": round(elapsed, 4),
    }


def _dynamic_track(spec: LaunchSpec, seed: int) -> dict[str, Any]:
    import triton_viz
    from triton_viz.clients import RaceDetector

    det = RaceDetector()
    args = spec.make_args(seed)  # fresh tensors; the interpreter mutates them
    t0 = time.perf_counter()
    error = None
    try:
        traced = triton_viz.trace(det)(spec.kernel_fn)
        traced[spec.grid](*args, **spec.constexprs)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    return {
        "status": getattr(det, "last_status", None),
        "reason": getattr(det, "unsupported_reason", None),
        "n_reports": len(getattr(det, "last_reports", []) or []),
        "error": error,
        "time_s": round(elapsed, 4),
    }


def _classify(static: dict[str, Any]) -> tuple[str, str]:
    """(verdict, terminal) from the static track's surfaces."""
    status = static["status"]
    if status == "ok":
        return ("race-free", static["provenance"] or "proved@T1")
    if status == "races":
        if static["confirmation"] == "confirmed":
            return ("race", "race-confirmed")
        return ("race", "races-unclassified")
    if status == "unsupported":
        if "race-unconfirmed" in (static["reason"] or ""):
            return ("abstain", "race-unconfirmed")
        return ("abstain", "unsupported")
    return ("abstain", status or "unknown")


def run_one(spec: LaunchSpec, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": spec.name,
        "pattern": spec.pattern,
        "expected": spec.expected,
        "race_pair": list(spec.race_pair) if spec.race_pair else None,
        "params_note": spec.params_note,
        "grid": list(spec.grid),
        "seed": seed,
    }
    try:
        t0 = time.perf_counter()
        ttir = _host_compile_ttir(spec)
        row["compile_s"] = round(time.perf_counter() - t0, 4)
        row["ttir_sha"] = hashlib.sha256(ttir.encode()).hexdigest()[:16]
    except Exception as e:  # noqa: BLE001
        row.update(
            verdict="error",
            terminal="compile-error",
            harness_error=f"{type(e).__name__}: {e}",
        )
        return row

    try:
        row["static"] = _static_track(spec, ttir, seed)
    except Exception as e:  # noqa: BLE001
        row.update(
            verdict="error",
            terminal="harness-error",
            harness_error=f"static track: {type(e).__name__}: {e}",
        )
        return row

    try:
        row["dynamic"] = _dynamic_track(spec, seed)
    except Exception as e:  # noqa: BLE001
        row["dynamic"] = {"error": f"{type(e).__name__}: {e}"}

    row["verdict"], row["terminal"] = _classify(row["static"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()

    from evaluation.kernels import load

    corpus = load(ns.corpus)
    spec = next(s for s in corpus.specs if s.name == ns.spec)
    row = run_one(spec, ns.seed)
    row["corpus"] = ns.corpus
    with open(ns.out, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main()
