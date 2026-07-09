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
import re
import signal
import threading
import time
from contextlib import contextmanager
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
        "assumes_termination": det.last_global_assumes_termination,
        "time_s": round(elapsed, 4),
    }


# The dynamic comparison runs a CONCRETE interpreter: a spin loop whose
# producer block is sequenced after the spinning one never terminates.
# The watchdog turns that into an honest "timeout" status — itself a
# dynamic-comparison data point for await-bearing kernels.
DYNAMIC_TIMEOUT_S = 60


@contextmanager
def _watchdog(seconds: float):
    if (
        not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _fire(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"dynamic track exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _fire)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        # Re-arm an enclosing SIGALRM timer with its remaining time — a
        # nested watchdog must not permanently defuse the outer one.
        if old_timer and old_timer[0] > 0:
            remaining = old_timer[0] - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining), old_timer[1])


def _dynamic_track(spec: LaunchSpec, seed: int) -> dict[str, Any]:
    import triton_viz
    from triton_viz.clients import RaceDetector

    det = RaceDetector()
    args = spec.make_args(seed)  # fresh tensors; the interpreter mutates them
    t0 = time.perf_counter()
    error = None
    timed_out = False
    try:
        traced = triton_viz.trace(det)(spec.kernel_fn)
        with _watchdog(DYNAMIC_TIMEOUT_S):
            traced[spec.grid](*args, **spec.constexprs)
    except TimeoutError as e:
        error = str(e)
        timed_out = True
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    return {
        "status": "timeout" if timed_out else getattr(det, "last_status", None),
        "reason": getattr(det, "unsupported_reason", None),
        "n_reports": len(getattr(det, "last_reports", []) or []),
        "error": error,
        "time_s": round(elapsed, 4),
    }


# ── mutation sensitivity mode (plan S5 build order step 4) ──────────
# Every PROVED row gets its TTIR mutated in ways that PLANT a race the
# proof's key ingredient was suppressing; a proof that survives every
# applicable mutant is a vacuity suspect (or a genuinely degenerate
# launch, e.g. n=0 disabling all accesses — the report lists survivors).

_RE_MUT_PID = re.compile(r"^(\s*)(%[-\w.#]+) = tt\.get_program_id x : i32(.*)$", re.M)
_RE_MUT_RMW = re.compile(
    r"^(\s*)(?:%[-\w.#]+ = )?tt\.atomic_rmw \w+, \w+, \w+, "
    r"(%[-\w.#]+), (%[-\w.#]+), (%[-\w.#]+)\s*:\s*\(([^,]+),.*$",
    re.M,
)


def _mutate_pid_pin(ttir: str) -> str | None:
    """Pin the x program id to 0 (keeping a dead read so the grid axis
    stays symbolic): every per-pid-disjointness proof must flip."""

    def repl(m: re.Match) -> str:
        return (
            f"{m.group(1)}%__mut_dead_pid = tt.get_program_id x : i32{m.group(3)}\n"
            f"{m.group(1)}{m.group(2)} = arith.constant 0 : i32{m.group(3)}"
        )

    new, n = _RE_MUT_PID.subn(repl, ttir, count=1)
    return new if n else None


def _mutate_sem_relax(ttir: str) -> str | None:
    """Drop every release/acquire to relaxed: every synchronization-based
    proof must flip."""
    out, changed = [], False
    for line in ttir.splitlines():
        if "tt.atomic_" in line:
            new = (
                line.replace(" acq_rel,", " relaxed,")
                .replace(" acquire,", " relaxed,")
                .replace(" release,", " relaxed,")
            )
            changed = changed or new != line
            line = new
        out.append(line)
    return "\n".join(out) if changed else None


def _mutate_atomic_to_store(ttir: str) -> str | None:
    """Demote every atomic RMW to a plain store: every atomicity-based
    proof must flip. (The dangling result SSA parses to DataDep — sound.)"""

    def repl(m: re.Match) -> str:
        return (
            f"{m.group(1)}tt.store {m.group(2)}, {m.group(3)}, "
            f"{m.group(4)} : {m.group(5)}"
        )

    new, n = _RE_MUT_RMW.subn(repl, ttir)
    return new if n else None


_MUTANTS = (
    ("pid_pin", _mutate_pid_pin),
    ("sem_relax", _mutate_sem_relax),
    ("atomic_to_store", _mutate_atomic_to_store),
)


def _mutation_track(spec: LaunchSpec, ttir: str, seed: int) -> dict[str, Any]:
    """Static-solver-only verdicts on each applicable mutant (no C2/C3:
    the interpreter would run the UNMUTATED kernel)."""
    from types import SimpleNamespace

    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    results: dict[str, str] = {}
    for name, mutate in _MUTANTS:
        mutant = mutate(ttir)
        if mutant is None:
            results[name] = "n/a"
            continue
        det = CompiledRaceDetector(confirm_races=False, differential_check=False)
        args = spec.make_args(seed)
        det.pre_warmup_callback(
            spec.kernel_fn, *args, grid=spec.grid, **spec.constexprs
        )
        det.post_warmup_callback(spec.kernel_fn, SimpleNamespace(asm={"ttir": mutant}))
        det.finalize()
        results[name] = det.last_global_status
    applicable = [s for s in results.values() if s != "n/a"]
    return {
        "results": results,
        "flipped": any(s == "races" for s in applicable),
        "applicable": len(applicable),
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


def _resolve_race_pair_lines(spec: LaunchSpec) -> list[int | None] | None:
    """Resolve the spec's race_pair NEEDLES to kernel source line numbers
    (witness-level scoring compares them against reported witnesses)."""
    if not spec.race_pair:
        return None
    import inspect

    fn = getattr(spec.kernel_fn, "fn", spec.kernel_fn)
    try:
        lines, start = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return [None for _ in spec.race_pair]
    out: list[int | None] = []
    for needle in spec.race_pair:
        for i, line in enumerate(lines):
            if needle in line:
                out.append(start + i)
                break
        else:
            out.append(None)
    return out


def run_one(spec: LaunchSpec, seed: int, mutate: bool = False) -> dict[str, Any]:
    kernel_fn = getattr(spec.kernel_fn, "fn", spec.kernel_fn)
    row: dict[str, Any] = {
        "name": spec.name,
        "pattern": spec.pattern,
        "expected": spec.expected,
        "race_pair": list(spec.race_pair) if spec.race_pair else None,
        "race_pair_lines": _resolve_race_pair_lines(spec),
        "params_note": spec.params_note,
        "grid": list(spec.grid),
        "seed": seed,
        # Kernel identity: the ladder audit groups rows of one
        # SPECIALIZATION (kernel, constexprs) to derive the kernel-level
        # "∃ racy input" truth that proved@T0 claims are checked against.
        # Non-JSON constexpr values (e.g. tl.float32 dtype objects) are
        # stringified for the row.
        "kernel": getattr(kernel_fn, "__name__", str(kernel_fn)),
        "constexprs": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in spec.constexprs.items()
        },
        "aliased": spec.aliased,
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

    if mutate and row["static"].get("status") == "ok":
        try:
            row["mutation"] = _mutation_track(spec, ttir, seed)
        except Exception as e:  # noqa: BLE001
            row["mutation"] = {"error": f"{type(e).__name__}: {e}"}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutate", action="store_true")
    ns = ap.parse_args()

    from evaluation.kernels import load

    corpus = load(ns.corpus)
    spec = next(s for s in corpus.specs if s.name == ns.spec)
    row = run_one(spec, ns.seed, mutate=ns.mutate)
    row["corpus"] = ns.corpus
    with open(ns.out, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main()
