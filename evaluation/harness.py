"""Per-kernel worker: one LaunchSpec in, one JSONL row out.

Runs INSIDE the per-spec subprocess (see runner.py). Phase order is fixed
and load-bearing: the REAL host compile happens before anything engages the
interpreter (static C2/C3 replay, then the dynamic-mode comparison) — the
reverse order trips the interpreter-patching hazard documented in
core/trace.py.

Verdict mapping for DRB-style scoring (plan S5):
  static ok            -> "race-free"  (terminal = provenance rung; the §3c
                          proved@T1-launch rung carries its any-grid
                          evidence in static["grid_fragile"] — an
                          independent attribute, never a race count)
  static races         -> "race"       (terminal = race-confirmed | races-unclassified)
  static unsupported   -> "abstain"    (terminal = race-unconfirmed | unsupported)
  abstain + L1 rung    -> "race-free" proved@enum | "race" race@enum
                          (the concrete per-instance enumeration rung,
                          reached only at ladder level L1+ and only when
                          the composed verdict is an abstention; analyzed-
                          launch extent, content-fragile)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any

from evaluation.spec import LaunchSpec
from triton_viz.core.config import config as cfg
from triton_viz.clients.race_detector.ladder import (
    LADDER_LEVEL_NAMES,
    LadderLevel,
    parse_ladder_level,
)


def _launch_binding(spec, args) -> dict:
    """Bind the launch entirely BY NAME.

    ``make_args`` returns the non-constexpr parameters in declaration
    order (the corpus convention), but a positional call misbinds any
    runtime parameter declared AFTER a constexpr (its value lands in the
    constexpr's slot) and collides with constexpr-None optional pointers.
    Zipping against the kernel's own arg_names sidesteps both."""
    names = [n for n in spec.kernel_fn.arg_names if n not in spec.constexprs]
    if len(names) != len(args):
        raise RuntimeError(
            f"launch binding mismatch: {len(args)} args for params {names}"
        )
    return {**dict(zip(names, args)), **spec.constexprs}


def _host_compile_ttir(spec: LaunchSpec) -> str:
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = spec.kernel_fn
    # Under TRITON_INTERPRET, @triton.jit yields InterpretedFunction, which
    # triton >= 3.7 ASTSource.hash() rejects (no .cache_key) — rebuild the
    # real JITFunction from the raw callable for the host compile.
    if not hasattr(fn, "cache_key") and hasattr(fn, "fn"):
        fn = triton.runtime.jit.JITFunction(fn.fn)
    src = ASTSource(fn=fn, signature=spec.signature, constexprs=spec.constexprs)
    # sm80 suffices for every pre-fp8 corpus and keeps the host compile
    # GPU-free, but fp8e4nv args (torchao) fail triton's frontend check
    # below cc 89 — target the real device capability when one exists
    cc = 80
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        cc = major * 10 + minor
    k = triton.compile(src, target=GPUTarget("cuda", cc, 32))
    return k.asm["ttir"]


def _static_track(
    spec: LaunchSpec,
    ttir: str,
    seed: int,
    ladder_level: LadderLevel = LadderLevel.L0,
) -> dict[str, Any]:
    from types import SimpleNamespace

    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    det = CompiledRaceDetector(
        confirm_races=True, differential_check=True, ladder_level=ladder_level
    )
    args = spec.make_args(seed)
    t0 = time.perf_counter()
    det.pre_warmup_callback(
        spec.kernel_fn, grid=spec.grid, **_launch_binding(spec, args)
    )
    det.post_warmup_callback(spec.kernel_fn, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()
    elapsed = time.perf_counter() - t0

    # tier-selector detail, recomputed via the public gate (the client does
    # not publish it): lets the T0 stretch show up as a re-run diff.
    t0_gate = None
    try:
        from triton_viz.clients.common.ttir_reader import parse_ttir
        from triton_viz.clients.race_detector.compiled.global_records import (
            t0_linearity_gate,
        )

        t0_gate = bool(t0_linearity_gate(parse_ttir(ttir, multipath=ladder_level >= 2)))
    except Exception:  # noqa: BLE001
        pass

    return _static_result(det, elapsed, t0_gate)


def _static_result(det: Any, elapsed: float, t0_gate: bool | None) -> dict[str, Any]:
    """The static track's row fragment from a client that has settled a
    launch (finalize() for the Triton track, analyze_graph() for the
    cuTile track): verdict, provenance and evidence, read from the same
    fields either way."""

    def _pair(rep: Any, kind_key: str) -> dict:
        return {
            "first": rep.first_record.source_location,
            "second": rep.second_record.source_location,
            kind_key: rep.race_type.name,
            "pids": [list(rep.witness_grid_a or ()), list(rep.witness_grid_b or ())],
        }

    # §3c guardrail 1: fragility evidence is carried as its own attribute
    # next to the launch-scoped proof — hazard wording, never "race"
    grid_fragile = [
        _pair(rep, "hazard") for rep in (getattr(det, "last_grid_fragile", []) or [])
    ]
    # §3n: the faithfully-refuted widened hazard's site pairs — evidence
    # for the composed dispatcher's content-fragile upgrade
    content_fragile = [
        {
            "first": rep.first_record.source_location,
            "second": rep.second_record.source_location,
            "hazard": rep.race_type.name,
        }
        for rep in (getattr(det, "last_content_hazard", []) or [])
    ]
    return {
        "status": det.last_global_status,
        "provenance": det.last_global_provenance,
        "confirmation": det.last_global_confirmation,
        "reason": det.last_global_reason,
        "n_reports": len(det.last_global_reports),
        "witnesses": [_pair(rep, "race_type") for rep in det.last_global_reports],
        "grid_fragile": grid_fragile,
        "content_fragile": content_fragile,
        "parse_unsupported": [r for r in det.last_ttir_unsupported if r],
        "differential": det.last_differential,
        "t0_gate": t0_gate,
        "assumes_termination": det.last_global_assumes_termination,
        "verdict_attrs": det.last_global_verdict,
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


def _cutile_bindings(args: list[dict]) -> tuple[dict, dict, bool]:
    """(params, tensors, aliased) from captured cuTile arg descriptors.

    Scalars bind under their python names; an array param ``p`` also
    binds its FLATTENED metadata slots (``p_1..p_r`` shape dims,
    ``p_{r+1}..p_{2r}`` strides — the cuTile calling convention the IR
    references). Tensor base addresses are synthesized: distinct alias
    groups get disjoint fake allocations (the solver only needs interval
    disjointness/overlap structure, which the capture recorded), aliased
    args share one base."""
    from triton_viz.clients.race_detector.compiled.global_records import GlobalTensor

    params: dict[str, int] = {}
    tensors: dict[str, GlobalTensor] = {}
    group_base: dict[int, int] = {}
    next_base = 1 << 40
    aliased = False
    for d in args:
        if d["kind"] == "scalar":
            v = d["value"]
            if isinstance(v, (bool, int)):
                params[d["name"]] = int(v)
        elif d["kind"] == "tensor":
            nm, rank = d["name"], len(d["shape"])
            for i, s in enumerate(d["shape"]):
                params[f"{nm}_{i + 1}"] = int(s)
            for i, s in enumerate(d["strides"]):
                params[f"{nm}_{rank + 1 + i}"] = int(s)
            group = d.get("alias", nm)
            if group in group_base:
                aliased = True
                base = group_base[group]
            else:
                base = next_base
                group_base[group] = base
                next_base += (d["numel"] * d["elem_size"] + 4095) & ~4095
                next_base += 4096  # guard gap between allocations
            tensors[nm] = GlobalTensor(
                data_ptr=base,
                numel=d["numel"],
                elem_size=d["elem_size"],
                contiguous=bool(d["contiguous"]),
            )
    return params, tensors, aliased


def _static_track_cutile(
    spec: LaunchSpec, seed: int, ladder_level: LadderLevel = LadderLevel.L0
) -> dict[str, Any]:
    """The compiled static track over the captured CuTile IR: the same
    tier selector and verdict settlement as the Triton track, through the
    client's public ``analyze_graph`` (cuda.tile has no interpreter, so
    there are no launch callbacks, no C2 replay and no C3 differential:
    race SATs terminate at races-unclassified, proofs carry their scope
    rungs and qualifiers exactly like the Triton track)."""
    from triton_viz.clients.common.cutile_ir_reader import parse_cutile_ir
    from triton_viz.clients.common.ttir_reader import UnsupportedTTIR
    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    info = spec.cutile or {}
    kname = info.get("kernel", spec.name)
    det = CompiledRaceDetector(
        confirm_races=False, differential_check=False, ladder_level=ladder_level
    )
    t0 = time.perf_counter()
    graph, parse_reason = None, None
    try:
        graph = parse_cutile_ir(
            info["ir"], kname, multipath=ladder_level >= LadderLevel.L2
        )
    except UnsupportedTTIR as e:
        parse_reason = f"{e.kind}: {e}"
    params: dict = {}
    tensors: dict = {}
    if graph is not None:
        params, tensors, _ = _cutile_bindings(info["args"])
    det.analyze_graph(
        graph,
        params,
        tensors,
        tuple(spec.grid),
        parse_reason=parse_reason,
        replay_note="no cuTile replay channel",
    )
    return _static_result(det, time.perf_counter() - t0, None)


def _run_one_cutile(
    spec: LaunchSpec, seed: int, ladder_level: LadderLevel = LadderLevel.L0
) -> dict[str, Any]:
    info = spec.cutile or {}
    row: dict[str, Any] = {
        "name": spec.name,
        "pattern": spec.pattern,
        "expected": spec.expected,
        "race_pair_lines": None,
        "params_note": spec.params_note,
        "grid": list(spec.grid),
        "seed": seed,
        "kernel": info.get("kernel", spec.name),
        "constexprs": dict(spec.constexprs),
        "aliased": spec.aliased,
        "frontend": "cutile",
        # cuda.tile has no interpreter, so the L1 rung can never run on
        # these rows; the level is still stamped (provenance discipline).
        "ladder_level": ladder_level.name,
        # the memory-model switch this process ran under; cuTile graphs keep
        # the legacy reading regardless (the reader does not yet turn tokens
        # into fences, design-fence-order.md stage 1d), stamped explicitly
        "fence_order": bool(cfg.race_detector_fence_order),
        "fence_order_applies": False,
    }
    try:
        row["static"] = _static_track_cutile(spec, seed, ladder_level)
    except Exception as e:  # noqa: BLE001
        row.update(
            verdict="error",
            terminal="harness-error",
            harness_error=f"cutile static track: {type(e).__name__}: {e}",
        )
        return row
    row["dynamic"] = {
        "status": "unsupported",
        "reason": "cuda.tile has no interpreter — static track only (v1)",
        "n_reports": 0,
        "premises": [],
        "witnesses": [],
        "error": None,
        "time_s": 0.0,
    }
    row["verdict"], row["terminal"] = _classify(row["static"], None)
    return row


def _dynamic_track(
    spec: LaunchSpec, seed: int, ladder_level: LadderLevel = LadderLevel.L0
) -> dict[str, Any]:
    import triton_viz
    from triton_viz.clients import RaceDetector
    from triton_viz.clients.race_detector.hb_common import (
        UnsupportedSymbolicRaceQuery,
    )

    # abort_on_error: once any capture path marks the launch unsupported,
    # finalize() discards every record and reports nothing, so all further
    # interpretation is provably dead work — the sweep had rows spinning
    # 40-60 s after their mark. Every mark site marks BEFORE raising, so
    # catching the abort and running finalize() classifies the launch
    # exactly as the mark-and-continue mode would have.
    det = RaceDetector(abort_on_error=True, ladder_level=ladder_level)
    args = spec.make_args(seed)  # fresh tensors; the interpreter mutates them
    t0 = time.perf_counter()
    error = None
    timed_out = False
    try:
        traced = triton_viz.trace(det)(spec.kernel_fn)
        with _watchdog(DYNAMIC_TIMEOUT_S):
            traced[spec.grid](**_launch_binding(spec, args))
    except TimeoutError as e:
        error = str(e)
        timed_out = True
    except UnsupportedSymbolicRaceQuery:
        det.finalize()  # idempotent: reads the mark, sets "unsupported"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    witnesses = [
        {
            "first": rep.first_record.source_location,
            "second": rep.second_record.source_location,
            "race_type": rep.race_type.name,
            "pids": [list(rep.witness_grid_a or ()), list(rep.witness_grid_b or ())],
        }
        for rep in (getattr(det, "last_reports", []) or [])
    ]
    return {
        "status": "timeout" if timed_out else getattr(det, "last_status", None),
        "reason": getattr(det, "unsupported_reason", None),
        "n_reports": len(getattr(det, "last_reports", []) or []),
        "premises": list(getattr(det, "last_premises", ()) or ()),
        "witnesses": witnesses,
        "error": error,
        "time_s": round(elapsed, 4),
    }


# ── the L1 rung: concrete per-instance enumeration (Route 1) ────────
# The rung itself has no time budget (design-route1-concrete-enumeration.md
# section 4): its watchdog here is evaluation protocol, the per-row
# subprocess budget at this level (runner.row_timeout_s: 200 s at L1,
# Hao 2026-09-04) minus what the symbolic tracks already spent and a
# margin for the subprocess's own startup and teardown, floored so a
# spin the taint did not see still ends in a NAMED refusal rather than
# a row-level crash. Measured: ~1.3-3 ms per instance for the destindex
# family (32768 instances in ~46 s), ~28 ms per instance for an
# attention kernel, 100+ ms for the chunked-prefill kernels.
ENUM_MIN_TIMEOUT_S = 30
ENUM_ROW_MARGIN_S = 10


def _enum_budget_s(
    row_started: float, ladder_level: LadderLevel = LadderLevel.L1
) -> float:
    from evaluation.runner import row_timeout_s

    remaining = (
        row_timeout_s(ladder_level)
        - (time.perf_counter() - row_started)
        - ENUM_ROW_MARGIN_S
    )
    return float(max(ENUM_MIN_TIMEOUT_S, remaining))


ENUM_TIMEOUT_S = 200  # the standalone default (the row budget at L1)


def _enum_track(
    spec: LaunchSpec,
    seed: int,
    static: dict[str, Any],
    timeout_s: float = ENUM_TIMEOUT_S,
) -> dict[str, Any]:
    """Route 1 on one launch: every instance evaluated concretely on fresh,
    cloned tensors; verdict at the analyzed-launch extent. Refusals are
    named (``"<kind>: detail"``). The spin pre-gate fires only when the
    static reader RECOGNIZED an await (``assumes_termination``): the
    sequential interpreter cannot terminate a cross-instance spin. The
    reader's ``spin-shape`` refusal kind is NOT a gate: it also covers
    carried-value ``scf.while`` loops that are plain data-dependent
    iteration (SWEEP_REPORT section 7), which the rung evaluates; a
    genuine spin the reader did not recognize refuses through the rung's
    own taint at its first poll."""
    from triton_viz.clients.race_detector.concrete_enum import enumerate_launch

    t0 = time.perf_counter()
    if static.get("assumes_termination"):
        return {
            "status": "unsupported",
            "reason": (
                "spin-shape: await-bearing kernel (static reader recognized an "
                "await); the sequential interpreter cannot terminate a "
                "cross-instance spin"
            ),
            "n_reports": 0,
            "witnesses": [],
            "instances": 0,
            "n_ops": 0,
            "time_s": round(time.perf_counter() - t0, 4),
        }
    args = spec.make_args(seed)  # fresh contents; enumerate_launch clones them
    outcome = enumerate_launch(
        spec.kernel_fn,
        (),
        _launch_binding(spec, args),
        spec.grid,
        timeout_s=timeout_s,
    )
    witnesses = [
        {
            "first": rep.first_record.source_location,
            "second": rep.second_record.source_location,
            "race_type": rep.race_type.name,
            "pids": [list(rep.witness_grid_a), list(rep.witness_grid_b)],
            "bytes": list(rep.byte_range),
        }
        for rep in outcome.reports
    ]
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "n_reports": len(outcome.reports),
        "witnesses": witnesses,
        "instances": outcome.n_instances,
        "n_ops": outcome.n_ops,
        "value_source_loads": outcome.n_value_source_loads,
        "instance_s": (
            round(outcome.instance_s, 6) if outcome.instance_s is not None else None
        ),
        "max_instance_s": (
            round(outcome.max_instance_s, 6)
            if outcome.max_instance_s is not None
            else None
        ),
        "run_s": round(outcome.run_s, 4),
        "analyze_s": round(outcome.analyze_s, 4),
        "timeout_s": timeout_s,
        "time_s": round(time.perf_counter() - t0, 4),
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


def _mutation_track(
    spec: LaunchSpec, ttir: str, seed: int, ladder_level: LadderLevel = LadderLevel.L0
) -> dict[str, Any]:
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
        # The mutants must run at the row's own level: an L2-only proof
        # mutated at L0 would just refuse (never "races") and read as a
        # vacuity survivor.
        det = CompiledRaceDetector(
            confirm_races=False, differential_check=False, ladder_level=ladder_level
        )
        args = spec.make_args(seed)
        det.pre_warmup_callback(
            spec.kernel_fn, grid=spec.grid, **_launch_binding(spec, args)
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


def _classify(
    static: dict[str, Any],
    dynamic: dict[str, Any] | None = None,
    enum: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """(verdict, terminal) from the composed dispatcher.

    The static track decides when it can; when it ABSTAINS and the
    interpreter track ran to completion, the interpreter's verdict is
    the decision — the plan's §I.3 composition (within each front-end's
    reachable region, the least concretization that decides). Those
    terminals live on the interpreter point of the concretization map:
    ``race@interp`` / ``proved@interp``, scoped per-launch (+ the
    contents-snapshot premise when an event address lowered through a
    load snapshot — carried in dynamic["premises"]).

    ``enum`` is the L1 rung's row (Route 1, run only when the composed
    verdict is an abstention): a clean concrete enumeration decides
    ``proved@enum``, concrete witnesses decide ``race@enum``; any
    refusal keeps the abstention. Absent (L0) the composition is exactly
    the pre-L1 one."""
    verdict, terminal = _classify_symbolic(static, dynamic)
    if verdict == "abstain" and enum:
        if enum.get("status") == "races" and (enum.get("n_reports") or 0) > 0:
            return ("race", "race@enum")
        if enum.get("status") == "ok" and not enum.get("reason"):
            return ("race-free", "proved@enum")
    return (verdict, terminal)


def _classify_symbolic(
    static: dict[str, Any], dynamic: dict[str, Any] | None = None
) -> tuple[str, str]:
    status = static["status"]
    if status == "ok":
        return ("race-free", static["provenance"] or "proved@T1")
    if status == "races":
        if static["confirmation"] == "confirmed":
            return ("race", "race-confirmed")
        return ("race", "races-unclassified")
    if status == "unsupported":
        dyn = dynamic or {}
        dyn_clean = dyn.get("status") == "ok" and not dyn.get("error")
        if "race-unconfirmed" in (static["reason"] or ""):
            # §3n (decision (b)): this reason is set ONLY when every
            # widened SAT was faithfully replayed on this launch's data
            # and none reproduced. When the interpreter ALSO ran this
            # launch clean, the composition owes it the launch-scoped
            # proof — the refuted hazard rides as the content-fragile
            # attribute (stamped by run_one), never as an abstention.
            # Capped / unavailable / unclassifiable demotions carry the
            # GENERIC reason, so they can never enter this upgrade.
            if dyn_clean:
                if (dyn.get("n_reports") or 0) > 0:
                    # concrete interp reports subsume the widened hazard
                    return ("race", "race@interp")
                return ("race-free", "proved@interp")
            # no proof exists — fail closed exactly as before
            return ("abstain", "race-unconfirmed")
        if dyn_clean:
            if (dyn.get("n_reports") or 0) > 0:
                return ("race", "race@interp")
            return ("race-free", "proved@interp")
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


def run_one(
    spec: LaunchSpec,
    seed: int,
    mutate: bool = False,
    ladder_level: LadderLevel = LadderLevel.L0,
) -> dict[str, Any]:
    # ladder_level: the detector's L0/L1/L2 switch (design §4b), stamped
    # into the results header by the runner. The cuTile track has no
    # interpreter (so no L1 rung); at L2 its reader runs in multipath mode.
    if spec.frontend == "cutile":
        return _run_one_cutile(spec, seed, ladder_level)
    row_started = time.perf_counter()
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
        # The ladder-depth stamp (provenance discipline: no dataset may
        # mix levels unnoticed); also carried in verdict_attrs by the
        # clients, which receive the same level.
        "ladder_level": ladder_level.name,
        # The memory-model switch this row's process ran under (fence-
        # ordered intra-instance semantics, design-fence-order.md; False
        # only for TRITON_VIZ_FENCE_ORDER=0 attribution runs).
        "fence_order": bool(cfg.race_detector_fence_order),
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
        row["static"] = _static_track(spec, ttir, seed, ladder_level)
    except Exception as e:  # noqa: BLE001
        row.update(
            verdict="error",
            terminal="harness-error",
            harness_error=f"static track: {type(e).__name__}: {e}",
        )
        return row

    try:
        row["dynamic"] = _dynamic_track(spec, seed, ladder_level)
    except Exception as e:  # noqa: BLE001
        row["dynamic"] = {"error": f"{type(e).__name__}: {e}"}

    row["verdict"], row["terminal"] = _classify(row["static"], row.get("dynamic"))
    if row["terminal"] == "proved@interp" and "race-unconfirmed" in (
        row["static"].get("reason") or ""
    ):
        # §3n guardrail 1: the attribute fires ONLY here — faithful
        # replay refuted every widened SAT AND the interpreter proved
        # this launch clean; the proof carries the contents-snapshot
        # premise the dynamic track reports (guardrail 2)
        va = dict(row["static"].get("verdict_attrs") or {})
        va["content_fragile"] = True
        row["static"]["verdict_attrs"] = va

    # The ladder switch: ONE gate. At L0 the rung does not run and the row
    # keeps today's abstention; at L1+ every symbolic rung has refused
    # (the composed verdict is an abstention), so the bottom rung decides
    # the launch by exhaustive per-instance concrete evaluation. Nothing
    # else in the pipeline consults the level.
    if ladder_level >= LadderLevel.L1 and row["verdict"] == "abstain":
        try:
            row["enum"] = _enum_track(
                spec,
                seed,
                row["static"],
                timeout_s=_enum_budget_s(row_started, ladder_level),
            )
        except Exception as e:  # noqa: BLE001
            row["enum"] = {
                "status": "unsupported",
                "reason": f"harness-error: {type(e).__name__}: {e}",
                "n_reports": 0,
                "witnesses": [],
            }
        row["verdict"], row["terminal"] = _classify(
            row["static"], row.get("dynamic"), row["enum"]
        )
        if row["terminal"] in ("proved@enum", "race@enum"):
            # analyzed-launch extent: these params, this grid, THESE
            # contents — the content-fragile attribute states the last
            # part, exactly as for proved@interp (same extent, different
            # provenance)
            va = dict(row["static"].get("verdict_attrs") or {})
            va["verdict"] = row["verdict"]
            va["proved_scope"] = (
                "this-params-this-grid" if row["verdict"] == "race-free" else None
            )
            va["race_evidence"] = "concrete" if row["verdict"] == "race" else None
            va["content_fragile"] = True
            va["conservative"] = False
            row["static"]["verdict_attrs"] = va

    if mutate and row["static"].get("status") == "ok":
        try:
            row["mutation"] = _mutation_track(spec, ttir, seed, ladder_level)
        except Exception as e:  # noqa: BLE001
            row["mutation"] = {"error": f"{type(e).__name__}: {e}"}
    return row


# ── worker mode: one process serves many rows (runner --reuse-workers) ──
#
# Per-row subprocesses cost 2-3 s of interpreter/torch import plus the
# corpus load per row (half of a 492-row change-surface run). A served
# worker keeps them; the runner still enforces the per-row budget (it
# kills a worker that does not answer in time) and crash containment (a
# dead worker is respawned), and recycles workers after a fixed number of
# rows or above an RSS limit. Row independence is the load-bearing
# property: triton's interpreter patches language state in place and a
# leaked patch breaks the next row's real compile (core/trace.py's
# warmup-only note), so the worker snapshots the language state before
# its first row and restores it after every row, reporting what leaked.

ROW_SENTINEL = "@@ROW@@"
_MISSING = object()


def _lang_state_watch() -> list[tuple[Any, str]]:
    import triton.language as tl
    from triton.runtime.interpreter import interpreter_builder

    watch: list[tuple[Any, str]] = [
        (tl.core.tensor, n)
        for n in ("__bool__", "__index__", "__repr__", "__str__", "T")
    ]
    watch += [
        (tl, n)
        for n in (
            "range",
            "static_range",
            "static_assert",
            "static_print",
            "multiple_of",
            "max_contiguous",
            "max_constancy",
            "reduce",
            "associative_scan",
        )
    ]
    watch += [(tl.core, n) for n in ("reduce", "associative_scan")]
    watch += [(tl.core.dtype, "to_ir")]
    watch += [
        (interpreter_builder, n)
        for n in dir(interpreter_builder)
        if not n.startswith("_") and callable(getattr(interpreter_builder, n, None))
    ]
    return watch


def _lang_state_snapshot(
    watch: list[tuple[Any, str]],
) -> list[tuple[Any, str, Any]]:
    return [(obj, n, vars(obj).get(n, _MISSING)) for obj, n in watch]


def _same_attr(a: Any, b: Any) -> bool:
    """Identity, or the same bound method (unpatch_op restores a builder
    op by setattr, which turns the class attribute into an equal
    instance-dict entry: not a leak)."""
    if a is b:
        return True
    fa, fb = getattr(a, "__func__", None), getattr(b, "__func__", None)
    return (
        fa is not None
        and fa is fb
        and getattr(a, "__self__", None) is getattr(b, "__self__", None)
    )


def _lang_state_restore(snapshot: list[tuple[Any, str, Any]]) -> list[str]:
    """Put every watched attribute back to its pristine value; returns the
    names that had leaked (for the worker's log)."""
    import triton

    leaked: list[str] = []
    for obj, n, orig in snapshot:
        cur = vars(obj).get(n, _MISSING)
        if cur is orig:
            continue
        if orig is _MISSING and _same_attr(
            cur, getattr(type(obj), n, None) and getattr(obj, n)
        ):
            # an instance-dict entry equal to the inherited attribute
            try:
                delattr(obj, n)
            except Exception:  # noqa: BLE001
                pass
            continue
        leaked.append(f"{getattr(obj, '__name__', type(obj).__name__)}.{n}")
        try:
            if orig is _MISSING:
                delattr(obj, n)
            else:
                setattr(obj, n, orig)
        except Exception:  # noqa: BLE001
            pass
    if triton.knobs.runtime.interpret:
        triton.knobs.runtime.interpret = False
        leaked.append("knobs.runtime.interpret")
    return leaked


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def serve(stdin: Any, stdout: Any) -> None:
    """The worker loop: one JSON request per line on stdin
    ``{"corpus", "spec", "seed", "mutate", "ladder_level", "out"}``;
    the row is written to ``out`` and one ``@@ROW@@ ok rss_mb=<n>`` (or
    ``@@ROW@@ error <message>``) line is printed. EOF ends the worker.
    ``probe`` is fault injection for the runner's tests (``crash``,
    ``hang``)."""
    import gc
    import os
    import signal as _signal

    from evaluation.kernels import load

    snapshot = _lang_state_snapshot(_lang_state_watch())
    corpora: dict[str, Any] = {}
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        probe = req.get("probe")
        if probe == "crash":
            os.kill(os.getpid(), _signal.SIGSEGV)
        if probe == "hang":
            time.sleep(3600)
        try:
            corpus = corpora.get(req["corpus"])
            if corpus is None:
                corpus = corpora[req["corpus"]] = load(req["corpus"])
            spec = next(s for s in corpus.specs if s.name == req["spec"])
            row = run_one(
                spec,
                int(req.get("seed", 0)),
                mutate=bool(req.get("mutate", False)),
                ladder_level=parse_ladder_level(req.get("ladder_level", "L0")),
            )
            row["corpus"] = req["corpus"]
            with open(req["out"], "w") as f:
                json.dump(row, f)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            status = f"error {type(e).__name__}: {e}".replace("\n", " ")[:400]
        leaked = _lang_state_restore(snapshot)
        if leaked:
            print(
                f"[serve] restored {len(leaked)} leaked attribute(s) after "
                f"{req.get('spec')}: {', '.join(leaked[:8])}",
                file=sys.stderr,
                flush=True,
            )
        gc.collect()
        print(
            f"{ROW_SENTINEL} {status} rss_mb={_rss_mb():.0f}", file=stdout, flush=True
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--spec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument(
        "--ladder-level",
        choices=LADDER_LEVEL_NAMES,
        default=LadderLevel.L0.name,
        help="ladder depth: L0 = shipped rungs only (default), L1 = + the "
        "concrete per-instance enumeration rung, L2 = + forked capture",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help="worker mode: serve rows requested on stdin (runner --reuse-workers)",
    )
    ns = ap.parse_args()

    if ns.serve:
        serve(sys.stdin, sys.stdout)
        return
    if not (ns.corpus and ns.spec and ns.out):
        ap.error("--corpus, --spec and --out are required (or --serve)")

    from evaluation.kernels import load

    corpus = load(ns.corpus)
    spec = next(s for s in corpus.specs if s.name == ns.spec)
    row = run_one(
        spec,
        ns.seed,
        mutate=ns.mutate,
        ladder_level=parse_ladder_level(ns.ladder_level),
    )
    row["corpus"] = ns.corpus
    with open(ns.out, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main()
