"""RQ3 scaling sweeps (plan S5 / paper sec:eval-scaling).

One dimension varies at a time on synthesized TTIR modules (host-compile
noise excluded — text synthesis keeps the sweep about the ANALYSIS);
recorded per point: parse ("capture"), encode + solver construction
("construct"), find_races ("solve") wall-clock, query count, per-query
mean/median/p95, SAT/UNSAT split, peak traced-memory, timeouts.

The paper predicts the shapes; the sweep confirms or falsifies:

  grid size     — INVARIANT: the grid enters as one symbolic bound, so
                  cost must not grow with launch size.
  tile width    — INVARIANT: an arange lane is one summary variable, so
                  cost must not grow with BLOCK.
  trip count    — INVARIANT: the loop is one symbolic iteration index.
  site count m  — QUADRATIC query growth: every cross-copy pair is a
                  query (2m^2 with the store/load pair per site), plus
                  the intra-instance pass.
  atomic count c — CUBIC constraint growth: the coherence axioms
                  quantify writer x reader x interposer.

Usage:  uv run python -m evaluation.scaling [--quick]
Writes results/SCALING.md.
"""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
T1_BASE = 0x10000
T2_BASE = 0x800000


# ── TTIR synthesis ───────────────────────────────────────────────


def _module(args: str, *body: str) -> str:
    inner = "\n    ".join(body)
    return (
        "module {\n"
        f"  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {inner}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


def synth_elementwise(block: int) -> str:
    """pid-tiled load/store pair, tile width `block`."""
    return _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
        f"%cB = arith.constant {block} : i32",
        "%pid = tt.get_program_id x : i32",
        f"%r = tt.make_range {{end = {block} : i32, start = 0 : i32}} : tensor<{block}xi32>",
        "%base = arith.muli %pid, %cB : i32",
        f"%sb = tt.splat %base : i32 -> tensor<{block}xi32>",
        f"%offs = arith.addi %sb, %r : tensor<{block}xi32>",
        f"%xp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<{block}x!tt.ptr<f32>>",
        f"%xa = tt.addptr %xp, %offs : tensor<{block}x!tt.ptr<f32>>, tensor<{block}xi32>",
        f"%v = tt.load %xa : tensor<{block}x!tt.ptr<f32>>",
        f"%op = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<{block}x!tt.ptr<f32>>",
        f"%oa = tt.addptr %op, %offs : tensor<{block}x!tt.ptr<f32>>, tensor<{block}xi32>",
        f"tt.store %oa, %v : tensor<{block}x!tt.ptr<f32>>",
    )


def synth_loop(trip: int) -> str:
    """pid-tiled store loop with a CONSTANT trip count baked into the
    bounds (the T1 model concretizes them; one symbolic iteration index)."""
    return _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        f"%cT = arith.constant {trip} : i32",
        "%c64 = arith.constant 64 : i32",
        f"%cSeg = arith.constant {trip * 64} : i32",
        "%pid = tt.get_program_id x : i32",
        "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "%base = arith.muli %pid, %cSeg : i32",
        "scf.for %i = %c0 to %cT step %c1  : i32 {",
        "%ib = arith.muli %i, %c64 : i32",
        "%s = arith.addi %base, %ib : i32",
        "%sb = tt.splat %s : i32 -> tensor<64xi32>",
        "%offs = arith.addi %sb, %r : tensor<64xi32>",
        "%op = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%oa = tt.addptr %op, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "%xp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%xa = tt.addptr %xp, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "%v = tt.load %xa : tensor<64x!tt.ptr<f32>>",
        "tt.store %oa, %v : tensor<64x!tt.ptr<f32>>",
        "scf.yield",
        "}",
    )


def synth_sites(m: int) -> str:
    """m disjoint pid-tiled store sites (each its own 64-lane stripe of one
    output tensor): query count must grow ~quadratically in m."""
    body = [
        "%c64 = arith.constant 64 : i32",
        f"%cM = arith.constant {m * 64} : i32",
        "%pid = tt.get_program_id x : i32",
        "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "%base = arith.muli %pid, %cM : i32",
        "%cst = arith.constant dense<1> : tensor<64xi32>",
    ]
    for j in range(m):
        body += [
            f"%o{j} = arith.constant {j * 64} : i32",
            f"%s{j} = arith.addi %base, %o{j} : i32",
            f"%sb{j} = tt.splat %s{j} : i32 -> tensor<64xi32>",
            f"%offs{j} = arith.addi %sb{j}, %r : tensor<64xi32>",
            f"%op{j} = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>",
            f"%oa{j} = tt.addptr %op{j}, %offs{j} : tensor<64x!tt.ptr<i32>>, tensor<64xi32>",
            f"tt.store %oa{j}, %cst : tensor<64x!tt.ptr<i32>>",
        ]
    return _module("%out_ptr: !tt.ptr<i32>", *body)


def synth_atomics(c: int) -> str:
    """c scalar acq_rel atomic_add sites on ONE counter cell: the coherence
    axioms range over writer x reader x interposer (O(c^3) constraints)."""
    body = [
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
    ]
    for j in range(c):
        body.append(
            f"%old{j} = tt.atomic_rmw add, acq_rel, gpu, %ctr_ptr, %c1, %true : "
            "(!tt.ptr<i32>, i32, i1) -> i32"
        )
    # one gated store so the launch has a conflict question to ask
    body += [
        "%c3 = arith.constant 3 : i32",
        "%done = arith.cmpi eq, %old0, %c3 : i32",
        "tt.store %out_ptr, %c1, %done : !tt.ptr<i32>",
    ]
    return _module("%ctr_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>", *body)


# ── measurement ──────────────────────────────────────────────────


def measure(ttir: str, tensors: dict, grid: tuple) -> dict:
    from triton_viz.clients.common.ttir_reader import parse_ttir
    from triton_viz.clients.race_detector.compiled.global_records import (
        encode_graph,
        symbolic_grid,
    )
    from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
        TwoCopySymbolicHBSolver,
    )
    from triton_viz.clients.race_detector.hb_common import (
        UnsupportedSymbolicRaceQuery,
    )

    tracemalloc.start()
    out: dict = {}
    t0 = time.perf_counter()
    graph = parse_ttir(ttir)
    out["capture_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc = encode_graph(graph, {}, tensors)
    g = symbolic_grid(enc) if grid is None else grid
    solver = TwoCopySymbolicHBSolver(enc.records, grid=g, arange_dict=enc.arange_dict)
    out["construct_s"] = time.perf_counter() - t0
    out["n_base_constraints"] = (
        len(solver.rf_constraints)
        + len(solver.atomic_coherence_constraints)
        + len(solver.counting_constraints)
    )

    t0 = time.perf_counter()
    timeout = False
    try:
        reports = solver.find_races()
        out["n_reports"] = len(reports)
    except UnsupportedSymbolicRaceQuery:
        timeout = True
        out["n_reports"] = -1
    out["solve_s"] = time.perf_counter() - t0
    out["timeout"] = timeout

    stats = getattr(solver, "query_stats", [])
    times = [s for _, s, _ in stats]
    out["n_queries"] = len(stats)
    out["n_sat"] = sum(1 for _, _, sat in stats if sat)
    if times:
        out["q_mean_ms"] = 1000 * statistics.fmean(times)
        out["q_median_ms"] = 1000 * statistics.median(times)
        out["q_p95_ms"] = 1000 * (
            statistics.quantiles(times, n=20)[-1] if len(times) >= 2 else times[0]
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["peak_mb"] = peak / 1e6
    return out


def _t(base: int, numel: int = 1 << 18) -> object:
    # Extents must NOT overlap between the two bases (1 MB extents, 8 MB
    # apart): the in-bounds premise then rules out cross-tensor aliasing,
    # keeping every invariance row at SAT=0 — variation would be a harness
    # artifact, not solver behavior.
    from triton_viz.clients.race_detector.compiled.global_records import GlobalTensor

    return GlobalTensor(data_ptr=base, elem_size=4, numel=numel)


def sweep(quick: bool) -> str:
    xy = {"x_ptr": _t(T1_BASE), "out_ptr": _t(T2_BASE)}
    from typing import Callable

    dims: list[tuple[str, str, list, Callable[[int], tuple]]] = [
        (
            "grid size (prediction: INVARIANT — one symbolic bound)",
            "grid",
            [4, 64, 1024, 2**20],
            lambda v: (synth_elementwise(64), xy, (v,)),
        ),
        (
            "tile width BLOCK (prediction: INVARIANT — one lane summary var)",
            "block",
            [32, 128, 512, 2048],
            lambda v: (synth_elementwise(v), xy, None),
        ),
        (
            "loop trip count (prediction: INVARIANT — one iteration index)",
            "trip",
            [2, 8, 64, 512],
            lambda v: (synth_loop(v), xy, None),
        ),
        (
            "static site count m (prediction: ~m^2 queries)",
            "m",
            [2, 4, 8] + ([] if quick else [16]),
            lambda v: (synth_sites(v), {"out_ptr": _t(T2_BASE)}, None),
        ),
        (
            "atomic site count c (prediction: O(c^3) coherence constraints)",
            "c",
            [1, 2, 4] + ([] if quick else [8]),
            lambda v: (
                synth_atomics(v),
                {"ctr_ptr": _t(T1_BASE, numel=1), "out_ptr": _t(T2_BASE, numel=1)},
                None,
            ),
        ),
    ]

    lines = ["# RQ3 scaling sweeps", ""]
    for title, label, values, make in dims:
        lines += [f"## {title}", ""]
        lines += [
            f"| {label} | capture s | construct s | solve s | queries | SAT "
            "| q mean ms | q p95 ms | base cons | peak MB | timeout |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for v in values:
            ttir, tensors, grid = make(v)
            r = measure(ttir, tensors, grid)
            lines.append(
                f"| {v} | {r['capture_s']:.3f} | {r['construct_s']:.3f} "
                f"| {r['solve_s']:.3f} | {r['n_queries']} | {r['n_sat']} "
                f"| {r.get('q_mean_ms', 0):.2f} | {r.get('q_p95_ms', 0):.2f} "
                f"| {r['n_base_constraints']} | {r['peak_mb']:.1f} "
                f"| {'YES' if r['timeout'] else '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ns = ap.parse_args()
    out = sweep(ns.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "SCALING.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
