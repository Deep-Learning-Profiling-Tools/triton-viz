"""T0 symbolic loop bounds (the S5 stretch).

The T0 encoder no longer concretizes scf.for bounds: a param-valued
lower/upper becomes the iteration-existence premise
``k >= 0 ∧ lower + k·step < upper`` (step still a compile-time constant —
symbolic k·step is the nonlinear bait the linearity gate exists to block).
The T0 claim then quantifies over EVERY trip count, and zero-trip launches
are subsumed (upper <= lower makes the premise UNSAT — no phantom
footprint).

The T1 path is UNCHANGED: param bounds at T1 still abstain with the
"not concrete at launch" reason (the liger corpus rows depend on it).
"""

from types import SimpleNamespace

import pytest
import torch

from triton_viz.clients.common.ttir_reader import UnsupportedTTIR, parse_ttir
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
    symbolic_grid,
    t0_linearity_gate,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_t1_rmw_static import _module


def _seg_walk_ttir(seg: int, mask_bound: int) -> str:
    """for k in range(0, n): if k < mask_bound: store(out + pid*seg + k).

    With mask_bound <= seg the per-pid segments stay disjoint for EVERY n
    (T0-provable with a symbolic trip count); with mask_bound > seg,
    iterations k in [seg, mask_bound) reach the next pid's segment — racy
    for a large enough n."""
    return _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        f"%cSeg = arith.constant {seg} : i32",
        f"%cMask = arith.constant {mask_bound} : i32",
        "%pid = tt.get_program_id x : i32",
        "%base = arith.muli %pid, %cSeg : i32",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%m = arith.cmpi slt, %k, %cMask : i32",
        "%off = arith.addi %base, %k : i32",
        "%oa = tt.addptr %out_ptr, %off : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1, %m : !tt.ptr<i32>",
        "scf.yield",
        "}",
    )


def _t0_races(graph):
    groups = encode_graph_t0(graph)
    for _name, enc in groups:
        solver = TwoCopySymbolicHBSolver(
            enc.records, grid=symbolic_grid(enc), arange_dict=enc.arange_dict
        )
        reports = solver.find_races()
        if reports:
            return reports
    return []


# ─────────────────── the encoder-level contract ───────────────────


def test_bounded_seg_walk_proves_at_t0_for_any_trip_count():
    """mask k < seg keeps every iteration inside the pid's own segment:
    UNSAT for ANY n — the claim the concrete-bounds T0 could never make."""
    g = parse_ttir(_seg_walk_ttir(seg=64, mask_bound=64))
    assert t0_linearity_gate(g)
    assert _t0_races(g) == []


def test_overflowing_seg_walk_is_sat_at_t0():
    """Mutation twin: mask k < 2*seg lets iterations spill into the next
    segment — SAT (so the tier selector falls to T1), never a proof."""
    g = parse_ttir(_seg_walk_ttir(seg=64, mask_bound=128))
    assert _t0_races(g), "spilling iterations must surface as a T0 SAT"


def test_unmasked_symbolic_loop_is_sat_at_t0():
    """No mask at all: for n > seg the walk crosses segments — must be SAT
    at T0 (the honest outcome; T1 then decides with the launch's n)."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%c64 = arith.constant 64 : i32",
        "%pid = tt.get_program_id x : i32",
        "%base = arith.muli %pid, %c64 : i32",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%off = arith.addi %base, %k : i32",
        "%oa = tt.addptr %out_ptr, %off : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1 : !tt.ptr<i32>",
        "scf.yield",
        "}",
    )
    assert _t0_races(parse_ttir(text))


def test_zero_trip_subsumed_by_existence_premise():
    """A store racy ONLY inside the loop, lower==upper==the same param:
    the existence premise is UNSAT for every valuation with upper <= lower,
    and for n >= 1 the fixed-address store races — T0 must be SAT (fall to
    T1), NOT a proof and NOT a crash."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%dead = arith.muli %pid, %c0 : i32",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%oa = tt.addptr %out_ptr, %dead : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1 : !tt.ptr<i32>",
        "scf.yield",
        "}",
    )
    assert _t0_races(parse_ttir(text)), "n >= 1 makes every block hit out[0]"


def test_symbolic_step_still_unsupported_at_t0():
    """k·step with a param step is nonlinear: encode_graph_t0 must refuse
    (the tier selector then falls to T1, where the step is concrete)."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32, %s: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "scf.for %k = %c0 to %n step %s  : i32 {",
        "%off = arith.addi %pid, %k : i32",
        "%oa = tt.addptr %out_ptr, %off : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1 : !tt.ptr<i32>",
        "scf.yield",
        "}",
    )
    with pytest.raises(UnsupportedTTIR, match="step is not a compile-time"):
        encode_graph_t0(parse_ttir(text))


def test_t1_pid_dependent_bounds_still_abstain():
    """The T1 path is untouched: a PID-dependent bound (liger's
    ``pid * rows_per_program`` slab pattern) keeps the 'not concrete at
    launch' abstention — the liger corpus rows depend on this exact
    behavior. (A merely UNCAPTURED param raises its own message earlier.)"""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%lo = arith.muli %pid, %n : i32",
        "%hi = arith.addi %lo, %n : i32",
        "scf.for %k = %lo to %hi step %c1  : i32 {",
        "%oa = tt.addptr %out_ptr, %k : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1 : !tt.ptr<i32>",
        "scf.yield",
        "}",
    )
    with pytest.raises(UnsupportedTTIR, match="not concrete at launch"):
        encode_graph(
            parse_ttir(text), {"n": 4}, {"out_ptr": GlobalTensor(0x1000, 4, 1 << 16)}
        )


def test_t1_concrete_bounds_unchanged():
    """Same kernel with the launch's n: T1 encodes and proves as before."""
    g = parse_ttir(_seg_walk_ttir(seg=64, mask_bound=64))
    enc = encode_graph(g, {"n": 32}, {"out_ptr": GlobalTensor(0x1000, 4, 1 << 16)})
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (4, 1, 1)),
        arange_dict=enc.arange_dict,
    )
    assert solver.find_races() == []


# ─────────────────── through the client (the rung) ───────────────────


def _drive(ttir: str, n: int) -> CompiledRaceDetector:
    det = CompiledRaceDetector(confirm_races=False)
    jit = SimpleNamespace(arg_names=["out_ptr", "n"])
    det.pre_warmup_callback(jit, torch.zeros(1 << 14, dtype=torch.int32), n, grid=(4,))
    det.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()
    return det


def test_client_awards_t0_rung_for_symbolic_trip_count():
    """End to end: the bounded segment walk now lands on proved@T0 — the
    'any scalar params (including any trip count), any grid' rung that the
    concrete-bounds encoder had to forfeit to T1."""
    det = _drive(_seg_walk_ttir(seg=64, mask_bound=64), n=32)
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T0"
    assert det.last_global_verdict is not None
    assert det.last_global_verdict["proved_scope"] == "any-params-any-grid"


def test_client_overflowing_walk_falls_to_t1_and_reports():
    """The spilling twin: T0 SAT → T1 with the launch's n=128 → iterations
    64..127 cross into pid+1's segment — definite reports."""
    det = _drive(_seg_walk_ttir(seg=64, mask_bound=128), n=128)
    assert det.last_global_status == "races"


def test_client_overflowing_walk_small_n_proves_at_t1_only():
    """Same spilling kernel at n=32: no iteration reaches the spill range,
    so T1 proves — but the rung must be T1, never T0 (the kernel is NOT
    race-free for every n)."""
    det = _drive(_seg_walk_ttir(seg=64, mask_bound=128), n=32)
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1"
