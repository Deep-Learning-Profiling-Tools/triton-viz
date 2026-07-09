"""T1 static-track tests for RMW-return modeling (spec part B, TTIR wiring).

Synthetic TTIR modules drive the shared reader → global_records → two-copy
solver path: integer atomic results bind to Observed terms, observations
are interned as copy-local vars with rf/counting justification, initial
values ride GlobalTensor.init_values, and every boundary (float atomics,
loop-carried atomics, plain-load indirection) stays fail-closed.
"""

from types import SimpleNamespace

import pytest
import torch

from triton_viz.clients.common.ttir_reader import (
    Observed,
    UnsupportedTTIR,
    mentions_observed,
    parse_ttir,
)
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)


def _t(ptr, numel=4096, elem=4, init=None):
    return GlobalTensor(data_ptr=ptr, elem_size=elem, numel=numel, init_values=init)


def _solve(graph, params, tensors):
    # Mirrors the client's T1 call: the launch grid sizes the unread axes
    # of atomic-bearing graphs (read axes stay symbolic).
    enc = encode_graph(graph, params, tensors)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (4, 1, 1)),
        arange_dict=enc.arange_dict,
    )
    return enc, solver.find_races()


def _module(args, *body_lines):
    body = "\n    ".join(body_lines)
    return (
        "module {\n"
        f"  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {body}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


# ─────────────────── last-block-done (mini TTIR) ───────────────────


def _lbd_ttir(sem, gate="num_programs"):
    """last-block-done: partial[pid] store; counter add; the last arriver
    reads the partials. The RACE-FREE form gates on num_programs(0)-1 — the
    T1 claim covers EVERY grid, and only a grid-tracking gate is race-free
    on every grid (a hardcoded `== 3` genuinely races when the launch grid
    exceeds 4: blocks with rank > 3 store after the gate opens)."""
    if gate == "num_programs":
        gate_lines = [
            "%np = tt.get_num_programs x : i32",
            "%nm1 = arith.subi %np, %c1 : i32",
            "%done = arith.cmpi eq, %old, %nm1 : i32",
        ]
    else:
        gate_lines = ["%done = arith.cmpi eq, %old, %c3 : i32"]
    return _module(
        "%partial_ptr: !tt.ptr<f32>, %counter_ptr: !tt.ptr<i32>, "
        "%out_ptr: !tt.ptr<f32>",
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "%c3 = arith.constant 3 : i32",
        "%cst = arith.constant 1.000000e+00 : f32",
        "%pid = tt.get_program_id x : i32",
        "%p = tt.addptr %partial_ptr, %pid : !tt.ptr<f32>, i32",
        "tt.store %p, %cst : !tt.ptr<f32>",
        f"%old = tt.atomic_rmw add, {sem}, gpu, %counter_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        *gate_lines,
        "%v = tt.load %partial_ptr, %done : !tt.ptr<f32>",
        "tt.store %out_ptr, %v, %done : !tt.ptr<f32>",
    )


_LBD_TENSORS = {
    "partial_ptr": _t(0x1000, numel=4),
    "counter_ptr": _t(0x2000, numel=1, init=(0,)),
    "out_ptr": _t(0x3000, numel=1),
}


def test_lbd_reader_binds_observed():
    g = parse_ttir(_lbd_ttir("acq_rel"))
    load = g.accesses[2]
    assert load.kind == "load"
    assert load.mask is not None and mentions_observed(load.mask)
    assert not load.mask_dropped
    rmw = g.accesses[1]
    assert rmw.atomic is not None and rmw.atomic.rmw_op == "add"
    assert rmw.atomic_val is not None
    assert not rmw.elem_float


def test_lbd_acq_rel_proved_t1():
    enc, reports = _solve(parse_ttir(_lbd_ttir("acq_rel")), {}, _LBD_TENSORS)
    assert enc.uncertain_event_ids == set()
    assert reports == []


def test_lbd_relaxed_races():
    """Mutation twin: no release/acquire, no sw — the partial-store/read
    pair must surface as a DEFINITE report (the mask is modeled, not
    widened)."""
    enc, reports = _solve(parse_ttir(_lbd_ttir("relaxed")), {}, _LBD_TENSORS)
    assert enc.uncertain_event_ids == set()
    partial = [
        r
        for r in reports
        if {r.first_record.tensor_name, r.second_record.tensor_name} == {"partial_ptr"}
    ]
    assert partial


def test_lbd_unknown_init_over_reports():
    """No init_values → counting axiom omitted → the epilogue gate cannot
    be pinned: the launch must report, never silently prove."""
    tensors = dict(_LBD_TENSORS)
    tensors["counter_ptr"] = _t(0x2000, numel=1, init=None)
    _, reports = _solve(parse_ttir(_lbd_ttir("acq_rel")), {}, tensors)
    assert reports


def test_lbd_hardcoded_gate_races_on_the_universal_grid_claim():
    """A gate hardcoding `old == 3` is race-free ONLY when the launch grid
    is exactly 4 — under T1's every-grid claim the blocks with rank > 3
    store after the gate opens, and that must surface as a race (this is a
    real grid-contract bug the num_programs gate fixes)."""
    _, reports = _solve(
        parse_ttir(_lbd_ttir("acq_rel", gate="const")), {}, _LBD_TENSORS
    )
    assert reports


# ─────────────────── work queue (observation address) ───────────────────


def _wq_ttir(idx_lines, idx_ssa):
    return _module(
        "%head_ptr: !tt.ptr<i32>, %buf_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "%c2 = arith.constant 2 : i32",
        "%pid = tt.get_program_id x : i32",
        "%old = tt.atomic_rmw add, relaxed, gpu, %head_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        *idx_lines,
        f"%b = tt.addptr %buf_ptr, {idx_ssa} : !tt.ptr<i32>, i32",
        "tt.store %b, %pid : !tt.ptr<i32>",
    )


_WQ_TENSORS = {
    "head_ptr": _t(0x2000, numel=1, init=(0,)),
    "buf_ptr": _t(0x10000, numel=1024),
}


def test_work_queue_single_fetch_proved_t1():
    """store buf[atomic_add(head, 1)]: the observation feeds the address,
    admitted because the counting axiom pins it to the rank."""
    _, reports = _solve(parse_ttir(_wq_ttir([], "%old")), {}, _WQ_TENSORS)
    assert reports == []


def test_work_queue_narrow_slots_race():
    """Mutation twin: buf[idx // 2] — adjacent ranks share a slot."""
    _, reports = _solve(
        parse_ttir(_wq_ttir(["%h = arith.divsi %old, %c2 : i32"], "%h")),
        {},
        _WQ_TENSORS,
    )
    assert reports


def test_work_queue_plain_load_is_unsupported():
    """B.4's plain-load twin: a LOADED head value in the address is
    data-dependent indirection — outside the model on both tracks, so the
    honest verdict is unsupported (indirect-address), not a proof."""
    text = _module(
        "%head_ptr: !tt.ptr<i32>, %buf_ptr: !tt.ptr<i32>",
        "%pid = tt.get_program_id x : i32",
        "%old = tt.load %head_ptr : !tt.ptr<i32>",
        "%b = tt.addptr %buf_ptr, %old : !tt.ptr<i32>, i32",
        "tt.store %b, %pid : !tt.ptr<i32>",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "indirect-address"


# ─────────────────── boundaries stay fail-closed ───────────────────


def test_float_atomic_result_stays_datadep():
    """fadd on a float counter: the result must NOT bind Observed — a
    downstream mask is widened (mask_dropped), same as before part B."""
    text = _module(
        "%fcounter_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
        "%true = arith.constant true",
        "%cst = arith.constant 1.000000e+00 : f32",
        "%old = tt.atomic_rmw fadd, acq_rel, gpu, %fcounter_ptr, %cst, %true : "
        "(!tt.ptr<f32>, f32, i1) -> f32",
        "%done = arith.cmpf oeq, %old, %cst : f32",
        "tt.store %out_ptr, %cst, %done : !tt.ptr<f32>",
    )
    g = parse_ttir(text)
    store = g.accesses[1]
    assert store.kind == "store"
    assert store.mask is None and store.mask_dropped
    enc = encode_graph(
        g,
        {},
        {"fcounter_ptr": _t(0x2000, numel=1), "out_ptr": _t(0x3000, numel=1)},
    )
    assert enc.records[0].old_value is None  # no Int observation for floats
    assert 1 in enc.uncertain_event_ids


def test_loop_carried_rmw_observation_not_modeled():
    """An RMW inside scf.for observes a different value per iteration; one
    var cannot stand for all of them, so downstream uses are widened
    (uncertain), never treated as exact."""
    text = _module(
        "%head_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%c4 = arith.constant 4 : i32",
        "%pid = tt.get_program_id x : i32",
        "scf.for %i = %c0 to %c4 step %c1  : i32 {",
        "%old = tt.atomic_rmw add, relaxed, gpu, %head_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%m = arith.cmpi eq, %old, %c0 : i32",
        "tt.store %out_ptr, %pid, %m : !tt.ptr<i32>",
        "scf.yield",
        "}",
    )
    g = parse_ttir(text)
    enc = encode_graph(
        g,
        {},
        {"head_ptr": _t(0x2000, numel=1, init=(0,)), "out_ptr": _t(0x3000, numel=1)},
    )
    store_seq = next(i for i, a in enumerate(g.accesses) if a.kind == "store")
    assert store_seq in enc.uncertain_event_ids
    rmw_rec = next(r for r in enc.records if r.atomic_kind == "rmw")
    assert rmw_rec.old_value is None


# ─────────────────── client wiring (init capture + tier loop) ────────────


def test_client_proves_lbd_via_captured_init_values():
    """Full client loop with a synthetic launch: pre_warmup captures the
    counter's PRE-LAUNCH zeros into GlobalTensor.init_values, and the T1
    solve lands on a proof."""
    det = CompiledRaceDetector(confirm_races=False)
    partial = torch.zeros(4, dtype=torch.float32)
    counter = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.float32)
    jit_fn = SimpleNamespace(arg_names=["partial_ptr", "counter_ptr", "out_ptr"])
    det.pre_warmup_callback(jit_fn, partial, counter, out, grid=(4,))
    assert det._launch_tensors["counter_ptr"].init_values == (0,)
    assert det._launch_tensors["partial_ptr"].init_values is None  # float
    det.post_warmup_callback(
        jit_fn, SimpleNamespace(asm={"ttir": _lbd_ttir("acq_rel")})
    )
    det.finalize()
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1"


def test_client_relaxed_lbd_reports():
    det = CompiledRaceDetector(confirm_races=False)
    partial = torch.zeros(4, dtype=torch.float32)
    counter = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.float32)
    jit_fn = SimpleNamespace(arg_names=["partial_ptr", "counter_ptr", "out_ptr"])
    det.pre_warmup_callback(jit_fn, partial, counter, out, grid=(4,))
    det.post_warmup_callback(
        jit_fn, SimpleNamespace(asm={"ttir": _lbd_ttir("relaxed")})
    )
    det.finalize()
    assert det.last_global_status == "races"
    assert det.last_global_reports


def test_capture_init_values_guards():
    cap = CompiledRaceDetector._capture_init_values
    assert cap(torch.arange(4, dtype=torch.int32), True) == (0, 1, 2, 3)
    assert cap(torch.zeros(2, dtype=torch.float32), True) is None
    assert cap(torch.zeros(2, dtype=torch.int32), False) is None  # non-contig
    big = torch.zeros(2000, dtype=torch.int32)
    assert cap(big, True) is None  # over the cap


def test_observed_term_is_leaf():
    o = Observed(3)
    assert mentions_observed(o)
    assert not mentions_observed(None)
