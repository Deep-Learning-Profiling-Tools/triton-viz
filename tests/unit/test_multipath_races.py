"""Encoder, solver and client tests for multipath capture (Route 3, L2).

Every proof here has a mutation twin that must flip to a report, and every
report's witnesses must satisfy the path predicates of both records (the
faithfulness obligation of design §5: an instance walks exactly one path,
and the records active under it are precisely its accesses).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from z3 import IntVal, Solver, sat, substitute, unsat

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
from triton_viz.clients.symbolic_engine import SymbolicExpr

from .test_t1_rmw_static import _module

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / f"{name}_sm80.ttir").read_text()


def _t(ptr, numel=1 << 16, elem=4, init=None):
    return GlobalTensor(data_ptr=ptr, elem_size=elem, numel=numel, init_values=init)


# disjoint allocations (the in-bounds premise turns overlapping fake bases
# into cross-tensor conflicts)
XO = {"x_ptr": _t(0x100000, numel=1 << 14), "out_ptr": _t(0x200000, numel=1 << 14)}


def _mp(text):
    return parse_ttir(text, multipath=True)


def _t1(graph, params, tensors, grid=(4, 1, 1)):
    enc = encode_graph(graph, params, tensors, multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, grid),
        arange_dict=enc.arange_dict,
        enum_fallback_grid=grid,
    )
    return enc, solver.find_races()


def _t0(graph):
    reports = []
    for _name, enc in encode_graph_t0(graph):
        solver = TwoCopySymbolicHBSolver(
            enc.records,
            grid=symbolic_grid(enc, None, t0=True),
            arange_dict=enc.arange_dict,
        )
        reports += solver.find_races()
    return reports


def _pids(rep):
    return rep.witness_grid_a[0], rep.witness_grid_b[0]


# ───────────────────── early-return guards ─────────────────────


def test_early_return_guard_proves_at_t0():
    """``if pid*64 >= T: return`` then a per-pid block: the guard is a
    linear path predicate, so the any-input/any-grid rung proves it."""
    g = _mp(_read("early_return_pid"))
    assert t0_linearity_gate(g)
    assert _t0(g) == []
    _, reports = _t1(g, {"n": 256, "T": 256}, XO)
    assert reports == []


def test_guard_race_witnesses_satisfy_the_guard():
    """Every instance past the guard writes out[0]: a race whose
    witnesses must BOTH pass the guard (pid < T), never a returned pid."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %T: i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi sge, %pid, %T : i32",
        "cf.cond_br %g, ^bb1, ^bb2",
        "^bb1:  // pred: ^bb0",
        "tt.return",
        "^bb2:  // pred: ^bb0",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
        "tt.return",
    )
    g = _mp(text)
    _, reports = _t1(g, {"T": 2}, {"out_ptr": _t(0x20000)}, grid=(4, 1, 1))
    assert reports
    for rep in reports:
        a, b = _pids(rep)
        assert a != b and a < 2 and b < 2


def test_opposite_arms_exclude_each_other_per_instance():
    """Mutation test of path-condition composition (design §5.2): the two
    arms of one guard writing the same cell are UNSAT together for ONE
    instance and SAT for two instances that satisfy both predicates."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi eq, %pid, %c0 : i32",
        "cf.cond_br %g, ^bb1, ^bb2",
        "^bb1:  // pred: ^bb0",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
        "tt.return",
        "^bb2:  // pred: ^bb0",
        "tt.store %out_ptr, %c0 : !tt.ptr<i32>",
        "tt.return",
    )
    g = _mp(text)
    enc = encode_graph(g, {}, {"out_ptr": _t(0x20000)}, multipath=True)
    then_store, else_store = enc.records
    same = Solver()
    same.add(then_store.active, else_store.active)
    assert same.check() == unsat
    cross = Solver()
    cross.add(
        substitute(then_store.active, (SymbolicExpr.PID0, IntVal(0))),
        substitute(else_store.active, (SymbolicExpr.PID0, IntVal(1))),
    )
    assert cross.check() == sat
    _, reports = _t1(g, {}, {"out_ptr": _t(0x20000)})
    # the else arm races with itself across instances (pids k, l != 0);
    # the then/else pair's witnesses are pid 0 and some other pid
    cross_arm = [
        r
        for r in reports
        if {r.first_record.event_id, r.second_record.event_id} == {0, 1}
    ]
    assert cross_arm
    for rep in cross_arm:
        assert 0 in _pids(rep) and set(_pids(rep)) != {0}


def test_loaded_value_guard_keeps_proofs_and_widens_reports():
    """``if y == -1: return`` with y loaded: the fall-through is widened.
    Widening only enlarges footprints, so a disjoint kernel still PROVES;
    a racy one yields only widened reports, which the client never
    certifies as a definite race."""
    g = _mp(_read("early_return_loaded"))
    enc = encode_graph(g, {"n": 256}, dict(XO, idx_ptr=_t(0x30000)), multipath=True)
    assert enc.uncertain_event_ids == {1, 2}
    assert _t0(g) == []

    racy = _module(
        "%idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%c-1 = arith.constant -1 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%ip = tt.addptr %idx_ptr, %pid : !tt.ptr<i32>, i32",
        "%y = tt.load %ip : !tt.ptr<i32>",
        "%g = arith.cmpi eq, %y, %c-1 : i32",
        "cf.cond_br %g, ^bb1, ^bb2",
        "^bb1:  // pred: ^bb0",
        "tt.return",
        "^bb2:  // pred: ^bb0",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
        "tt.return",
    )
    det = CompiledRaceDetector(confirm_races=False, ladder_level=2)
    jit = SimpleNamespace(arg_names=["idx_ptr", "out_ptr"])
    det.pre_warmup_callback(
        jit,
        torch.zeros(64, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
        grid=(4,),
    )
    det.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": racy}))
    det.finalize()
    assert det.last_global_status == "unsupported"
    assert "over-approximation" in (det.last_global_reason or "")


def test_nested_guards_with_a_merge_prove_at_t0_and_the_mutant_races():
    """pid 0 writes [0, 64) (unless n < 0, then it returns); pid k > 0
    writes [64k + n, 64k + n + 64): disjoint for every n and T, so the
    Select-bound merge value proves at T0. Merging pid 0 onto base 64
    instead collides with pid 1 at n = 0."""
    g = _mp(_read("nested_guard_merge"))
    assert t0_linearity_gate(g)
    assert _t0(g) == []
    mutant = _read("nested_guard_merge").replace(
        "^bb5(%c0_i32 : i32)", "^bb5(%c64_i32 : i32)"
    )
    gm = _mp(mutant)
    assert _t0(gm)
    _, reports = _t1(gm, {"n": 0, "T": 4}, XO)
    assert reports
    assert all(set(_pids(r)) == {0, 1} for r in reports)


# ───────────────────── loops ─────────────────────


def test_guard_then_loop_is_input_dependent():
    """Offsets 64·pid + k·T past the guard 64·pid < T: some T (e.g. 100)
    lets pid 1's block reach pid 0's next iteration, so T0 is SAT and the
    kernel falls to T1, where T = 128 proves it for this input."""
    g = _mp(_read("guard_then_loop"))
    assert not t0_linearity_gate(g)  # k·T is symbolic×symbolic at T0
    _, reports = _t1(g, {"n": 3, "T": 128}, XO, grid=(2, 1, 1))
    assert reports == []
    _, reports = _t1(g, {"n": 3, "T": 100}, XO, grid=(2, 1, 1))
    assert reports


def test_nested_loops_prove_at_t1_and_the_flattened_mutant_races():
    g = _mp(_read("nested_loops"))
    assert not t0_linearity_gate(g)  # pid·n is symbolic×symbolic at T0
    _, reports = _t1(g, {"n": 2, "m": 3}, XO)
    assert reports == []
    racy = _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32, %m: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "scf.for %i = %c0 to %n step %c1  : i32 {",
        "scf.for %j = %c0 to %m step %c1  : i32 {",
        "%a = arith.muli %pid, %n : i32",
        "%b = arith.muli %i, %m : i32",
        "%ab = arith.addi %a, %b : i32",
        "%off = arith.addi %ab, %j : i32",
        "%p = tt.addptr %x_ptr, %off : !tt.ptr<f32>, i32",
        "%v = tt.load %p : !tt.ptr<f32>",
        "%q = tt.addptr %out_ptr, %off : !tt.ptr<f32>, i32",
        "tt.store %q, %v : !tt.ptr<f32>",
        "}",
        "}",
    )
    _, reports = _t1(_mp(racy), {"n": 2, "m": 3}, XO)
    assert reports


def test_sequential_loops_race_at_n2_and_prove_at_n1():
    g = _mp(_read("sequential_loops"))
    _, reports = _t1(g, {"n": 2}, XO)
    assert reports and all(r.first_record.tensor_name == "out_ptr" for r in reports)
    _, reports = _t1(g, {"n": 1}, XO)
    assert reports == []


def test_grid_stride_pid_linear_bound_at_t1():
    """``for row in range(pid, n_rows, NUM_PRGMS)``: the single-path T1
    encoder demands a concrete lower bound; at L2 the pid-linear bound
    keeps the T0 existence premise. The kernel is race-free only for
    grids of at most NUM_PRGMS instances (pid 4 walks pid 0's rows), so
    the ANY-grid T1 query finds out-of-extent witnesses only (the
    launch-scoped rung, checked through the client below); stride 32
    overlaps the next row within the extent."""
    g = parse_ttir(_read("grid_stride"))  # single-path parse: one loop
    with pytest.raises(UnsupportedTTIR, match="not concrete at launch"):
        encode_graph(g, {"n_rows": 8, "stride": 64}, XO)
    _, reports = _t1(g, {"n_rows": 8, "stride": 64}, XO)
    assert reports
    assert all(max(_pids(r)) >= 4 for r in reports)
    _, reports = _t1(g, {"n_rows": 8, "stride": 32}, XO)
    assert any(max(_pids(r)) < 4 for r in reports)


def test_loop_under_scf_if_proves_at_t0_and_the_inverted_mutant_races():
    g = _mp(_read("loop_under_if"))
    assert t0_linearity_gate(g)
    assert _t0(g) == []
    mutant = _read("loop_under_if").replace(
        "arith.cmpi eq, %pid, %c0_i32", "arith.cmpi ne, %pid, %c0_i32"
    )
    _, reports = _t1(_mp(mutant), {"n": 2}, XO)
    assert reports


def test_rmw_inside_a_nested_loop_stays_footprint_only():
    """The RMW observation model is defined for non-loop atomics only
    (one observation cannot stand for one per iteration); nesting must
    not re-enable it, and the counting axiom stays off (design §5.3)."""
    text = _module(
        "%cnt_ptr: !tt.ptr<i32>, %n: i32, %m: i32",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "scf.for %i = %c0 to %n step %c1  : i32 {",
        "scf.for %j = %c0 to %m step %c1  : i32 {",
        "%o = tt.atomic_rmw add, acq_rel, gpu, %cnt_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "}",
        "}",
    )
    g = _mp(text)
    (rmw,) = g.accesses
    assert rmw.in_loop and len(rmw.loops) == 2
    enc = encode_graph(
        g,
        {"n": 2, "m": 2},
        {"cnt_ptr": _t(0x40000, numel=1, init=(0,))},
        multipath=True,
    )
    (rec,) = enc.records
    assert rec.old_value is None
    assert len(rec.premises) == 2  # one existence premise per loop level
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=symbolic_grid(enc, (4, 1, 1)), arange_dict=enc.arange_dict
    )
    assert not solver._counting


# ───────────────────── the design's §6.1 fixture ─────────────────────


def _role_split_ttir(*, writer_sem="release", spin_sem="acquire", scope="gpu"):
    """Producer/consumer with an EARLY-RETURN role split: pid 0 publishes
    and returns (a cf.* block), every other pid awaits then reads."""
    return _module(
        "%flag_ptr: !tt.ptr<i32>, %data_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%isp = arith.cmpi eq, %pid, %c0 : i32",
        "cf.cond_br %isp, ^bb1, ^bb2",
        "^bb1:  // pred: ^bb0",
        "tt.store %data_ptr, %c1 : !tt.ptr<i32>",
        f"%x = tt.atomic_rmw exch, {writer_sem}, {scope}, %flag_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "tt.return",
        "^bb2:  // pred: ^bb0",
        "scf.while : () -> () {",
        f"%o = tt.atomic_rmw add, {spin_sem}, {scope}, %flag_ptr, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "%v = tt.load %data_ptr : !tt.ptr<i32>",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %op, %v : !tt.ptr<i32>",
        "tt.return",
    )


_PC_TENSORS = {
    "flag_ptr": _t(0x2000, numel=1, init=(0,)),
    "data_ptr": _t(0x3000, numel=1),
    "out_ptr": _t(0x4000, numel=64),
}


def test_role_split_producer_consumer_proves_under_termination():
    with pytest.raises(UnsupportedTTIR, match="cf.cond_br"):
        parse_ttir(_role_split_ttir())
    g = _mp(_role_split_ttir())
    enc, reports = _t1(g, {}, _PC_TENSORS)
    assert enc.assumes_termination
    assert reports == []


@pytest.mark.parametrize(
    "mutation",
    [dict(writer_sem="relaxed"), dict(spin_sem="relaxed"), dict(scope="cta")],
    ids=["relaxed-writer", "relaxed-spinner", "cta-scope"],
)
def test_role_split_mutations_race_on_the_data_cell(mutation):
    _, reports = _t1(_mp(_role_split_ttir(**mutation)), {}, _PC_TENSORS)
    data = [
        r
        for r in reports
        if {r.first_record.tensor_name, r.second_record.tensor_name} == {"data_ptr"}
    ]
    assert data
    for rep in data:
        assert 0 in _pids(rep) and set(_pids(rep)) != {0}


# ───────────────────── through the client: the ladder switch ─────────────────────


def _drive(ttir, arg_names, args, grid, level):
    det = CompiledRaceDetector(confirm_races=False, ladder_level=level)
    jit = SimpleNamespace(arg_names=arg_names)
    det.pre_warmup_callback(jit, *args, grid=grid)
    det.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()
    return det


def _f32(n=1 << 14):
    return torch.zeros(n, dtype=torch.float32)


def test_client_l0_refuses_and_l2_proves_the_early_return_guard():
    names = ["x_ptr", "out_ptr", "n", "T"]
    args = (_f32(), _f32(), 256, 256)
    l0 = _drive(_read("early_return_pid"), names, args, (4,), 0)
    assert l0.last_global_status == "unsupported"
    assert (l0.last_global_reason or "").startswith("control-flow: line")
    assert "cf.cond_br" in (l0.last_global_reason or "")
    l2 = _drive(_read("early_return_pid"), names, args, (4,), 2)
    assert l2.last_global_status == "ok"
    assert l2.last_global_provenance == "proved@T0"
    assert l2.last_global_verdict["proved_scope"] == "any-params-any-grid"


def test_client_l0_refuses_and_l2_proves_the_grid_stride_loop():
    names = ["x_ptr", "out_ptr", "n_rows", "stride"]
    args = (_f32(), _f32(), 8, 64)
    l0 = _drive(_read("grid_stride"), names, args, (4,), 0)
    assert l0.last_global_status == "unsupported"
    assert "not concrete at launch" in (l0.last_global_reason or "")
    l2 = _drive(_read("grid_stride"), names, args, (4,), 2)
    assert l2.last_global_status == "ok"
    # any-grid SAT (pid >= NUM_PRGMS re-walks pid 0's rows), launch-pinned
    # UNSAT: the launch-scoped rung with grid-fragility evidence
    assert l2.last_global_provenance == "proved@T1-launch"
    assert l2.last_grid_fragile
    racy = _drive(_read("grid_stride"), names, (_f32(), _f32(), 8, 32), (4,), 2)
    assert racy.last_global_status == "races"


def test_client_l2_reports_the_nested_loop_mutant_and_proves_the_original():
    names = ["x_ptr", "out_ptr", "n", "m"]
    l0 = _drive(_read("nested_loops"), names, (_f32(), _f32(), 2, 3), (4,), 0)
    assert (l0.last_global_reason or "").startswith("nested-loop:")
    l2 = _drive(_read("nested_loops"), names, (_f32(), _f32(), 2, 3), (4,), 2)
    assert l2.last_global_status == "ok"
    assert l2.last_global_provenance == "proved@T1"


def test_client_differential_is_unavailable_for_multipath_graphs():
    """C3 has no multi-loop / block-graph enumerator: it must report
    unavailable (None), never a mismatch, for an L2-only graph."""
    det = CompiledRaceDetector(
        confirm_races=False, differential_check=True, ladder_level=2
    )
    jit = SimpleNamespace(arg_names=["x_ptr", "out_ptr", "n", "m"])
    det.pre_warmup_callback(jit, _f32(), _f32(), 2, 3, grid=(4,))
    det.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": _read("nested_loops")}))
    det.finalize()
    assert det.last_global_status == "ok"
    assert det.last_differential is None


def test_mixed_and_mask_row_proves_instead_of_phantom_overlap():
    """The same idiom through the encoder: with the bounds conjunct kept,
    rows of C elements per pid never overlap, so the widened access yields
    no report at all (the phantom WAW that single-path widening produced
    was confirmed only by the interpreter's `and`-truthiness artifact)."""
    text = _module(
        "%out_ptr: !tt.ptr<f32>, %tgt_ptr: !tt.ptr<i32>, %C: i32",
        "%c1 = arith.constant 1.0 : f32",
        "%cm1 = arith.constant -1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>",
        "%Cs = tt.splat %C : i32 -> tensor<256xi32>",
        "%bounds = arith.cmpi slt, %offs, %Cs : tensor<256xi32>",
        "%tp = tt.addptr %tgt_ptr, %pid : !tt.ptr<i32>, i32",
        "%tgt = tt.load %tp : !tt.ptr<i32>",
        "%g = arith.cmpi ne, %tgt, %cm1 : i32",
        "%gs = tt.splat %g : i1 -> tensor<256xi1>",
        "%m = arith.andi %bounds, %gs : tensor<256xi1>",
        "%row = arith.muli %pid, %C : i32",
        "%rs = tt.splat %row : i32 -> tensor<256xi32>",
        "%o = arith.addi %rs, %offs : tensor<256xi32>",
        "%ps = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>",
        "%p = tt.addptr %ps, %o : tensor<256x!tt.ptr<f32>>, tensor<256xi32>",
        "%vs = tt.splat %c1 : f32 -> tensor<256xf32>",
        "tt.store %p, %vs, %m : tensor<256x!tt.ptr<f32>>",
    )
    tensors = {
        "out_ptr": _t(0x200000, numel=1 << 14),
        "tgt_ptr": _t(0x300000, numel=64),
    }
    enc0 = encode_graph(parse_ttir(text), {"C": 64}, tensors)
    s0 = TwoCopySymbolicHBSolver(
        enc0.records, grid=symbolic_grid(enc0, (4, 1, 1)), arange_dict=enc0.arange_dict
    )
    assert s0.find_races()  # single-path: the dropped mask fabricates the overlap
    enc, reports = _t1(_mp(text), {"C": 64}, tensors)
    assert 1 in enc.uncertain_event_ids  # still widened
    assert reports == []
