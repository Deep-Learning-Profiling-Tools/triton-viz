"""Semantic checks for exact simplification of small integer snapshots.

The oracle is the original array encoding, restricted only by the existing
consumer domain premise. These tests never assume the launch's pid extent
when checking a table lookup or collapse the two program copies together.
"""

import pytest
from z3 import Array, Int, IntSort, Select, Solver, sat, substitute, unsat

from triton_viz.clients.common.ttir_reader import (
    AccessGraph,
    Bin,
    Cmp,
    Const,
    Loaded,
    Pid,
    UnsupportedTTIR,
    parse_ttir,
)
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    _RaceEnv,
    encode_graph,
    encode_graph_t0,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)


def _empty_graph():
    return AccessGraph("snapshot_lookup", [], [], None, multipath=True)


def _env(values, *, numel=None, symbolic_params=False):
    meta = GlobalTensor(
        data_ptr=0x100000,
        elem_size=8,
        numel=len(values) if numel is None else numel,
        snapshot=values,
    )
    return _RaceEnv(
        _empty_graph(),
        {},
        tensors={"table": meta},
        multipath=True,
        symbolic_params=symbolic_params,
    )


def _load(offset=None, *, mask=None, other=None):
    return Loaded(0, "table", Pid(0) if offset is None else offset, mask, other)


@pytest.mark.parametrize(
    "values",
    [
        (7,),
        (-11, -11, -11),
        (True, True, True),
        (True, False, True),
        (0, 1, 2),
        (9, 5, 1, -3),
        (0, 29, 64),
        (-9, 23, -4, 23),
        (-(1 << 63), (1 << 63) - 1, 0),
        tuple((i * i) % 37 for i in range(129)),
    ],
    ids=[
        "singleton",
        "constant",
        "bool-constant",
        "bool-nonaffine",
        "identity",
        "affine-negative-step",
        "varlen",
        "nonaffine-signed",
        "int64-extremes",
        "large-table",
    ],
)
@pytest.mark.parametrize(
    "offset",
    [Pid(0), Bin("-", Bin("*", Const(3), Pid(0)), Const(2))],
    ids=["pid", "affine-index"],
)
def test_snapshot_lookup_matches_array_on_its_domain(values, offset):
    env = _env(values)
    leaf = _load(offset)
    actual = env.eval(leaf)
    original = Array("original_snapshot", IntSort(), IntSort())
    solver = Solver()
    solver.add(*env.snapshot_assumptions)
    solver.add(*(Select(original, i) == int(v) for i, v in enumerate(values)))
    solver.add(*env.domain_premises_for([leaf]))
    solver.add(actual != Select(original, env.eval(offset)))
    assert solver.check() == unsat
    assert env.used_snapshot
    assert not env.free_loaded


def test_short_snapshot_keeps_unobserved_in_bounds_entries_free():
    # Invalid/incomplete metadata must not turn an uncaptured table suffix
    # into an extrapolated affine function or a final constant branch.
    env = _env((0, 1), numel=4)
    leaf = _load()
    value = env.eval(leaf)
    solver = Solver()
    solver.add(*env.snapshot_assumptions, *env.domain_premises_for([leaf]))
    solver.add(env.eval(Pid(0)) == 3, value == 91)
    assert solver.check() == sat


@pytest.mark.parametrize("values", [(), (4,), (0, 1, 2), (0, 29, 64)])
@pytest.mark.parametrize("offset", [Pid(0), Const(-1), Const(0), Const(10)])
def test_lookup_matches_original_array_even_without_consumer_domain(values, offset):
    # Atomic operands can consume the expression without an access-local
    # table domain. Equivalence must also hold for their uncaptured indices.
    env = _env(values, numel=max(8, len(values)))
    actual = env.eval(_load(offset))
    solver = Solver()
    solver.add(*env.snapshot_assumptions)
    solver.add(actual != Select(env._snap_arrays["table"], env.eval(offset)))
    assert solver.check() == unsat


def test_lookup_domain_rejects_active_oob_and_keeps_masked_other():
    env = _env((0, 29, 64))
    pid = env.eval(Pid(0))
    active = _load()
    env.eval(active)
    for invalid in (-1, 3, 100000):
        solver = Solver()
        solver.add(*env.snapshot_assumptions, *env.domain_premises_for([active]))
        solver.add(pid == invalid)
        assert solver.check() == unsat

    masked = _load(mask=Cmp("slt", Pid(0), Const(0)), other=Const(-19))
    value = env.eval(masked)
    solver = Solver()
    solver.add(*env.snapshot_assumptions, *env.domain_premises_for([masked]))
    solver.add(pid == 100000)
    assert solver.check() == sat
    solver.add(value != -19)
    assert solver.check() == unsat


def test_masked_unspecified_values_remain_independent_between_copies():
    env = _env((0, 1, 2))
    leaf = _load(mask=Cmp("slt", Pid(0), Const(0)))
    value = env.eval(leaf)
    assert len(env.pad_vars) == 1
    pid = env.eval(Pid(0))
    pad = env.pad_vars[0]
    pa, pb = Int("pid_a"), Int("pid_b")
    pad_a, pad_b = (
        Array("pad_a", IntSort(), IntSort()),
        Array("pad_b", IntSort(), IntSort()),
    )
    a = substitute(value, (pid, pa), (pad, pad_a))
    b = substitute(value, (pid, pb), (pad, pad_b))
    solver = Solver()
    solver.add(*env.snapshot_assumptions, pa == 4, pb == 5, a == 11, b == -7)
    for domain in env.domain_premises_for([leaf]):
        solver.add(substitute(domain, (pid, pa)), substitute(domain, (pid, pb)))
    assert solver.check() == sat


def test_adjacent_table_reads_use_each_program_copys_own_index():
    env = _env((0, 29, 64))
    lo = _load()
    hi = Loaded(1, "table", Bin("+", Pid(0), Const(1)), None, None)
    length = env.eval(Bin("-", hi, lo))
    pid = env.eval(Pid(0))
    pa, pb = Int("sequence_a"), Int("sequence_b")
    a, b = substitute(length, (pid, pa)), substitute(length, (pid, pb))
    solver = Solver()
    solver.add(*env.snapshot_assumptions, pa == 0, pb == 1)
    for domain in env.domain_premises_for([lo, hi]):
        solver.add(substitute(domain, (pid, pa)), substitute(domain, (pid, pb)))
    assert solver.check() == sat
    solver.add(a != 29)
    assert solver.check() == unsat
    solver = Solver()
    solver.add(*env.snapshot_assumptions, pa == 0, pb == 1, b != 35)
    assert solver.check() == unsat


def test_content_free_lookup_does_not_use_snapshot_or_domain():
    env = _env((0, 1, 2), symbolic_params=True)
    leaf = _load()
    value = env.eval(leaf)
    assert not env.used_snapshot
    assert env.free_loaded == {0}
    assert env.snapshot_assumptions == []
    assert env.domain_premises_for([leaf]) == ()
    solver = Solver()
    solver.add(env.eval(Pid(0)) == 1, value == 500)
    assert solver.check() == sat


def _module(args, *lines):
    body = "\n    ".join(lines)
    return (
        "module {\n"
        f"  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {body}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


def test_written_snapshot_and_content_free_addresses_still_refuse():
    graph = parse_ttir(
        _module(
            "%table: !tt.ptr<i32>, %out: !tt.ptr<i32>",
            "%c1 = arith.constant 1 : i32",
            "%pid = tt.get_program_id x : i32",
            "%ip = tt.addptr %table, %pid : !tt.ptr<i32>, i32",
            "%i = tt.load %ip : !tt.ptr<i32>",
            "%op = tt.addptr %out, %i : !tt.ptr<i32>, i32",
            "tt.store %op, %c1 : !tt.ptr<i32>",
        ),
        multipath=True,
    )
    tensors = {
        "table": GlobalTensor(0x100000, 4, 3, snapshot=(0, 1, 2)),
        "out": GlobalTensor(0x100000, 4, 64),
    }
    with pytest.raises(UnsupportedTTIR, match="overlaps the source") as err:
        encode_graph(graph, {}, tensors, multipath=True)
    assert err.value.kind == "indirect-address"
    with pytest.raises(UnsupportedTTIR, match="no usable snapshot"):
        encode_graph_t0(graph, multipath=True)


def _varlen_graph(stride):
    return parse_ttir(
        _module(
            "%table: !tt.ptr<i32>, %out: !tt.ptr<i32>",
            "%c0 = arith.constant 0 : i32",
            "%c1 = arith.constant 1 : i32",
            "%c63 = arith.constant 63 : i32",
            "%c64 = arith.constant 64 : i32",
            f"%stride = arith.constant {stride} : i32",
            "%pid = tt.get_program_id x : i32",
            "%next = arith.addi %pid, %c1 : i32",
            "%lp = tt.addptr %table, %pid : !tt.ptr<i32>, i32",
            "%lo = tt.load %lp : !tt.ptr<i32>",
            "%hp = tt.addptr %table, %next : !tt.ptr<i32>, i32",
            "%hi = tt.load %hp : !tt.ptr<i32>",
            "%len = arith.subi %hi, %lo : i32",
            "%roundup = arith.addi %len, %c63 : i32",
            "%count = arith.divsi %roundup, %c64 : i32",
            "%base = arith.muli %pid, %stride : i32",
            "%lane = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "scf.for %k = %c0 to %count step %c1 : i32 {",
            "%advance = arith.muli %k, %len : i32",
            "%row = arith.addi %base, %advance : i32",
            "%rows = tt.splat %row : i32 -> tensor<64xi32>",
            "%offset = arith.addi %rows, %lane : tensor<64xi32>",
            "%chunk = arith.muli %k, %c64 : i32",
            "%chunks = tt.splat %chunk : i32 -> tensor<64xi32>",
            "%at = arith.addi %chunks, %lane : tensor<64xi32>",
            "%lens = tt.splat %len : i32 -> tensor<64xi32>",
            "%mask = arith.cmpi slt, %at, %lens : tensor<64xi32>",
            "%outs = tt.splat %out : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>",
            "%op = tt.addptr %outs, %offset : tensor<64x!tt.ptr<i32>>, tensor<64xi32>",
            "%ones = tt.splat %c1 : i32 -> tensor<64xi32>",
            "tt.store %op, %ones, %mask : tensor<64x!tt.ptr<i32>>",
            "scf.yield",
            "}",
        ),
        multipath=True,
    )


@pytest.mark.parametrize("stride,races", [(64, False), (16, True)])
def test_varlen_rounded_loop_preserves_proof_and_cross_sequence_collision(
    stride, races
):
    graph = _varlen_graph(stride)
    tensors = {
        "table": GlobalTensor(0x100000, 4, 3, snapshot=(0, 29, 64)),
        "out": GlobalTensor(0x300000, 4, 256),
    }
    enc = encode_graph(graph, {}, tensors, multipath=True)
    assert enc.content_qualified and not enc.uncertain_event_ids
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (2, 1, 1)),
        arange_dict=enc.arange_dict,
        extra_assumptions=enc.assumptions,
    )
    reports = solver.find_races()
    assert bool(reports) == races
    if races:
        assert all(
            {r.witness_grid_a[0], r.witness_grid_b[0]} == {0, 1} for r in reports
        )
