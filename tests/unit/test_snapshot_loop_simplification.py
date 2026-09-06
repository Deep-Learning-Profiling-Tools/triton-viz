"""Exact finite-table constant propagation and single-trip loop regression tests."""

import pytest
from z3 import Solver, sat, simplify, unsat

from triton_viz.clients.common.ttir_reader import (
    AccessGraph,
    Bin,
    Cmp,
    Const,
    IterArgInfo,
    IterArgOffset,
    Loaded,
    LoopInfo,
    LoopVar,
    Pid,
)
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    _RaceEnv,
    encode_graph,
)

from .test_snapshot_simplification import _env, _varlen_graph


def _rounded_count(*, mask=None, other=None, next_axis=0):
    lo = Loaded(0, "table", Pid(0), mask, other)
    hi = Loaded(1, "table", Bin("+", Pid(next_axis), Const(1)), mask, other)
    return Bin("//", Bin("+", Bin("-", hi, lo), Const(63)), Const(64))


def _count_loop_env(
    values, *, start=0, numel=None, mask=None, other=None, symbolic_params=False
):
    count = _rounded_count(mask=mask, other=other)
    loop = LoopInfo(
        "%count_loop", "%k", Const(start), Bin("+", Const(start), count), Const(1)
    )
    graph = AccessGraph("rounded_loop", [], [], loop, multipath=True)
    return _RaceEnv(
        graph,
        {},
        multipath=True,
        symbolic_params=symbolic_params,
        tensors={
            "table": GlobalTensor(
                0x100000, 4, len(values) if numel is None else numel, snapshot=values
            )
        },
    )


@pytest.mark.parametrize(
    "values,expected",
    [
        ((0, 29, 64), 1),
        ((0, 0, 64), None),
        ((0, 0, 0), 0),
        ((0, 65, 130), 2),
    ],
)
def test_snapshot_rounded_length_is_constant_only_across_all_sequences(
    values, expected
):
    env = _env(values)
    assert env._snapshot_constant(_rounded_count()) == expected


@pytest.mark.parametrize(
    "values,counts",
    [
        ((0, 29, 64), (1, 1)),
        ((0, 0, 64), (0, 1)),
        ((0, 0, 0), (0, 0)),
        ((0, 65, 130), (2, 2)),
    ],
)
def test_rounded_loop_keeps_exact_iteration_count_and_source_domain(values, counts):
    env = _count_loop_env(values)
    # Query the actual loop premises: this also checks that eliminating a
    # loop variable from expressions never admits another iteration.
    pid = env.eval(Pid(0))
    for sequence, count in enumerate(counts):
        for iteration in (-1, 0, 1, 2):
            solver = Solver()
            solver.add(*env.snapshot_assumptions, *env.loop_premises)
            solver.add(pid == sequence, env.loop_var == iteration)
            assert solver.check() == (sat if 0 <= iteration < count else unsat)
    for sequence in (-1, 2, 100000):
        solver = Solver()
        solver.add(*env.snapshot_assumptions, *env.loop_premises)
        solver.add(pid == sequence, env.loop_var == 0)
        assert solver.check() == unsat
    assert env.zero_trip == (counts == (0, 0))
    assert env._binding("%count_loop").single_trip == (counts == (1, 1))


def test_zero_count_table_removes_body_accesses_only():
    graph = _varlen_graph(64)
    enc = encode_graph(
        graph,
        {},
        {
            "table": GlobalTensor(0x100000, 4, 3, snapshot=(0, 0, 0)),
            "out": GlobalTensor(0x300000, 4, 256),
        },
        multipath=True,
    )
    assert enc.content_qualified
    # Loads outside the zero-trip loop remain; its store disappears.
    assert {record.event_id for record in enc.records} == {0, 1}


@pytest.mark.parametrize("other", [None, Const(0)])
def test_masked_bounds_do_not_become_unconditional_constants(other):
    mask = Cmp("slt", Pid(0), Const(1))
    env = _env((0, 29, 64))
    assert env._snapshot_constant(_rounded_count(mask=mask, other=other)) is None
    bound_env = _count_loop_env((0, 29, 64), mask=mask, other=other)
    assert not bound_env._binding("%count_loop").single_trip
    # pid 1 has both loads disabled: with other=0 it takes no iterations;
    # without other its free values may permit a second iteration.
    solver = Solver()
    solver.add(*bound_env.snapshot_assumptions, *bound_env.loop_premises)
    solver.add(bound_env.eval(Pid(0)) == 1, bound_env.loop_var == 1)
    assert solver.check() == (sat if other is None else unsat)


def test_partial_snapshot_cannot_make_every_sequence_single_trip():
    env = _env((0, 29, 64), numel=4)
    assert env._snapshot_constant(_rounded_count()) is None
    bound_env = _count_loop_env((0, 29, 64), numel=4)
    assert not bound_env._binding("%count_loop").single_trip
    solver = Solver()
    solver.add(*bound_env.snapshot_assumptions, *bound_env.loop_premises)
    solver.add(bound_env.eval(Pid(0)) == 2, bound_env.loop_var == 1)
    # The uncaptured fourth entry may make the final segment need two chunks.
    assert solver.check() == sat


@pytest.mark.parametrize("mode", ["missing", "t0"])
def test_unusable_snapshot_cannot_produce_bound_constants(mode):
    env = _RaceEnv(
        AccessGraph("free_bound", [], [], None, multipath=True),
        {},
        multipath=True,
        symbolic_params=(mode == "t0"),
        tensors={
            "table": GlobalTensor(
                0x100000,
                4,
                3,
                snapshot=None if mode == "missing" else (0, 29, 64),
            )
        },
    )
    assert env._snapshot_constant(_rounded_count()) is None
    assert not env.used_snapshot
    assert not env.snapshot_assumptions


def test_independent_table_indices_and_remaining_pid_are_not_folded():
    env = _env((0, 29, 64))
    assert env._snapshot_constant(_rounded_count(next_axis=1)) is None
    assert env._snapshot_constant(Bin("+", _rounded_count(), Pid(1))) is None


@pytest.mark.parametrize("values,count", [((0, 29, 64), 1), ((0, 65, 130), 2)])
def test_single_trip_pointer_keeps_initial_offset_with_nonzero_induction_start(
    values, count
):
    env = _count_loop_env(values, start=5)
    initial = Bin("+", Bin("*", Pid(0), Const(7)), Const(11))
    env.graph.iter_args[0] = IterArgInfo(0, "out", initial, Const(23), "%count_loop")
    offset = env.eval(IterArgOffset(0))
    induction = env.eval(LoopVar("%count_loop"))
    if count == 1:
        # The sole pointer is the iter_arg's initial pointer, while the
        # induction value is the lower bound 5. Neither uses k anymore.
        assert simplify(offset - env.eval(initial)).as_long() == 0
        assert simplify(induction).as_long() == 5
    else:
        # Two trips must retain the second pointer advance, not silently
        # adopt the new single-trip shortcut.
        solver = Solver()
        solver.add(*env.snapshot_assumptions, *env.loop_premises)
        solver.add(env.eval(Pid(0)) == 1, env.loop_var == 1)
        assert solver.check() == sat
        solver.add(offset != 41)
        assert solver.check() == unsat
