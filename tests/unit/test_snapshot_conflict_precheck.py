"""Snapshot expressions and the UNSAT-only arithmetic conflict precheck."""

import z3

from triton_viz.clients.common.ttir_reader import (
    AccessGraph,
    Arange,
    Bin,
    Const,
    Loaded,
    LoopInfo,
    LoopVar,
    Pid,
)
from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    _RaceEnv,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_two_copy_symbolic_hb_solver import _scalar_store


def _terms(*, mask=None, other=None, next_axis=2):
    sequence = Bin("//", Pid(2), Const(2))
    next_sequence = Bin("+", Bin("//", Pid(next_axis), Const(2)), Const(1))
    lo = Loaded(0, "row", sequence, mask, other)
    hi = Loaded(1, "row", next_sequence, mask, other)
    chunks = Loaded(2, "chunks", sequence, None, None)
    count = Bin("//", Bin("+", Bin("-", hi, lo), Const(15)), Const(16))
    return sequence, lo, hi, chunks, count


def _env(*, row=(0, 29, 64), chunks=(0, 2, 5), numel=3, loop=False, symbolic=False):
    count = _terms()[-1]
    lp = (
        LoopInfo("%chunks", "%k", Const(-1), Bin("-", count, Const(1)), Const(1))
        if loop
        else None
    )
    return _RaceEnv(
        AccessGraph("correlated_chunks", [], [], lp, multipath=True),
        {},
        multipath=True,
        symbolic_params=symbolic,
        tensors={
            "row": GlobalTensor(0x100000, 4, numel, snapshot=row),
            "chunks": GlobalTensor(0x200000, 8, len(chunks), snapshot=chunks),
        },
    )


def _rwkv_store_solver(*, chunks=(0, 2, 5)):
    """Two sequence lengths and reverse chunk indices from the RWKV case."""
    env = _env(loop=True, chunks=chunks)
    _seq, lo, hi, chunk_base, count = _terms()
    reverse_chunk = Bin("-", Bin("-", count, Const(2)), LoopVar("%chunks"))
    chunk = Bin("+", chunk_base, reverse_chunk)
    head = Bin("%", Pid(2), Const(2))
    lane = Arange("%store_lane", 0, 64)
    offset = Bin(
        "+",
        Bin("*", Const(4096), Bin("+", Bin("*", Const(2), chunk), head)),
        lane,
    )
    address = 0x300000 + 4 * env.eval(offset)
    domain = env.domain_premises_for([lo, hi, chunk_base])
    record = _scalar_store(
        address,
        event_id=0,
        elem_size=4,
        mask=z3.And(*env.loop_premises, *domain),
    )
    record.copy_local_vars = env.loop_vars
    return TwoCopySymbolicHBSolver(
        [record],
        grid=(1, 1, z3.Int("grid_2")),
        arange_dict=env.arange_dict,
        extra_assumptions=tuple(env.snapshot_assumptions),
    )


def test_correlated_snapshot_cases_enable_cross_and_intra_conflict_proofs(monkeypatch):
    solver = _rwkv_store_solver()

    def no_full_pair_query(*args, **kwargs):
        raise AssertionError("the weaker conflict condition should prove disjointness")

    monkeypatch.setattr(solver, "_race_query_is_sat", no_full_pair_query)
    assert solver.find_races() == []
    assert [(kind, answer) for kind, _, answer in solver.query_stats] == [
        ("cross", False),
        ("intra", False),
    ]
    assert not solver.enum_used


def test_overlapping_chunk_table_preserves_a_cross_sequence_race():
    # Sequence 0 occupies chunks 0,1, sequence 1 occupies 1,2,3.
    # Treating their table indices or their loop ordinals as equal would
    # incorrectly remove the collision in chunk 1.
    solver = _rwkv_store_solver(chunks=(0, 1, 4))
    reports = solver.find_races()
    assert reports
    assert any(
        {r.witness_grid_a[2] // 2, r.witness_grid_b[2] // 2} == {0, 1} for r in reports
    )


def test_out_of_domain_snapshot_values_can_still_overlap():
    env = _env()
    chunks = _terms()[3]
    value = env.eval(Bin("+", chunks, Const(1)))
    pa, pb = z3.Ints("outside_a outside_b")
    original_pid = env.eval(Pid(2))
    a = z3.substitute(value, (original_pid, pa))
    b = z3.substitute(value, (original_pid, pb))
    conditions = [pa == 8, pb == 10, pa != pb, a == b]
    # Deliberately no consumer-domain premise, as for an atomic operand.
    # Both uncaptured table entries can hold the same integer.
    oracle = z3.Solver()
    oracle.add(*env.snapshot_assumptions, *conditions)
    assert oracle.check() == z3.sat
    assert not cs.conflict_impossible(conditions, ((pb, pa),))


def test_identical_select_occurrences_share_one_abstract_value():
    table = z3.Array("shared_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("shared_index")
    loaded = z3.Select(table, index)
    assert cs.conflict_impossible([loaded >= 7, loaded < 7])


def test_different_copy_indices_remain_independent_after_select_abstraction():
    table = z3.Array("index_table", z3.IntSort(), z3.IntSort())
    ia, ib = z3.Ints("index_a index_b")
    a, b = z3.Select(table, ia), z3.Select(table, ib)
    conditions = [ia == 0, ib == 1, a == 7, b == 11]
    oracle = z3.Solver()
    oracle.add(*conditions)
    assert oracle.check() == z3.sat
    # Shape correspondence is not an equality between program instances.
    assert not cs.conflict_impossible(conditions, ((ib, ia),))


def test_different_arrays_at_the_same_index_remain_independent():
    a = z3.Array("array_a", z3.IntSort(), z3.IntSort())
    b = z3.Array("array_b", z3.IntSort(), z3.IntSort())
    index = z3.Int("array_index")
    conditions = [z3.Select(a, index) == 7, z3.Select(b, index) == 11]
    oracle = z3.Solver()
    oracle.add(*conditions)
    assert oracle.check() == z3.sat
    assert not cs.conflict_impossible(conditions)


def test_ite_domain_removes_only_the_unreachable_array_branch():
    table = z3.Array("fallback_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("fallback_index")
    expression = z3.If(index == 0, 2, z3.If(index == 1, 3, z3.Select(table, index)))
    assert cs.conflict_impossible([index >= 0, index < 2, expression >= 4])
    conditions = [index == 2, expression == 4]
    oracle = z3.Solver()
    oracle.add(*conditions)
    assert oracle.check() == z3.sat
    assert not cs.conflict_impossible(conditions)
