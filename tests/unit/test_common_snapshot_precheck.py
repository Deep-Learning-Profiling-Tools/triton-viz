"""Common snapshot premises enable prechecks for ordinary linear accesses."""

import pytest
import z3

from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector import two_copy_symbolic_hb_solver as tc
from triton_viz.clients.symbolic_engine import SymbolicExpr

from .test_two_copy_symbolic_hb_solver import _scalar_store


def _snapshot():
    table = z3.Array("common_snapshot_table", z3.IntSort(), z3.IntSort())
    return (z3.Select(table, 0) == 7, z3.Select(table, 1) == 11)


def _linear_store_solver(*, assumptions=None, duplicate_lanes=False):
    lane = z3.Int("common_snapshot_lane")
    address = (
        4096 + 256 * SymbolicExpr.PID0 + 4 * (lane / 2 if duplicate_lanes else lane)
    )
    record = _scalar_store(
        address,
        event_id=0,
        elem_size=4,
        mask=z3.And(lane >= 0, lane < 64),
    )
    return tc.TwoCopySymbolicHBSolver(
        [record],
        grid=(z3.Int("common_snapshot_grid"), 1, 1),
        arange_dict={(0, 64, "lane"): (lane, None)},
        extra_assumptions=_snapshot() if assumptions is None else assumptions,
    )


def _overlapping_solver(stride=0):
    record = _scalar_store(4096 + stride * SymbolicExpr.PID0, event_id=0, elem_size=4)
    return tc.TwoCopySymbolicHBSolver(
        [record], grid=(2, 1, 1), extra_assumptions=_snapshot()
    )


def test_common_snapshots_enable_linear_cross_and_intra_proofs(monkeypatch):
    solver = _linear_store_solver()
    assert all(
        cs.conflict_precheck_features(e.addr) == (False, False) for e in solver.events
    )

    def no_full_pair_query(*args, **kwargs):
        raise AssertionError("the linear necessary condition should prove disjointness")

    monkeypatch.setattr(solver, "_race_query_is_sat", no_full_pair_query)
    assert solver.find_races() == []
    assert [(kind, answer) for kind, _, answer in solver.query_stats] == [
        ("cross", False),
        ("intra", False),
    ]
    assert not solver.enum_used
    # The complete base system still independently certifies feasibility.
    monkeypatch.setattr(solver, "_conflict_precheck", no_full_pair_query)
    assert solver.check_feasibility()


@pytest.mark.parametrize("stride", [0, 2])
def test_common_snapshots_preserve_overlapping_and_partial_byte_races(
    monkeypatch, stride
):
    solver = _overlapping_solver(stride)
    original = solver._race_query_is_sat
    full_pairs = []

    def counted_full_pair(*args, **kwargs):
        full_pairs.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_race_query_is_sat", counted_full_pair)
    assert not solver._conflict_precheck(*solver.events)
    reports = solver.find_races()
    assert full_pairs and reports
    for report in reports:
        a, b = report.witness_grid_a, report.witness_grid_b
        assert a != b
        # With stride=2 these four-byte stores overlap despite unequal starts.
        assert abs(stride * (a[0] - b[0])) < 4
    assert not solver.enum_used


def test_common_snapshots_preserve_duplicate_lanes_in_one_instance():
    solver = _linear_store_solver(duplicate_lanes=True)
    reports = solver.find_races()
    assert reports
    assert any("single program instance" in report.reason for report in reports)
    assert any(kind == "intra" and answer for kind, _, answer in solver.query_stats)


def test_common_snapshot_feature_updates_after_tuple_and_list_changes(monkeypatch):
    solver = _linear_store_solver(assumptions=())
    calls = []

    def capture(conditions, *args, **kwargs):
        calls.append(conditions)
        return False

    monkeypatch.setattr(tc, "conflict_impossible", capture)

    def check(expected_calls):
        assert not solver._conflict_precheck(*solver.events)
        assert len(calls) == expected_calls

    check(0)
    solver.extra_assumptions = _snapshot()
    check(1)
    check(2)
    solver.extra_assumptions = ()
    check(2)
    mutable = []
    solver.extra_assumptions = mutable
    check(2)
    mutable.extend(_snapshot())
    check(3)
    mutable.clear()
    check(3)
    mutable.append(_snapshot()[0])
    check(4)


def test_common_snapshot_feature_scan_is_reused_for_immutable_sources(monkeypatch):
    solver = _linear_store_solver()
    scanned = []
    original = tc.conflict_precheck_features

    def counted_features(expression):
        scanned.append(expression)
        return original(expression)

    monkeypatch.setattr(tc, "conflict_precheck_features", counted_features)
    monkeypatch.setattr(tc, "conflict_impossible", lambda *args, **kwargs: False)
    solver._conflict_precheck(*solver.events)
    first_count = len(scanned)
    assert first_count == len(solver.events) + 1
    solver._conflict_precheck(*solver.events)
    assert len(scanned) == first_count
    solver.extra_assumptions = ()
    solver._conflict_precheck(*solver.events)
    assert len(scanned) == first_count + 1


@pytest.mark.parametrize("array_congruence", [False, True])
def test_infeasible_common_snapshots_never_become_a_feasibility_certificate(
    monkeypatch, array_congruence
):
    table = z3.Array("common_infeasible_table", z3.IntSort(), z3.IntSort())
    if array_congruence:
        # Select abstraction may forget array congruence. The original array
        # system, not its arithmetic relaxation, must decide feasibility.
        ia, ib = z3.Ints("common_infeasible_a common_infeasible_b")
        assumptions = (ia == ib, z3.Select(table, ia) == 7, z3.Select(table, ib) == 11)
    else:
        assumptions = (z3.Select(table, 0) == 7, z3.Select(table, 0) == 11)
    solver = _linear_store_solver(assumptions=assumptions)

    def no_full_pair_query(*args, **kwargs):
        raise AssertionError("disjoint linear addresses require no complete pair query")

    monkeypatch.setattr(solver, "_race_query_is_sat", no_full_pair_query)
    assert solver.find_races() == []
    monkeypatch.setattr(solver, "_conflict_precheck", no_full_pair_query)
    assert not solver.check_feasibility()
    assert not solver.enum_used


def test_unknown_common_snapshot_precheck_falls_back_to_complete_pair(monkeypatch):
    options = []

    class UnknownPrecheck:
        def set(self, **kwargs):
            options.append(kwargs)

        def add(self, *conditions):
            pass

        def check(self):
            return z3.unknown

    monkeypatch.setattr(cs.z3, "SolverFor", lambda logic: UnknownPrecheck())
    solver = _overlapping_solver(stride=2)
    assert solver.find_races()
    assert options and all(
        item == {"timeout": cs._PRECHECK_TIMEOUT_MS} for item in options
    )
    assert not solver.enum_used


@pytest.mark.parametrize("primed", [False, True])
def test_disabled_snapshot_precheck_is_respected_with_common_cache(monkeypatch, primed):
    solver = _linear_store_solver()
    if primed:
        assert solver._conflict_precheck(*solver.events)
    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
    assert not solver._conflict_precheck(*solver.events)
