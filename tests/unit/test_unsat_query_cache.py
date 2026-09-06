"""Reuse only completed, identical full symbolic UNSAT queries."""

import pytest
import z3

from triton_viz.clients.race_detector import two_copy_symbolic_hb_solver as tc
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from triton_viz.clients.symbolic_engine import SymbolicExpr

from .test_two_copy_symbolic_hb_solver import _scalar_store


def _repeated_stores(monkeypatch, *, stride=8, grid=(2, 1, 1), **kwargs):
    address = 4096 + stride * SymbolicExpr.PID0
    records = [
        _scalar_store(address, event_id=i, program_seq=i, elem_size=4) for i in range(2)
    ]
    core = tc.TwoCopySymbolicHBSolver(records, grid=grid, fence_order=False, **kwargs)
    # Exercise full-query reuse independently of optional conflict prechecks.
    monkeypatch.setattr(core, "_conflict_precheck", lambda *a, **kw: False)
    return core


def _record_checks(monkeypatch, core):
    original = core._race_query_is_sat
    results = []

    def checked(query, first, second):
        answer = original(query, first, second)
        results.append(answer)
        return answer

    monkeypatch.setattr(core, "_race_query_is_sat", checked)
    return results


def test_identical_full_unsat_queries_skip_solver_creation_and_keep_pair_stats(
    monkeypatch,
):
    core = _repeated_stores(monkeypatch)
    results = _record_checks(monkeypatch, core)
    original_solver = tc.Solver
    created = []

    def counted_solver(*args, **kwargs):
        query = original_solver(*args, **kwargs)
        created.append(query)
        return query

    monkeypatch.setattr(tc, "Solver", counted_solver)
    assert core.find_races() == []
    assert results == [False]
    assert len(created) == 1
    assert [(kind, answer) for kind, _, answer in core.query_stats] == [
        ("cross", False)
    ] * 4

    # A new public invocation must perform its own symbolic decision.
    assert core.find_races() == []
    assert results == [False, False]
    assert len(created) == 2


@pytest.mark.parametrize("stride", [0, 2])
def test_sat_queries_keep_independent_models_including_partial_byte_overlap(
    monkeypatch, stride
):
    core = _repeated_stores(monkeypatch, stride=stride)
    results = _record_checks(monkeypatch, core)
    reports = core.find_races()
    assert results == [True] * 4
    assert len(reports) == 3
    for report in reports:
        first, second = report.witness_grid_a, report.witness_grid_b
        assert first != second
        assert abs(stride * (first[0] - second[0])) < 4
        assert report.model


@pytest.mark.parametrize("changed", ["grid", "extra_tuple", "extra_list", "hb_cycle"])
def test_withdrawing_a_base_restriction_during_one_pass_cannot_reuse_old_unsat(
    monkeypatch, changed
):
    grid_size = z3.Int("unsat_cache_mutable_grid")
    core = _repeated_stores(monkeypatch, stride=0, grid=(grid_size, 1, 1))
    unrestricted_grid = core.grid_constraints
    if changed == "grid":
        core.grid_constraints = z3.And(unrestricted_grid, grid_size == 1)
    elif changed == "extra_tuple":
        core.extra_assumptions = (grid_size == 1,)
    elif changed == "extra_list":
        core.extra_assumptions = [grid_size == 1]
    else:
        core.hb[0][0] = z3.BoolVal(True)

    original = core._race_query_is_sat
    answers = []

    def checked_then_withdrawn(query, first, second):
        answer = original(query, first, second)
        answers.append(answer)
        if len(answers) == 1:
            assert not answer
            if changed == "grid":
                core.grid_constraints = unrestricted_grid
            elif changed == "extra_tuple":
                core.extra_assumptions = ()
            elif changed == "extra_list":
                core.extra_assumptions.clear()
            else:
                core.hb[0][0] = z3.BoolVal(False)
        return answer

    monkeypatch.setattr(core, "_race_query_is_sat", checked_then_withdrawn)
    assert core.find_races()
    assert answers == [False, True, True, True]


def test_equal_addresses_with_different_hb_relations_do_not_share_a_decision(
    monkeypatch,
):
    core = _repeated_stores(monkeypatch, stride=0)
    first_a = next(e for e in core.events if e.copy == "a" and e.event_id == 0)
    first_b = next(e for e in core.events if e.copy == "b" and e.event_id == 0)
    core.hb[first_a.idx][first_b.idx] = z3.BoolVal(True)
    answers = _record_checks(monkeypatch, core)
    reports = core.find_races()
    assert answers == [False, True, True, True]
    pairs = {(report.first.event_id, report.second.event_id) for report in reports}
    assert (0, 0) not in pairs
    assert (0, 1) in pairs and (1, 1) in pairs


def test_cross_instance_unsat_does_not_hide_duplicate_lanes_in_one_instance(
    monkeypatch,
):
    lane = z3.Int("unsat_cache_duplicate_lane")
    record = _scalar_store(
        4096 + 8 * SymbolicExpr.PID0,
        event_id=0,
        elem_size=4,
        mask=z3.And(lane >= 0, lane < 2),
    )
    core = tc.TwoCopySymbolicHBSolver(
        [record], grid=(2, 1, 1), arange_dict={(0, 2): (lane, None)}
    )
    monkeypatch.setattr(core, "_conflict_precheck", lambda *a, **kw: False)
    answers = _record_checks(monkeypatch, core)
    reports = core.find_races()
    assert answers == [False, True]
    assert [(kind, answer) for kind, _, answer in core.query_stats] == [
        ("cross", False),
        ("intra", True),
    ]
    assert reports and all(
        report.witness_grid_a == report.witness_grid_b for report in reports
    )


def test_unknown_without_fallback_remains_unsupported_and_uncached(monkeypatch):
    core = _repeated_stores(monkeypatch)

    def unknown_query(*args):
        raise UnsupportedSymbolicRaceQuery("forced symbolic unknown")

    monkeypatch.setattr(core, "_race_query_is_sat", unknown_query)
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="forced symbolic unknown"):
        core.find_races()
    assert not core._unsat_race_queries
    assert not core.enum_used


def test_enumerated_unsat_does_not_skip_later_symbolic_attempts(monkeypatch):
    core = _repeated_stores(monkeypatch, enum_fallback_grid=(2, 1, 1))
    symbolic_calls = []
    enumerations = []
    original_enum = core._enumerate_pair

    def unknown_query(*args):
        symbolic_calls.append(True)
        raise UnsupportedSymbolicRaceQuery("forced symbolic unknown")

    def counted_enum(*args, **kwargs):
        answer = original_enum(*args, **kwargs)
        enumerations.append(answer[0])
        return answer

    monkeypatch.setattr(core, "_race_query_is_sat", unknown_query)
    monkeypatch.setattr(core, "_enumerate_pair", counted_enum)
    assert core.find_races() == []
    assert len(symbolic_calls) == 4
    assert enumerations == [False] * 4
    assert core.enum_used
    assert not core._unsat_race_queries


def test_launch_extent_unsat_cannot_hide_an_any_grid_race(monkeypatch):
    # Address ignores pid1: the fallback launch is disjoint, but a second
    # pid1 row aliases every corresponding pid0 store in the symbolic grid.
    grid1 = z3.Int("unsat_cache_any_grid_1")
    core = _repeated_stores(
        monkeypatch, grid=(2, grid1, 1), enum_fallback_grid=(2, 1, 1)
    )
    original = core._race_query_is_sat
    calls = []

    def first_unknown_then_exact(query, first, second):
        calls.append(True)
        if len(calls) == 1:
            raise UnsupportedSymbolicRaceQuery("force launch-only fallback")
        return original(query, first, second)

    monkeypatch.setattr(core, "_race_query_is_sat", first_unknown_then_exact)
    reports = core.find_races()
    assert core.enum_used and len(calls) == 4
    assert reports
    assert any(
        report.witness_grid_a[1] != report.witness_grid_b[1] for report in reports
    )


def test_nonstandard_mutable_guard_is_recoerced_instead_of_memoized(monkeypatch):
    class MutableGuard:
        enabled = False

        def sort(self):
            return z3.IntSort()

        def __ne__(self, other):
            return z3.BoolVal(self.enabled)

    guard = MutableGuard()
    core = _repeated_stores(monkeypatch, stride=0, extra_assumptions=(guard,))
    original = core._race_query_is_sat
    answers = []

    def checked_then_enabled(query, first, second):
        answer = original(query, first, second)
        answers.append(answer)
        guard.enabled = True
        return answer

    monkeypatch.setattr(core, "_race_query_is_sat", checked_then_enabled)
    assert core.find_races()
    assert answers == [False, True, True, True]
    assert not core._unsat_race_queries


def test_feasibility_still_checks_live_launch_premises_after_race_cache_hits(
    monkeypatch,
):
    core = _repeated_stores(monkeypatch, stride=0, grid=(1, 1, 1))
    assert core.find_races() == []
    assert core._unsat_race_queries

    def forbidden_cache_lookup(*args):
        raise AssertionError("feasibility cannot consult the race-query cache")

    monkeypatch.setattr(core, "_known_unsat_race_query", forbidden_cache_lookup)
    assert core.check_feasibility()
    core.launch_premises.append(z3.BoolVal(False))
    assert not core.check_feasibility()
    core.launch_premises.clear()
    assert core.check_feasibility()
    assert not core.check_feasibility((False,))


def test_ast_ids_from_distinct_z3_contexts_never_match(monkeypatch):
    core = _repeated_stores(monkeypatch)
    first_context, second_context = z3.Context(), z3.Context()
    first = z3.BoolVal(False, ctx=first_context)
    second = z3.BoolVal(False, ctx=second_context)
    # Built-in false has the same local AST id in fresh contexts. Both
    # formulas are trivially UNSAT, so no artificial SAT fact is inserted.
    assert first.get_id() == second.get_id()
    core._remember_unsat_race_query(first)
    assert core._known_unsat_race_query(first)
    assert not core._known_unsat_race_query(second)
    core._remember_unsat_race_query(second)
    assert core._known_unsat_race_query(first)
    assert core._known_unsat_race_query(second)
