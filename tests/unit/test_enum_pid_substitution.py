"""Preserve each complete finite query and its witness under PID substitution."""

import time

import pytest
import z3

from triton_viz.clients.race_detector import two_copy_symbolic_hb_solver as tc
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery


def _core(grid=(2, 1, 1)):
    core = object.__new__(tc.TwoCopySymbolicHBSolver)
    core.enum_fallback_grid = grid
    core.enum_used = False
    core._enum_deadline = None
    return core


def _build(*conditions):
    def build():
        solver = z3.Solver()
        solver.add(*conditions)
        return solver

    return build


def _error():
    return UnsupportedSymbolicRaceQuery("original symbolic query undecided")


def test_sat_model_restores_all_pid_coordinates_and_other_witness_values():
    pa = [z3.Int(f"pid_a_{i}") for i in range(3)]
    pb = [z3.Int(f"pid_b_{i}") for i in range(3)]
    first, second = (1, 1, 1), (0, 1, 0)
    lane_a, lane_b = z3.Ints("enum_lane_a enum_lane_b")
    observations = z3.Array("enum_snapshot", z3.IntSort(), z3.IntSort())
    ordered = z3.Bool("enum_hb")
    conditions = [
        *(
            pid == value
            for pids, values in ((pa, first), (pb, second))
            for pid, value in zip(pids, values)
        ),
        lane_a != lane_b,
        lane_a == 2,
        lane_b == 3,
        observations[pa[0] + 2 * pa[1] + 4 * pa[2]] == lane_b,
        z3.Not(ordered),
    ]
    core = _core((2, 2, 2))
    result, model = core._enumerate_pair(_build(*conditions), False, _error())
    assert result and core.enum_used
    assert tuple(model.eval(pid).as_long() for pid in pa) == first
    assert tuple(model.eval(pid).as_long() for pid in pb) == second
    assert z3.is_true(model.eval(z3.And(*conditions), model_completion=True))
    assert not getattr(core, "_unsat_race_queries", {})


@pytest.mark.parametrize("same_instance", [False, True])
def test_checks_every_case_without_populating_symbolic_unsat_cache(
    monkeypatch, same_instance
):
    core = _core((3, 1, 1))
    count = len(core._enum_pid_cases(same_instance))
    builds, checks = [], []
    original_check = z3.Solver.check

    def build():
        builds.append(True)
        return _build(z3.BoolVal(False))()

    def check(self, *args):
        checks.append(True)
        return original_check(self, *args)

    monkeypatch.setattr(z3.Solver, "check", check)
    assert core._enumerate_pair(build, same_instance, _error()) == (False, None)
    assert len(builds) == count and len(checks) == count
    assert not getattr(core, "_unsat_race_queries", {})
    assert core._enumerate_pair(build, same_instance, _error()) == (False, None)
    assert len(builds) == count * 2 and len(checks) == count * 2


def test_changed_assertions_between_cases_cannot_reuse_an_old_unsat():
    core = _core((3, 1, 1))
    cases = core._enum_pid_cases(False)
    calls = 0

    def build():
        nonlocal calls
        calls += 1
        # Every earlier case is infeasible, but the final one is a race.
        return _build(z3.BoolVal(calls == len(cases)))()

    result, model = core._enumerate_pair(build, False, _error())
    assert result and calls == len(cases)
    assert model.eval(z3.Int("pid_a_0")).as_long() == cases[-1][0][0]
    assert model.eval(z3.Int("pid_b_0")).as_long() == cases[-1][1][0]


@pytest.mark.parametrize("same_instance", [False, True])
def test_infeasible_hb_array_or_lane_premises_are_not_dropped(same_instance):
    core = _core((2, 1, 1))
    pa, pb = z3.Ints("pid_a_0 pid_b_0")
    source = z3.Array("enum_infeasible_source", z3.IntSort(), z3.IntSort())
    hb = z3.Bool("enum_infeasible_hb")
    lane = z3.Int("enum_infeasible_lane")
    for contradiction in (
        (hb, z3.Not(hb)),
        (source[pa] == 7, source[pa] == 8),
        (lane >= 0, lane < 0),
        (pa == pb, pa != pb),
    ):
        assert core._enumerate_pair(
            _build(*contradiction), same_instance, _error()
        ) == (False, None)


@pytest.mark.parametrize("offset", [-2, -1, 0, 1])
def test_signed_division_zero_divisor_and_partial_overlap_keep_original_models(offset):
    core = _core((3, 1, 1))
    pa, pb = z3.Ints("pid_a_0 pid_b_0")
    value = z3.Int("enum_dividend")
    # Negative divisors and the original underspecified div/mod-by-zero
    # applications must be substituted, not approximated or omitted.
    a = value / (pa + offset) + value % (pa + offset)
    b = value / (pb + offset) + value % (pb + offset)
    conditions = (value == -7, pa != pb, a < b + 4, b < a + 4)
    expected = z3.Solver()
    expected.add(*conditions, pa >= 0, pa < 3, pb >= 0, pb < 3)
    result, model = core._enumerate_pair(_build(*conditions), False, _error())
    assert result == (expected.check() == z3.sat)
    if result:
        assert z3.is_true(model.eval(z3.And(*conditions), model_completion=True))


def test_unknown_case_reraises_original_error_without_caching(monkeypatch):
    core = _core()
    original = _error()
    monkeypatch.setattr(z3.Solver, "check", lambda *args: z3.unknown)
    with pytest.raises(UnsupportedSymbolicRaceQuery) as exc:
        core._enumerate_pair(_build(z3.BoolVal(False)), False, original)
    assert exc.value is original and core.enum_used
    assert not getattr(core, "_unsat_race_queries", {})


def test_expired_total_budget_still_refuses_before_building():
    core = _core()
    core._enum_deadline = time.monotonic() - 1
    original = _error()

    def forbidden():
        raise AssertionError("expired case was built")

    with pytest.raises(UnsupportedSymbolicRaceQuery) as exc:
        core._enumerate_pair(forbidden, False, original)
    assert exc.value is original
