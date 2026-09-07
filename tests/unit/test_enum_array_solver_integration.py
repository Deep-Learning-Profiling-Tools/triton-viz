"""The exact array solver keeps its bindings across finite PID instantiation."""

import pytest
import z3

from triton_viz.clients.race_detector.guarded_division import (
    guarded_division_normal_form,
)
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from evaluation.solver_prototypes.single_read_arrays import single_read_array_solver
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)


def _core(grid=(3, 1, 1)):
    core = object.__new__(TwoCopySymbolicHBSolver)
    core.enum_fallback_grid = grid
    core.enum_used = False
    core._enum_deadline = None
    return core


def test_tactic_reset_restores_arrays_and_all_pid_coordinates_for_late_sat():
    pa = [z3.Int(f"pid_a_{i}") for i in range(3)]
    pb = [z3.Int(f"pid_b_{i}") for i in range(3)]
    expected_a, expected_b = (1, 1, 1), (0, 1, 0)
    array = z3.Array("enum_integrated_array", z3.IntSort(), z3.IntSort())
    original = z3.And(
        *(
            pid == value
            for pids, values in ((pa, expected_a), (pb, expected_b))
            for pid, value in zip(pids, values)
        ),
        z3.Select(array, pa[0] + 4 * pb[1]) == 17 + pa[2],
    )
    builds = []

    def build():
        solver = single_read_array_solver(original)
        assert solver is not None
        builds.append(solver)
        return solver

    core = _core((2, 2, 2))
    result, model = core._enumerate_pair(
        build, False, UnsupportedSymbolicRaceQuery("original unknown")
    )
    assert result and len(builds) > 1
    assert tuple(model.eval(pid).as_long() for pid in pa) == expected_a
    assert tuple(model.eval(pid).as_long() for pid in pb) == expected_b
    assert z3.is_true(model.eval(original, model_completion=True))
    assert model.eval(z3.Select(array, pa[0] + 4 * pb[1])).as_long() == 18
    assert core.enum_used and not getattr(core, "_unsat_race_queries", {})


def test_array_read_identity_survives_every_cross_instance_case():
    pa, pb = z3.Ints("pid_a_0 pid_b_0")
    array = z3.Array("enum_integrated_identity", z3.IntSort(), z3.IntSort())
    read = z3.Select(array, pa + pb)
    original = z3.And(read == pa, read == pb, pa != pb)
    builds = []

    def build():
        solver = single_read_array_solver(original)
        assert solver is not None
        builds.append(solver)
        return solver

    core = _core()
    assert core._enumerate_pair(
        build, False, UnsupportedSymbolicRaceQuery("original unknown")
    ) == (False, None)
    assert len(builds) == 6
    assert core.enum_used and not getattr(core, "_unsat_race_queries", {})


def test_guarded_divisor_equalities_and_array_bindings_share_a_native_model():
    pa, pb = z3.Ints("pid_a_0 pid_b_0")
    divisor = z3.If(pa < 3, 2, 7)
    array = z3.Array("enum_integrated_guarded", z3.IntSort(), z3.IntSort())
    original = z3.And(
        pa >= 0,
        pa < 3,
        pb >= 0,
        pb < 3,
        pa != pb,
        6 / divisor == 3,
        z3.Select(array, pa + pb) == 6 / divisor + pa,
    )
    normalized = guarded_division_normal_form(original)

    def build():
        solver = single_read_array_solver(normalized)
        assert solver is not None
        return solver

    result, model = _core()._enumerate_pair(
        build, False, UnsupportedSymbolicRaceQuery("original unknown")
    )
    assert result
    assert z3.is_true(model.eval(normalized, model_completion=True))
    assert z3.is_true(model.eval(original, model_completion=True))


def test_tactic_resource_limit_remains_effective_after_pid_substitution_reset():
    pa = z3.Int("pid_a_0")
    array = z3.Array("enum_integrated_limit", z3.IntSort(), z3.IntSort())
    original = z3.Select(array, pa) == pa + 1

    def build():
        solver = single_read_array_solver(original)
        assert solver is not None
        solver.set(rlimit=1)
        return solver

    error = UnsupportedSymbolicRaceQuery("original symbolic unknown")
    core = _core()
    with pytest.raises(UnsupportedSymbolicRaceQuery) as exc:
        core._enumerate_pair(build, False, error)
    assert exc.value is error
    assert core.enum_used and not getattr(core, "_unsat_race_queries", {})
