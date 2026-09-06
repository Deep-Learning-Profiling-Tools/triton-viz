"""Base assertion reuse retains mutable premises and HB acyclicity."""

import pytest
import z3

from triton_viz.clients.race_detector import two_copy_symbolic_hb_solver as tc

from .test_two_copy_symbolic_hb_solver import _scalar_store


_FAMILIES = (
    "arange_constraints_a",
    "arange_constraints_b",
    "rf_constraints",
    "atomic_coherence_constraints",
    "counting_constraints",
    "value_causality_constraints",
    "extra_assumptions",
)


def _solver():
    return tc.TwoCopySymbolicHBSolver(
        [_scalar_store(z3.IntVal(4096), event_id=0)], grid=(1, 1, 1)
    )


def _original_base(solver):
    """The previous per-assertion construction is the equivalence oracle."""
    result = z3.Solver()
    result.add(solver.grid_constraints)
    for family in _FAMILIES:
        for condition in getattr(solver, family):
            result.add(
                tc.as_bool(condition) if family == "extra_assumptions" else condition
            )
    for i in range(len(solver.events)):
        result.add(z3.Not(solver.hb[i][i]))
    return result


def _assert_equivalent(solver):
    actual = solver._base_solver()
    expected = _original_base(solver)
    oracle = z3.Solver()
    oracle.add(z3.Xor(z3.And(*actual.assertions()), z3.And(*expected.assertions())))
    assert oracle.check() == z3.unsat
    return actual


def test_every_constraint_family_is_retained_in_one_equivalent_assertion():
    solver = _solver()
    for i, family in enumerate(_FAMILIES):
        setattr(solver, family, [z3.Int(f"base_family_{i}") > i])
    solver.hb[0][0] = z3.Bool("base_conditional_cycle")
    query = _assert_equivalent(solver)
    assert len(query.assertions()) == 1
    assert query.check() == z3.sat


@pytest.mark.parametrize("family", _FAMILIES)
def test_mutable_family_append_replace_clear_invalidates_cached_premises(family):
    solver = _solver()
    conditions = []
    setattr(solver, family, conditions)
    assert solver._base_solver().check() == z3.sat
    original_cache = solver._base_constraint_cache

    conditions.append(z3.BoolVal(False))
    assert _assert_equivalent(solver).check() == z3.unsat
    assert solver._base_constraint_cache is not original_cache
    false_cache = solver._base_constraint_cache

    # Same container and length, different actual element.
    conditions[0] = z3.BoolVal(True)
    assert _assert_equivalent(solver).check() == z3.sat
    assert solver._base_constraint_cache is not false_cache

    conditions.clear()
    assert _assert_equivalent(solver).check() == z3.sat
    setattr(solver, family, (z3.BoolVal(False),))
    assert _assert_equivalent(solver).check() == z3.unsat


def test_grid_replacement_and_container_replacement_rebuild_cache():
    solver = _solver()
    assert solver._base_solver().check() == z3.sat
    solver.grid_constraints = z3.BoolVal(False)
    assert _assert_equivalent(solver).check() == z3.unsat
    cached = solver._base_constraint_cache
    solver.rf_constraints = list(solver.rf_constraints)
    solver._base_solver()
    assert solver._base_constraint_cache is not cached


def test_diagonal_entry_row_and_matrix_replacements_retain_cycles():
    solver = _solver()
    assert solver._base_solver().check() == z3.sat
    solver.hb[0][0] = z3.BoolVal(True)
    assert _assert_equivalent(solver).check() == z3.unsat

    cycle = z3.Bool("base_mutable_cycle")
    solver.hb[0] = [cycle, solver.hb[0][1]]
    query = _assert_equivalent(solver)
    query.add(cycle)
    assert query.check() == z3.unsat
    query = solver._base_solver()
    query.add(z3.Not(cycle))
    assert query.check() == z3.sat

    cached = solver._base_constraint_cache
    solver.hb = [list(row) for row in solver.hb]
    solver._base_solver()
    assert solver._base_constraint_cache is not cached
    solver.hb[0][0] = z3.BoolVal(False)
    assert _assert_equivalent(solver).check() == z3.sat


def test_event_count_change_updates_required_diagonal_constraints():
    solver = _solver()
    solver._base_solver()
    solver.events.append(solver.events[-1])
    for row in solver.hb:
        row.append(z3.BoolVal(False))
    solver.hb.append([z3.BoolVal(False), z3.BoolVal(False), z3.BoolVal(True)])
    assert _assert_equivalent(solver).check() == z3.unsat
    solver.events.pop()
    assert _assert_equivalent(solver).check() == z3.sat


def test_unchanged_sources_reuse_expression_without_recoercion_or_negations(
    monkeypatch,
):
    solver = _solver()
    solver.extra_assumptions = (1, z3.IntVal(2), z3.Int("base_nonzero_guard"))
    first = solver._base_solver()
    expression = solver._base_constraint_cache.expression

    def unexpected_construction(*args, **kwargs):
        raise AssertionError("a cache hit must not rebuild Z3 expressions")

    monkeypatch.setattr(tc, "as_bool", unexpected_construction)
    monkeypatch.setattr(tc, "Not", unexpected_construction)
    monkeypatch.setattr(tc, "And", unexpected_construction)
    second = solver._base_solver()
    assert solver._base_constraint_cache.expression is expression
    assert first is not second
    first.add(z3.BoolVal(False))
    assert first.check() == z3.unsat
    assert second.check() == z3.sat


def test_false_diagonals_do_not_construct_tautological_negations(monkeypatch):
    solver = _solver()
    calls = []
    original_not = tc.Not

    def counted_not(condition):
        calls.append(condition)
        return original_not(condition)

    monkeypatch.setattr(tc, "Not", counted_not)
    solver._base_solver()
    assert not calls
    cycle = z3.Bool("base_counted_cycle")
    solver.hb[1][1] = cycle
    solver._base_solver()
    assert len(calls) == 1 and calls[0] is cycle


@pytest.mark.parametrize(
    "guard,expected",
    [(False, z3.unsat), (0, z3.unsat), (0.0, z3.unsat), (3, z3.sat)],
)
def test_immutable_extra_guards_keep_original_as_bool_semantics(guard, expected):
    solver = _solver()
    solver.extra_assumptions = (guard,)
    assert _assert_equivalent(solver).check() == expected


def test_mutable_custom_guard_uses_original_coercion_on_each_call():
    class MutableGuard:
        enabled = True

        def sort(self):
            return z3.IntSort()

        def __ne__(self, other):
            return z3.BoolVal(self.enabled)

    solver = _solver()
    guard = MutableGuard()
    solver.extra_assumptions = (guard,)
    assert _assert_equivalent(solver).check() == z3.sat
    assert solver._base_constraint_cache is None
    guard.enabled = False
    assert _assert_equivalent(solver).check() == z3.unsat


@pytest.mark.parametrize("family", ["grid_constraints", "rf_constraints"])
def test_nested_mutable_constraint_containers_use_original_add_path(family):
    solver = _solver()
    nested = [z3.BoolVal(True)]
    setattr(solver, family, nested if family == "grid_constraints" else [nested])
    assert _assert_equivalent(solver).check() == z3.sat
    assert solver._base_constraint_cache is None
    nested[0] = z3.BoolVal(False)
    assert _assert_equivalent(solver).check() == z3.unsat


def test_feasibility_launch_premises_and_query_extras_are_never_cached():
    solver = _solver()
    assert solver.check_feasibility()
    cached = solver._base_constraint_cache
    solver.launch_premises.append(z3.BoolVal(False))
    assert not solver.check_feasibility()
    solver.launch_premises.clear()
    assert solver.check_feasibility()
    assert not solver.check_feasibility((False,))
    assert solver.check_feasibility()
    assert solver._base_constraint_cache is cached
    # The single-instance launch is feasible, although cross-copy is not.
    assert solver._new_solver().check() == z3.unsat


def test_late_coherence_hb_constraints_are_included_before_first_query(monkeypatch):
    original = tc.TwoCopySymbolicHBSolver._build_coherence_hb_constraints

    def append_after_hb(self):
        original(self)
        assert self.hb
        self.atomic_coherence_constraints.append(z3.BoolVal(False))

    monkeypatch.setattr(
        tc.TwoCopySymbolicHBSolver,
        "_build_coherence_hb_constraints",
        append_after_hb,
    )
    solver = _solver()
    assert _assert_equivalent(solver).check() == z3.unsat
    assert not solver.check_feasibility()


def test_nonboolean_ordinary_constraints_are_not_silently_coerced():
    solver = _solver()
    solver.rf_constraints.append(1)
    with pytest.raises(z3.Z3Exception):
        _original_base(solver)
    with pytest.raises(z3.Z3Exception):
        solver._base_solver()


def test_one_shot_nonstandard_family_keeps_original_iteration_behavior():
    solver = _solver()
    solver.rf_constraints = iter([[z3.BoolVal(False)]])
    assert solver._base_solver().check() == z3.unsat
    assert solver._base_constraint_cache is None
    assert solver._base_solver().check() == z3.sat


def test_nested_extra_list_retains_original_coercion_error():
    solver = _solver()
    solver.extra_assumptions = [[z3.BoolVal(True)]]
    # Existing as_bool treats list.sort as a sort accessor; caching must not
    # accept a value that the original coercion path rejected.
    with pytest.raises(AttributeError):
        _original_base(solver)
    with pytest.raises(AttributeError):
        solver._base_solver()
