"""Exact free-array elimination must preserve scope and original witnesses."""

import pytest
import z3

from evaluation.solver_prototypes import single_read_arrays
from evaluation.solver_prototypes.single_read_arrays import (
    single_read_array_normal_form,
    single_read_array_solver,
)


def _array(name="single_array"):
    return z3.Array(name, z3.IntSort(), z3.IntSort())


def _check_lift(original, expected=z3.sat):
    rewritten = single_read_array_normal_form(original)
    assert rewritten is not None
    # This implication is stronger than comparing the two SAT answers:
    # every native witness for the rewritten query must be an old witness.
    implication = z3.Solver()
    implication.add(rewritten, z3.Not(original))
    assert implication.check() == z3.unsat
    for formula in (original, rewritten):
        solver = z3.Solver()
        solver.add(formula)
        assert solver.check() == expected
        if expected == z3.sat:
            assert z3.is_true(
                z3.simplify(solver.model().eval(original, model_completion=True))
            )
    return rewritten


def test_repeated_read_retains_its_identity_and_native_array_model():
    array = _array()
    index, value = z3.Ints("single_index single_value")
    read = z3.Select(array, index)
    _check_lift(z3.And(index > 1000000, read == value, read + 1 == value + 1))
    _check_lift(z3.And(read == value, read + 1 == value), z3.unsat)


def test_distinct_arrays_at_equal_indices_remain_independent():
    a, b, index = _array("single_a"), _array("single_b"), z3.Int("single_index")
    _check_lift(z3.And(z3.Select(a, index) == -7, z3.Select(b, index) == 11))


def test_nested_reads_of_different_arrays_have_a_native_model():
    a, b, index = _array("nested_a"), _array("nested_b"), z3.Int("nested_index")
    inner = z3.Select(b, index)
    outer = z3.Select(a, inner)
    _check_lift(z3.And(index < 0, inner == 7, outer == 13))
    # The inner scalar disappears under simultaneous substitution of the
    # outer Select. Its constant-array binding still makes lifting valid.
    _check_lift(outer == 13)


def test_one_fixed_cell_constraint_is_retained():
    array = _array()
    read = z3.Select(array, 3)
    _check_lift(z3.And(read == 9, read > 8))
    _check_lift(z3.And(read == 9, read < 8), z3.unsat)


@pytest.mark.parametrize("active", [False, True])
def test_masks_and_inactive_fallback_values_are_retained(active):
    array = _array()
    index, flag = z3.Int("mask_index"), z3.Bool("mask_active")
    value = z3.If(flag, z3.Select(array, index), 17)
    _check_lift(z3.And(flag == active, index == -19, value == 17))
    _check_lift(
        z3.And(flag == active, index == -19, value != 17),
        z3.sat if active else z3.unsat,
    )


@pytest.mark.parametrize("bound", [-10, 0, 4, 1000000])
def test_symbolic_grid_and_negative_values_do_not_get_pinned(bound):
    array = _array()
    pid, grid = z3.Ints("original_pid original_grid")
    original = z3.And(
        grid == bound, pid == bound - 1, z3.Select(array, pid) == -bound - 23
    )
    _check_lift(original)


def test_projected_sat_equivalence_over_scalar_assignments():
    array = _array()
    x, y = z3.Ints("projection_x projection_y")
    read = z3.Select(array, x + y)
    original = z3.And(x <= read, read < y, read % 2 == 0)
    rewritten = single_read_array_normal_form(original)
    assert rewritten is not None
    for xv in range(-3, 4):
        for yv in range(-3, 4):
            old, new = z3.Solver(), z3.Solver()
            old.add(original, x == xv, y == yv)
            new.add(rewritten, x == xv, y == yv)
            assert old.check() == new.check()


@pytest.mark.parametrize("index_relation", ["none", "equal", "different"])
def test_multiple_reads_of_same_array_are_refused(index_relation):
    array = _array()
    i, j = z3.Ints("multi_i multi_j")
    conditions = [z3.Select(array, i) == 4, z3.Select(array, j) == 5]
    if index_relation == "equal":
        conditions.append(i == j)
    elif index_relation == "different":
        conditions.append(i != j)
    assert single_read_array_normal_form(z3.And(*conditions)) is None


def test_nested_reads_of_same_array_are_refused():
    array = _array()
    assert (
        single_read_array_normal_form(z3.Select(array, z3.Select(array, 0)) == 3)
        is None
    )


def test_snapshot_equalities_elsewhere_in_complete_query_prevent_elimination():
    array, i = _array(), z3.Int("snapshot_index")
    original = z3.And(z3.Select(array, 0) == 5, z3.Select(array, i) == 9, i == 0)
    assert single_read_array_normal_form(original) is None
    solver = z3.Solver()
    solver.add(original)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("use", ["equal", "different", "store", "constant", "uf"])
def test_non_select_array_uses_are_refused(use):
    a, b, index = _array("other_a"), _array("other_b"), z3.Int("other_index")
    extra = {
        "equal": a == b,
        "different": a != b,
        "store": z3.Select(z3.Store(a, 0, 4), index) == 4,
        "constant": a == z3.K(z3.IntSort(), 7),
        "uf": z3.Function("array_predicate", a.sort(), z3.BoolSort())(a),
    }[use]
    assert (
        single_read_array_normal_form(z3.And(z3.Select(a, index) == 3, extra)) is None
    )


@pytest.mark.parametrize("kind", ["forall", "exists", "lambda"])
def test_quantified_uses_are_refused(kind):
    array, index = _array(), z3.Int("bound_index")
    formula = {
        "forall": z3.ForAll(index, z3.Select(array, index) == index),
        "exists": z3.Exists(index, z3.Select(array, index) == index),
        "lambda": z3.Select(z3.Lambda(index, index + 1), 2) == 3,
    }[kind]
    assert single_read_array_normal_form(formula) is None


@pytest.mark.parametrize("sort", [z3.BoolSort(), z3.RealSort(), z3.BitVecSort(8)])
def test_other_array_sorts_are_refused(sort):
    array = z3.Array("other_sort", z3.IntSort(), sort)
    value = z3.Const("other_value", sort)
    assert single_read_array_normal_form(z3.Select(array, 0) == value) is None


def test_formula_without_arrays_and_non_boolean_input_are_unchanged():
    assert single_read_array_normal_form(z3.Int("plain") > 0) is None
    assert single_read_array_normal_form(z3.Int("plain")) is None


def test_non_default_context_is_supported():
    ctx = z3.Context()
    integer = z3.IntSort(ctx)
    array = z3.Array("context_array", integer, integer)
    original = z3.Select(array, z3.IntVal(2, ctx)) == z3.IntVal(7, ctx)
    rewritten = single_read_array_normal_form(original)
    assert rewritten is not None and rewritten.ctx == ctx
    solver = z3.Solver(ctx=ctx)
    solver.add(rewritten, z3.Not(original))
    assert solver.check() == z3.unsat
    exact = single_read_array_solver(original)
    assert exact is not None
    assert exact.check() == z3.sat
    assert z3.is_true(exact.model().eval(original, model_completion=True))


def test_solver_factory_retains_original_model_and_scalar_enumeration_pins():
    array, pid, grid = _array(), z3.Int("factory_pid"), z3.Int("factory_grid")
    original = z3.And(pid >= 0, pid < grid, z3.Select(array, pid) == pid + 7)
    solver = single_read_array_solver(original)
    assert solver is not None
    # The production fallback appends existing scalar PID pins to a newly
    # built query. They must retain the same meaning with model conversion.
    solver.set(timeout=1000)
    solver.add(grid == 100, pid == 42)
    assert solver.check() == z3.sat
    model = solver.model()
    assert z3.is_true(model.eval(original, model_completion=True))
    assert model.eval(z3.Select(array, pid)).as_long() == 49
    solver.push()
    solver.add(pid == 43)
    assert solver.check() == z3.unsat
    solver.pop()
    assert solver.check() == z3.sat
    assert z3.is_true(solver.model().eval(original, model_completion=True))


def test_solver_factory_refuses_nonmatching_formulas():
    array = _array()
    assert single_read_array_solver(z3.Int("no_array") > 0) is None
    assert single_read_array_solver(z3.Select(array, 0) != z3.Select(array, 1)) is None


def test_solver_factory_preserves_resource_limit_unknown():
    array, index = _array(), z3.Int("limited_index")
    solver = single_read_array_solver(z3.Select(array, index) == index + 3)
    assert solver is not None
    solver.set(timeout=1000, rlimit=1)
    assert solver.check() == z3.unknown
    assert solver.reason_unknown()


@pytest.mark.parametrize("failing_step", ["rewrite", "tactic"])
def test_solver_factory_preparation_exception_returns_original_path(
    monkeypatch, failing_step
):
    def fail(*args, **kwargs):
        raise z3.Z3Exception("injected setup failure")

    original = z3.Select(_array(), 0) == 7
    if failing_step == "rewrite":
        monkeypatch.setattr(single_read_arrays, "single_read_array_normal_form", fail)
    else:
        monkeypatch.setattr(z3, "Tactic", fail)
    assert single_read_array_solver(original) is None
    solver = z3.Solver()
    solver.add(original)
    assert solver.check() == z3.sat


@pytest.mark.parametrize("kind", ["nonlinear_sat", "nonlinear_unsat", "uninterpreted"])
def test_alternate_arithmetic_backend_retains_general_smt_and_native_models(kind):
    array = _array()
    x, y = z3.Ints("general_x general_y")
    read = z3.Select(array, x)
    if kind == "nonlinear_sat":
        original = z3.And(x * y == 6, x > 1, y > 1, read == x + y)
        expected = z3.sat
    elif kind == "nonlinear_unsat":
        original = z3.And(x * x == 2, read == x)
        expected = z3.unsat
    else:
        function = z3.Function("general_function", z3.IntSort(), z3.IntSort())
        original = z3.And(function(read) == read + 1, read > x)
        expected = z3.sat
    exact = single_read_array_solver(original)
    assert exact is not None
    exact.set(timeout=1000)
    assert exact.check() == expected
    if expected == z3.sat:
        assert z3.is_true(
            z3.simplify(exact.model().eval(original, model_completion=True))
        )
