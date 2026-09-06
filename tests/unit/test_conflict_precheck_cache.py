"""Expression reuse preserves each conflict query's original premises."""

from dataclasses import replace

import pytest
import z3

from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector import two_copy_symbolic_hb_solver as tc
from triton_viz.clients.symbolic_engine import SymbolicExpr

from .test_conflict_simplification import _formula
from .test_two_copy_symbolic_hb_solver import _scalar_store


def _answer(conditions, cache=None):
    relaxed = cs._linear_relaxation(
        conditions, simplify_first=True, expression_cache=cache
    )
    assert relaxed is not None
    oracle = z3.SolverFor("QF_LIA", ctx=relaxed.ctx)
    oracle.add(relaxed)
    return oracle.check()


def _store_solver(extra_assumptions=()):
    g0, g2 = z3.Ints("cache_grid_0 cache_grid_2")
    p2 = SymbolicExpr.PID2
    table = z3.Array("cache_store_table", z3.IntSort(), z3.IntSort())
    address = z3.If(p2 == 0, 4096, z3.Select(table, p2))
    return tc.TwoCopySymbolicHBSolver(
        [_scalar_store(address, event_id=0, elem_size=4, mask=p2 == 0)],
        grid=(g0, 1, g2),
        extra_assumptions=extra_assumptions,
    )


def test_cached_select_keeps_correlation_and_applicability_on_repeated_queries():
    table = z3.Array("cache_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("cache_index")
    loaded = z3.Select(table, index)
    cache = cs._PureSelectExpressionCache()
    low, high = loaded >= 7, loaded < 7
    assert _answer([low], cache) == z3.sat
    assert _answer([high], cache) == z3.sat
    # Both conditions now hit the memo, including their Select metadata.
    assert _answer([low, high], cache) == z3.unsat
    assert _answer([low], cache) == z3.sat


@pytest.mark.parametrize("different_array", [False, True])
def test_cached_copy_reads_remain_independent(different_array):
    table = z3.Array("cache_copy_table", z3.IntSort(), z3.IntSort())
    other = z3.Array("cache_other_table", z3.IntSort(), z3.IntSort())
    ia, ib = z3.Ints("cache_index_a cache_index_b")
    a = z3.Select(table, ia)
    b = z3.Select(other if different_array else table, ib)
    cache = cs._PureSelectExpressionCache()
    conditions = [ia == 0, ib == 1, a == 7, b == 11]
    assert _answer(conditions, cache) == _answer(conditions) == z3.sat
    # Intra substitutions precede memo lookup; same-index reads correlate.
    if not different_array:
        same_index = [z3.substitute(c, (ib, ia)) for c in [a == 7, b == 11]]
        assert _answer(same_index, cache) == z3.unsat
    assert _answer(conditions, cache) == z3.sat


@pytest.mark.parametrize("context", ["not", "antecedent", "ite"])
def test_cached_unsupported_boolean_keeps_negative_polarity(context):
    table = z3.Array("cache_bool_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("cache_bool_index")
    predicate = z3.Function("cache_predicate", z3.IntSort(), z3.BoolSort())(index)
    condition = {
        "not": z3.Not(predicate),
        "antecedent": z3.Implies(predicate, z3.BoolVal(False)),
        "ite": z3.If(predicate, z3.BoolVal(False), z3.BoolVal(True)),
    }[context]
    cache = cs._PureSelectExpressionCache()
    conditions = [z3.Select(table, index) == 1, condition]
    assert _answer(conditions, cache) == _answer(conditions) == z3.sat
    assert _answer([*conditions, predicate], cache) == z3.unsat
    assert _answer(conditions, cache) == z3.sat


def test_cached_array_guard_retains_in_and_out_of_domain_results():
    table = z3.Array("cache_guard_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("cache_guard_index")
    address = z3.If(index == 0, 2, z3.If(index == 1, 3, z3.Select(table, index)))
    cache = cs._PureSelectExpressionCache()
    for conditions, expected in [
        ([index >= 0, index < 2, address >= 4], z3.unsat),
        ([index == 2, address == 4], z3.sat),
        ([index >= 0, index < 2, address >= 4], z3.unsat),
    ]:
        assert _answer(conditions, cache) == _answer(conditions) == expected


def test_revoked_and_replaced_extra_pins_do_not_survive_common_cache():
    solver = _store_solver()
    a, b = solver.events
    assert not solver._conflict_precheck(a, b)
    expression_cache = solver._conflict_expression_cache
    solver.extra_assumptions = (solver.grid[0] == 1,)
    assert solver._conflict_precheck(a, b)
    solver.extra_assumptions = ()
    assert not solver._conflict_precheck(a, b)
    assert solver._conflict_expression_cache is expression_cache


def test_constructor_copies_list_and_direct_mutable_extra_list_is_not_cached():
    original = []
    solver = _store_solver(original)
    a, b = solver.events
    original.append(solver.grid[0] == 1)
    # The existing constructor owns an immutable snapshot of its input.
    assert solver.extra_assumptions == ()
    assert not solver._conflict_precheck(a, b)
    solver.extra_assumptions = original
    assert solver._conflict_precheck(a, b)
    original.clear()
    assert not solver._conflict_precheck(a, b)
    original.append(solver.grid[0] == 1)
    assert solver._conflict_precheck(a, b)


@pytest.mark.parametrize("source", ["grid", "ranges", "mutable_ranges"])
def test_grid_and_range_changes_invalidate_common_constraints(source):
    solver = _store_solver()
    a, b = solver.events
    assert not solver._conflict_precheck(a, b)
    old_grid = solver.grid_constraints
    if source == "grid":
        solver.grid_constraints = z3.And(old_grid, solver.grid[0] == 1)
    else:
        solver.arange_constraints_a = (solver.ctx_a.pid[0] == 0,)
        solver.arange_constraints_b = (solver.ctx_b.pid[0] == 0,)
        if source == "mutable_ranges":
            solver.arange_constraints_a = list(solver.arange_constraints_a)
            solver.arange_constraints_b = list(solver.arange_constraints_b)
    assert solver._conflict_precheck(a, b)
    if source == "grid":
        solver.grid_constraints = old_grid
    elif source == "ranges":
        solver.arange_constraints_a = solver.arange_constraints_b = ()
    else:
        solver.arange_constraints_a.clear()
        solver.arange_constraints_b.clear()
    assert not solver._conflict_precheck(a, b)


def test_context_replacement_rebuilds_shared_substitution_and_intra_expression(
    monkeypatch,
):
    queries = []

    def capture(conditions, *args, **kwargs):
        queries.append(conditions)
        return False

    monkeypatch.setattr(tc, "conflict_impossible", capture)
    solver = _store_solver()
    a, b = solver.events
    old_a, old_b = solver.ctx_a.pid[0], solver.ctx_b.pid[0]
    replacement = z3.Int("cache_replacement_pid")
    solver.extra_assumptions = (old_a == 0, old_b == 1)
    solver._conflict_precheck(a, b, same_instance=True, lane_cond=z3.BoolVal(True))
    first = solver._conflict_precheck_common()
    assert first.intra_expression is not None
    solver.ctx_a = replace(solver.ctx_a, pid=(replacement, *solver.ctx_a.pid[1:]))
    assert not solver._conflict_precheck(
        a, b, same_instance=True, lane_cond=z3.BoolVal(True)
    )
    second = solver._conflict_precheck_common()
    assert second is not first
    expected = tc.apply_sub(second.expression, second.shared)
    assert second.intra_expression.eq(expected)
    assert (old_b, replacement) in second.shared
    for conditions, expected_result in zip(queries, (z3.unsat, z3.sat)):
        oracle = z3.Solver()
        oracle.add(*conditions)
        assert oracle.check() == expected_result


def test_common_and_its_normalization_are_reused_as_single_nodes(monkeypatch):
    solver = _store_solver()
    table = z3.Array("cache_large_snapshot", z3.IntSort(), z3.IntSort())
    solver.extra_assumptions = tuple(z3.Select(table, i) == 3 * i for i in range(128))
    conversions = []
    original_as_bool = tc.as_bool

    def counted_as_bool(condition):
        conversions.append(condition)
        return original_as_bool(condition)

    monkeypatch.setattr(tc, "as_bool", counted_as_bool)
    common = solver._conflict_precheck_common()
    assert len(conversions) == 128
    assert solver._conflict_precheck_common() is common
    assert len(conversions) == 128
    cache = cs._PureSelectExpressionCache()
    first = cache.normalize(common.expression)
    abstracted, saw_select = cache.abstract(first)
    assert saw_select
    original_simplify = cs.z3.simplify

    def reject_common_simplification(expression):
        assert not expression.eq(common.expression)
        return original_simplify(expression)

    monkeypatch.setattr(cs.z3, "simplify", reject_common_simplification)
    assert cache.normalize(common.expression) is first
    repeated, repeated_select = cache.abstract(first)
    assert repeated is abstracted and repeated_select
    solver.extra_assumptions = solver.extra_assumptions[:-1]
    assert solver._conflict_precheck_common() is not common
    assert len(conversions) == 255


def test_separate_solvers_own_separate_expression_caches():
    first, second = _store_solver(), _store_solver()
    first._conflict_precheck(*first.events)
    second._conflict_precheck(*second.events)
    assert first._conflict_expression_cache is not second._conflict_expression_cache
    assert first._conflict_precheck_common() is not second._conflict_precheck_common()


def test_radix_path_never_reuses_pair_independent_select_cache():
    class RejectCache:
        def relaxation(self, conditions):
            raise AssertionError("radix lemmas must remain query-local")

    conditions, correspondence, _ = _formula()
    assert cs.conflict_impossible(
        conditions, correspondence, expression_cache=RejectCache()
    )


def test_primed_cache_does_not_override_disabled_snapshot_factor(monkeypatch):
    table = z3.Array("cache_disabled_table", z3.IntSort(), z3.IntSort())
    loaded = z3.Select(table, z3.Int("cache_disabled_index"))
    cache = cs._PureSelectExpressionCache()
    assert _answer([loaded == 7], cache) == z3.sat
    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
    assert cache.relaxation([loaded == 7]) is None


def test_expression_cache_keeps_custom_z3_context():
    context = z3.Context()
    integer = z3.IntSort(ctx=context)
    table = z3.Array("cache_context_table", integer, integer)
    loaded = z3.Select(table, z3.Int("cache_context_index", ctx=context))
    cache = cs._PureSelectExpressionCache()
    assert _answer([loaded >= 7, loaded < 7], cache) == z3.unsat
    assert _answer([loaded == 7], cache) == z3.sat
