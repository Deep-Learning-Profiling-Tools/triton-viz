"""Sound UNSAT-only conflict filtering, including mixed-radix counterexamples."""

import pytest
import z3

from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.clients.symbolic_engine import SymbolicExpr

from .test_two_copy_symbolic_hb_solver import _rmw_record, _scalar_store


def _formula(
    *,
    missing=None,
    extra=False,
    lane_extent=64,
    bytes_per_block=256,
    access_bytes=4,
    separate_radices=False,
    mismatched_quotient=False,
):
    qa, qb, da, db, la, lb, r, other_r = z3.Ints("qa qb da db la lb radix other_radix")
    ea, eb = z3.Ints("extra_a extra_b")
    rb = other_r if separate_radices else r
    fa, fb = qa * r + da, (2 * qb if mismatched_quotient else qb) * rb + db
    aa = bytes_per_block * (fa + (ea if extra else 0)) + 4 * la
    ab = bytes_per_block * (fb + (eb if extra else 0)) + 4 * lb
    bounds = {
        "positive": r >= 1,
        "lower_a": da >= 0,
        "upper_a": da < r,
        "lower_b": db >= 0,
        "upper_b": db < rb,
    }
    conditions = [value for name, value in bounds.items() if name != missing]
    conditions += [
        la >= 0,
        la < lane_extent,
        lb >= 0,
        lb < lane_extent,
        z3.Or(qa != qb, da != db),
        aa < ab + access_bytes,
        ab < aa + access_bytes,
    ]
    if separate_radices:
        conditions.append(other_r >= 1)
    correspondence = ((qb, qa), (db, da), (lb, la), (eb, ea), (other_r, r))
    variables = dict(qa=qa, qb=qb, da=da, db=db, la=la, lb=lb, r=r, rb=rb, ea=ea, eb=eb)
    return conditions, correspondence, variables


def _assert_witness(conditions, variables, **values):
    solver = z3.Solver()
    solver.add(
        *conditions, *(variables[name] == value for name, value in values.items())
    )
    assert solver.check() == z3.sat


def test_mixed_radix_precheck_proves_disjoint_blocks_without_pinning_radix():
    conditions, correspondence, _ = _formula()
    assert cs.conflict_impossible(conditions, correspondence)


@pytest.mark.parametrize(
    "missing", ["positive", "lower_a", "upper_a", "lower_b", "upper_b"]
)
def test_every_original_radix_domain_premise_is_required(missing):
    conditions, correspondence, variables = _formula(missing=missing)
    assert not cs.conflict_impossible(conditions, correspondence)
    if missing == "lower_a":
        _assert_witness(conditions, variables, r=2, qa=1, da=-1, qb=0, db=1, la=0, lb=0)
    if missing == "upper_a":
        _assert_witness(conditions, variables, r=2, qa=0, da=2, qb=1, db=0, la=0, lb=0)


def test_bound_inside_disjunction_does_not_authorize_injectivity():
    conditions, correspondence, variables = _formula(missing="upper_a")
    conditions.append(z3.Or(variables["da"] < variables["r"], variables["qa"] == 0))
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(conditions, variables, r=2, qa=0, da=2, qb=1, db=0, la=0, lb=0)


def test_copy_local_radices_never_become_one_shared_radix_by_shape_matching():
    conditions, correspondence, variables = _formula(separate_radices=True)
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(
        conditions, variables, r=2, rb=3, qa=3, qb=2, da=0, db=0, la=0, lb=0
    )


def test_different_quotient_coefficients_do_not_get_matching_shape_lemmas():
    conditions, correspondence, variables = _formula(mismatched_quotient=True)
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(conditions, variables, r=2, qa=2, qb=1, da=0, db=0, la=0, lb=0)


def test_outer_terms_may_cancel_the_flattened_index_difference():
    conditions, correspondence, variables = _formula(extra=True)
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(
        conditions, variables, r=2, qa=0, qb=0, da=0, db=1, ea=1, eb=0, la=0, lb=0
    )


def test_lane_range_can_carry_into_the_next_block():
    conditions, correspondence, variables = _formula(lane_extent=65)
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(conditions, variables, r=2, qa=0, qb=0, da=0, db=1, la=64, lb=0)


def test_partial_byte_overlap_is_not_replaced_by_start_address_equality():
    conditions, correspondence, variables = _formula(lane_extent=1, bytes_per_block=2)
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(conditions, variables, r=2, qa=0, qb=0, da=0, db=1, la=0, lb=0)


def _abc_shaped_solver(*, duplicate_columns=False):
    p0, p1, p2 = SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2
    g0, g1, g2 = z3.Ints("grid_0 grid_1 grid_2")
    row, col = z3.Ints("abc_row abc_col")
    addr = 1_000_000 + 4 * (
        4096 * (p0 * g2 + p2) + 4096 * p1 + 64 * row + (0 if duplicate_columns else col)
    )
    mask = z3.And(p1 * 64 + row >= 0, p1 * 64 + row < 64, col >= 0, col < 64)
    record = _scalar_store(addr, event_id=0, elem_size=4, mask=mask)
    grid = (1, 1, g2) if duplicate_columns else (g0, g1, g2)
    extra = (g2 == 1,) if duplicate_columns else ()
    return TwoCopySymbolicHBSolver(
        [record],
        grid=grid,
        extra_assumptions=extra,
        arange_dict={(0, 64, "row"): (row, None), (0, 64, "col"): (col, None)},
    )


def test_abc_shaped_cross_and_intra_queries_use_shortcut_and_keep_query_stats(
    monkeypatch,
):
    solver = _abc_shaped_solver()

    def full_query_must_not_run(*args):
        raise AssertionError("the necessary conflict conditions already prove UNSAT")

    monkeypatch.setattr(solver, "_race_query_is_sat", full_query_must_not_run)
    assert solver.find_races() == []
    assert [(kind, answer) for kind, _, answer in solver.query_stats] == [
        ("cross", False),
        ("intra", False),
    ]
    assert not solver.enum_used
    # Feasibility is still the original independent obligation.
    monkeypatch.setattr(solver, "_conflict_precheck", full_query_must_not_run)
    assert solver.check_feasibility()


def test_duplicate_columns_preserve_intra_instance_witness():
    solver = _abc_shaped_solver(duplicate_columns=True)
    reports = solver.find_races()
    assert reports
    assert any("single program instance" in report.reason for report in reports)
    assert any(kind == "intra" and answer for kind, _, answer in solver.query_stats)


@pytest.mark.parametrize("atomic", [False, True])
def test_partial_byte_overlap_survives_full_solver_and_atomic_scope_rules(atomic):
    g2 = z3.Int("grid_2")
    address = 2 * (SymbolicExpr.PID0 * g2 + SymbolicExpr.PID2)
    record = (
        _rmw_record(address, event_id=0, program_seq=0, elem_size=4)
        if atomic
        else _scalar_store(address, event_id=0, elem_size=4)
    )
    solver = TwoCopySymbolicHBSolver(
        [record], grid=(1, 1, g2), extra_assumptions=(g2 == 2,)
    )
    reports = solver.find_races()
    assert reports
    assert reports[0].witness_grid_a != reports[0].witness_grid_b


def test_unknown_precheck_returns_to_original_solver_budget(monkeypatch):
    settings = []

    class UnknownPrecheck:
        def set(self, **options):
            settings.append(options)

        def add(self, *conditions):
            pass

        def check(self):
            return z3.unknown

    monkeypatch.setattr(cs.z3, "SolverFor", lambda logic: UnknownPrecheck())
    g2 = z3.Int("grid_2")
    address = 2 * (SymbolicExpr.PID0 * g2 + SymbolicExpr.PID2)
    solver = TwoCopySymbolicHBSolver(
        [_scalar_store(address, event_id=0, elem_size=4)],
        grid=(1, 1, g2),
        extra_assumptions=(g2 == 2,),
    )
    assert solver.find_races()
    assert settings and all(options == {"timeout": 500} for options in settings)
    assert not solver.enum_used


@pytest.mark.parametrize("context", ["not", "antecedent", "ite"])
def test_unknown_boolean_subterms_remain_free_in_negative_positions(context):
    conditions, correspondence, variables = _formula(lane_extent=1, bytes_per_block=2)
    predicate = z3.Function("unmodeled_predicate", z3.IntSort(), z3.BoolSort())(
        variables["qa"]
    )
    condition = {
        "not": z3.Not(predicate),
        "antecedent": z3.Implies(predicate, z3.BoolVal(False)),
        "ite": z3.If(predicate, z3.BoolVal(False), z3.BoolVal(True)),
    }[context]
    conditions.append(condition)
    assert not cs.conflict_impossible(conditions, correspondence)
    _assert_witness(conditions, variables, r=2, qa=0, qb=0, da=0, db=1, la=0, lb=0)


@pytest.mark.parametrize("factor", ["radix", "snapshot"])
def test_private_ablation_switch_disables_only_its_optional_path(monkeypatch, factor):
    if factor == "radix":
        conditions, correspondence, _ = _formula()
        assert cs.conflict_impossible(conditions, correspondence)
        monkeypatch.setattr(cs, "_ENABLE_RADIX_PRECHECK", False)
        assert not cs.conflict_impossible(conditions, correspondence)
    else:
        table = z3.Array("ablation_snapshot", z3.IntSort(), z3.IntSort())
        value = z3.Select(table, z3.Int("ablation_index"))
        conditions = [value < 0, value >= 0]
        assert cs.conflict_impossible(conditions)
        monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
        assert not cs.conflict_impossible(conditions)


def test_original_launch_pins_allow_shortcut_without_hiding_any_grid_collision(
    monkeypatch,
):
    g0, g2 = z3.Ints("launch_grid_0 launch_grid_2")
    pid2 = SymbolicExpr.PID2
    table = z3.Array("launch_fallback", z3.IntSort(), z3.IntSort())
    address = z3.If(pid2 == 0, 4096, z3.Select(table, pid2))
    record = _scalar_store(
        address, event_id=0, elem_size=4, mask=z3.And(pid2 >= 0, pid2 < 1)
    )
    # The address omits pid0. A grid with two pid0 values really collides.
    any_grid = TwoCopySymbolicHBSolver([record], grid=(g0, 1, g2))
    a, b = any_grid.events
    assert not any_grid._conflict_precheck(a, b)
    assert any_grid.find_races()
    # These are precisely the extra assumptions used by a launch requery.
    pinned = TwoCopySymbolicHBSolver(
        [record],
        grid=(g0, 1, g2),
        extra_assumptions=(g0 == 1, g2 == 1),
    )

    def no_full_query(*args):
        raise AssertionError("the original launch pins already exclude another block")

    monkeypatch.setattr(pinned, "_race_query_is_sat", no_full_query)
    assert pinned.find_races() == []
    assert [(kind, answer) for kind, _, answer in pinned.query_stats] == [
        ("cross", False)
    ]
    assert not pinned.enum_used
