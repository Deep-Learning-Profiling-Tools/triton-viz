"""Production cache lifetimes and premise changes preserve query semantics."""

import z3

from triton_viz.clients.race_detector import two_copy_symbolic_hb_solver as tc
from triton_viz.clients.symbolic_engine import SymbolicExpr

from .test_conflict_precheck_cache import _store_solver
from .test_guarded_division import _check, _grouped
from .test_two_copy_symbolic_hb_solver import _scalar_store


def test_production_normalization_keeps_current_guards_with_shared_cache():
    solver = _store_solver()
    original, _, _, _, _ = _grouped()
    normalized = solver._normalized_race_query(original, ())
    base = solver._base_constraint_conjunction()
    assert normalized is not None
    assert _check(z3.Xor(z3.And(base, original), normalized))[0] == z3.unsat
    cache = solver._guarded_division_applicability_cache

    unmasked, (pid, lane, grid), _, size, _ = _grouped(masked=False)
    normalized = solver._normalized_race_query(unmasked, ())
    assert solver._guarded_division_applicability_cache is cache
    for instance, divisor in ((4, 0), (8, -2)):
        result, model_solver = _check(
            normalized, grid == 16, pid == instance, lane == 0
        )
        assert result == z3.sat
        assert model_solver.model().eval(size).as_long() == divisor


def test_production_snapshot_cache_respects_mutated_cells_and_removed_launch_pin():
    table = z3.Array("production_snapshot_table", z3.IntSort(), z3.IntSort())
    grid = z3.Int("production_snapshot_grid")
    cells = [table[index] == value for index, value in enumerate((2, 0, 1))]
    solver = tc.TwoCopySymbolicHBSolver(
        [_scalar_store(4 * table[SymbolicExpr.PID0], event_id=0, elem_size=4)],
        grid=(grid, 1, 1),
        extra_assumptions=(*cells, grid == 3),
    )
    a, b = solver.events
    assert solver._conflict_precheck(a, b)
    cache = solver._snapshot_lemma_cache
    assumptions = [*cells, grid == 3]
    solver.extra_assumptions = assumptions
    assert solver._conflict_precheck(a, b)
    assumptions[1] = table[1] == 2
    assert not solver._conflict_precheck(a, b)
    witness = z3.Solver()
    witness.add(
        solver._base_constraint_conjunction(),
        solver._race_expr(a, b),
        solver.different_blocks,
    )
    assert witness.check() == z3.sat
    assumptions[:] = cells
    # An unobserved table cell can alias a known destination outside the launch.
    assert not solver._conflict_precheck(a, b)
    assumptions.append(grid == 3)
    assert solver._conflict_precheck(a, b)
    assert solver._snapshot_lemma_cache is cache


def test_scan_caches_are_reset_for_each_public_invocation():
    solver = _store_solver()
    solver.find_races()
    guard_cache = solver._guarded_division_applicability_cache
    snapshot_cache = solver._snapshot_lemma_cache
    assert guard_cache is not None
    assert snapshot_cache is not None
    solver.find_races()
    assert solver._guarded_division_applicability_cache is not guard_cache
    assert solver._snapshot_lemma_cache is not snapshot_cache
    other = _store_solver()
    other.find_races()
    assert (
        other._guarded_division_applicability_cache
        is not solver._guarded_division_applicability_cache
    )
    assert other._snapshot_lemma_cache is not solver._snapshot_lemma_cache
