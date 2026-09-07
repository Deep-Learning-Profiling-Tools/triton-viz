"""Snapshot lemmas preserve scope through the production conflict precheck."""

import z3

from triton_viz.clients.race_detector import conflict_simplification as cs


def _conflict(values=(2, 0, 1), *, launch=True):
    table = z3.Array("destination", z3.IntSort(), z3.IntSort())
    pa, pb, la, lb, grid = z3.Ints("pa pb la lb grid")
    address_a, address_b = 16 * table[pa] + 4 * la, 16 * table[pb] + 4 * lb
    conditions = [
        *(table[index] == value for index, value in enumerate(values)),
        grid >= 1,
        pa >= 0,
        pb >= 0,
        pa < grid,
        pb < grid,
        pa != pb,
        la >= 0,
        lb >= 0,
        la < 4,
        lb < 4,
        address_a < address_b + 4,
        address_b < address_a + 4,
    ]
    if launch:
        conditions.append(grid == len(values))
    return conditions, table, (pa, pb, la, lb, grid)


def _precheck(conditions):
    return cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache(),
    )


def _assert_sat(conditions, *witness):
    solver = z3.Solver()
    solver.set(timeout=1000)
    solver.add(*conditions, *witness)
    assert solver.check() == z3.sat


def test_permutation_lemmas_discharge_existing_launch_conflict():
    conditions, _, _ = _conflict()
    assert _precheck(conditions)


def test_any_grid_query_retains_out_of_snapshot_collision():
    conditions, table, (pa, pb, la, lb, grid) = _conflict(launch=False)
    _assert_sat(
        conditions,
        grid == 4,
        pa == 0,
        pb == 3,
        la == 0,
        lb == 0,
        table[3] == table[0],
    )
    assert not _precheck(conditions)


def test_duplicate_destination_cells_preserve_existing_launch_collision():
    conditions, _, (pa, pb, la, lb, _) = _conflict((2, 2, 1))
    _assert_sat(conditions, pa == 0, pb == 1, la == 0, lb == 0)
    assert not _precheck(conditions)


def test_disabled_snapshot_factor_does_not_generate_or_use_lemmas(monkeypatch):
    conditions, _, _ = _conflict()
    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)

    def disabled(*args, **kwargs):
        raise AssertionError("the disabled snapshot factor must not derive lemmas")

    monkeypatch.setattr(cs, "snapshot_table_lemmas", disabled)
    assert not _precheck(conditions)
