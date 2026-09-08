"""Cached original-premise facts survive changing pairs without widening proofs."""

import pytest
import z3

from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector.snapshot_lemmas import (
    SnapshotLemmaCache,
    snapshot_table_lemmas,
)


def _case(values=(2, 0, 1), *, context=None):
    integer = z3.IntSort(ctx=context)
    table = z3.Array("destination", integer, integer)
    a, b = z3.Ints("a b", ctx=context)
    cells = [table[index] == value for index, value in enumerate(values)]
    reads = [
        table[a] == z3.Int("va", ctx=context),
        table[b] == z3.Int("vb", ctx=context),
    ]
    return cells, reads, table, a, b


def _assert_entailed(conditions, lemmas):
    solver = z3.Solver(ctx=conditions[0].ctx)
    solver.set(timeout=2000)
    solver.add(*conditions, z3.Not(z3.And(*lemmas)))
    assert solver.check() == z3.unsat


def _assert_collision(conditions, lemmas, a, b, table):
    solver = z3.Solver(ctx=table.ctx)
    solver.set(timeout=2000)
    solver.add(*conditions, *lemmas, a == 0, b == 1, table[a] == table[b])
    assert solver.check() == z3.sat


def test_common_table_cells_and_certificate_are_visited_once(monkeypatch):
    cells, reads, table, a, b = _case(tuple(range(1024)))
    original_cell = SnapshotLemmaCache._cell_equation
    original_certify = SnapshotLemmaCache._certify
    cell_calls, certificate_calls = {}, []

    def cell(node):
        result = original_cell(node)
        if result is not None:
            cell_calls[node] = cell_calls.get(node, 0) + 1
        return result

    def certify(roots):
        certificate_calls.append(roots)
        return original_certify(roots)

    monkeypatch.setattr(SnapshotLemmaCache, "_cell_equation", staticmethod(cell))
    monkeypatch.setattr(SnapshotLemmaCache, "_certify", staticmethod(certify))
    cache = SnapshotLemmaCache()
    common = z3.And(*cells)
    first = cache.lemmas([common, *reads, a != b])
    assert len(first) == 3
    for pair in range(10):
        # A different pair root wraps the same table, and traverses shared reads.
        conditions = [z3.And(common, a >= pair), *reads, b > pair]
        lemmas = cache.lemmas(conditions)
        assert all(new is old for new, old in zip(lemmas, first))
    assert len(cell_calls) == len(cells)
    assert set(cell_calls.values()) == {1}
    assert len(certificate_calls) == 1
    # Existing certificate and read summaries also handle a newly added read.
    c = z3.Int("new_pair_index")
    assert len(cache.lemmas([common, *reads, table[c] >= 0])) == 6
    assert len(certificate_calls) == 1
    # Substitution can change non-cell conjuncts while preserving every cell.
    joined = z3.And(*cells, *reads)
    cache.lemmas([joined])
    cache.lemmas([z3.substitute(joined, (b, a))])
    assert len(certificate_calls) == 1


@pytest.mark.parametrize("change", ["remove", "duplicate", "conflict", "conditional"])
def test_mutated_conditions_revalidate_current_cell_premises(change):
    cells, reads, table, a, b = _case()
    conditions = [*cells, *reads]
    cache = SnapshotLemmaCache()
    assert len(cache.lemmas(conditions)) == 3
    if change == "remove":
        del conditions[1]
    elif change == "duplicate":
        conditions[1] = table[1] == 2
    elif change == "conflict":
        conditions.append(table[1] == 2)
    else:
        conditions[:3] = [z3.Or(z3.Bool("skip_cells"), z3.And(*cells))]
    lemmas = cache.lemmas(conditions)
    assert len(lemmas) == (2 if change == "duplicate" else 0)
    if change != "conflict":
        _assert_collision(conditions, lemmas, a, b, table)


def test_cells_split_across_roots_are_combined_and_rechecked():
    cells, reads, table, a, b = _case()
    cache = SnapshotLemmaCache()
    assert cache.lemmas([cells[0], cells[2], *reads]) == ()
    conditions = [z3.And(cells[0], cells[2]), cells[1], *reads]
    _assert_entailed(conditions, cache.lemmas(conditions))
    assert len(cache.lemmas(conditions)) == 3
    assert cache.lemmas([conditions[0], *reads]) == ()


def test_custom_contexts_and_substituted_reads_do_not_share_foreign_facts():
    cache = SnapshotLemmaCache()
    for context in (None, z3.Context(), z3.Context()):
        cells, reads, _, a, b = _case(context=context)
        first = cache.lemmas([*cells, *reads])
        assert len(first) == 3
        intra = [z3.substitute(read, (b, a)) for read in reads]
        assert len(cache.lemmas([*cells, *intra])) == 1
        _assert_entailed([*cells, *intra], cache.lemmas([*cells, *intra]))
        assert all(lemma.ctx == a.ctx for lemma in first)


def test_quantified_and_conditional_cells_never_lift_into_asserted_facts():
    cells, reads, table, a, b = _case()
    cache = SnapshotLemmaCache()
    assert len(cache.lemmas([*cells, *reads])) == 3
    assert cache.lemmas([*cells, z3.Exists([a, b], table[a] != table[b])]) == ()
    assert cache.lemmas([z3.Implies(z3.Bool("guard"), z3.And(*cells)), *reads]) == ()
    assert cache.lemmas([z3.Exists([a], z3.And(*cells)), *reads]) == ()


def test_cached_facts_match_standalone_and_preserve_outside_values():
    cells, reads, table, a, b = _case()
    conditions = [z3.And(*cells), *reads]
    cache = SnapshotLemmaCache()
    lemmas = cache.lemmas(conditions)
    expected = snapshot_table_lemmas(z3.And(*conditions))
    assert [lemma.sexpr() for lemma in lemmas] == [lemma.sexpr() for lemma in expected]
    _assert_entailed(conditions, lemmas)
    solver = z3.Solver()
    solver.add(*conditions, *lemmas, a == 0, b == 10, table[a] == table[b])
    assert solver.check() == z3.sat


def test_current_read_set_controls_optional_injectivity_limit():
    cells, reads, table, _, _ = _case()
    cache = SnapshotLemmaCache()
    assert len(cache.lemmas([*cells, *reads])) == 3
    many = [table[z3.Int(f"many_{i}")] >= 0 for i in range(17)]
    assert len(cache.lemmas([*cells, *many])) == 17
    assert len(cache.lemmas([*cells, *reads])) == 3


def test_production_precheck_retains_launch_and_changed_table_collisions():
    cells, _, table, a, b = _case()
    expression_cache = cs._PureSelectExpressionCache()
    snapshot_cache = SnapshotLemmaCache()
    conditions = [*cells, a >= 0, b >= 0, a != b, table[a] == table[b]]

    def impossible():
        return cs.conflict_impossible(
            conditions,
            simplify_first=True,
            expression_cache=expression_cache,
            snapshot_cache=snapshot_cache,
        )

    assert not impossible()
    conditions.extend([a < 3, b < 3])
    assert impossible()
    del conditions[1]
    assert not impossible()
    conditions.insert(1, table[1] == 2)
    assert not impossible()


def test_disabled_factor_does_not_consult_populated_snapshot_cache(monkeypatch):
    cells, reads, _, _, _ = _case()
    cache = SnapshotLemmaCache()
    assert cache.lemmas([*cells, *reads])

    def disabled(*args, **kwargs):
        raise AssertionError("disabled snapshot extraction must not run")

    monkeypatch.setattr(cache, "lemmas", disabled)
    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
    assert not cs.conflict_impossible([*cells, *reads], snapshot_cache=cache)
