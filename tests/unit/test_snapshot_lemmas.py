"""Original finite equations certify guarded lemmas without changing scope."""

import pytest
import z3

from triton_viz.clients.race_detector.snapshot_lemmas import snapshot_table_lemmas
from triton_viz.clients.race_detector import snapshot_lemmas


def _query(values=(9, 4, 7), *, start=0, context=None):
    integer = z3.IntSort(ctx=context)
    array = z3.Array("snapshot", integer, integer)
    a, b, observed_a, observed_b = z3.Ints("a b observed_a observed_b", ctx=context)
    cells = [array[start + i] == value for i, value in enumerate(values)]
    expression = z3.And(*cells, array[a] == observed_a, array[b] == observed_b)
    return expression, array, a, b


def _check(expression, *extra):
    solver = z3.Solver(ctx=expression.ctx)
    solver.set(timeout=2000)
    solver.add(expression, *extra)
    answer = solver.check()
    assert answer != z3.unknown
    return answer


def _certify(expression):
    lemmas = snapshot_table_lemmas(expression)
    assert lemmas
    assert _check(expression, z3.Not(z3.And(*lemmas))) == z3.unsat
    return z3.And(expression, *lemmas)


@pytest.mark.parametrize("start", [-8, 0, 13])
def test_exact_interval_range_and_injectivity_are_entailed(start):
    expression, array, a, b = _query(start=start)
    augmented = _certify(expression)
    assert len(snapshot_table_lemmas(expression)) == 3
    assert (
        _check(
            augmented,
            a >= start,
            a <= start + 2,
            b >= start,
            b <= start + 2,
            a != b,
            array[a] == array[b],
        )
        == z3.unsat
    )


def test_duplicate_values_preserve_collision():
    expression, array, a, b = _query((9, 9, 7))
    augmented = _certify(expression)
    assert len(snapshot_table_lemmas(expression)) == 2
    assert _check(augmented, a == 0, b == 1, array[a] == array[b]) == z3.sat


def test_missing_cell_does_not_certify_interval():
    expression, array, a, b = _query()
    expression = z3.And(
        *[atom for atom in expression.children() if not z3.eq(atom, array[1] == 4)]
    )
    assert snapshot_table_lemmas(expression) == ()
    assert _check(expression, a == 0, b == 1, array[a] == array[b]) == z3.sat


def test_out_of_domain_equal_values_and_extreme_values_remain_possible():
    expression, array, a, b = _query()
    augmented = _certify(expression)
    assert _check(augmented, a == 0, b == 1000000, array[a] == array[b]) == z3.sat
    assert (
        _check(augmented, a == -1, b == 3, array[a] == -100, array[b] == 100) == z3.sat
    )


@pytest.mark.parametrize("connective", ["or", "implies", "ite"])
def test_conditional_equations_never_become_snapshot_facts(connective):
    expression, array, a, b = _query()
    cells, reads = expression.children()[:3], expression.children()[3:]
    flag = z3.Bool("flag")
    conditional = {
        "or": z3.Or(flag, z3.And(*cells)),
        "implies": z3.Implies(flag, z3.And(*cells)),
        "ite": z3.If(flag, z3.And(*cells), z3.BoolVal(True)),
    }[connective]
    original = z3.And(conditional, *reads)
    assert snapshot_table_lemmas(original) == ()
    assert (
        _check(
            original,
            flag if connective == "or" else z3.Not(flag),
            a == 0,
            b == 1,
            array[a] == array[b],
        )
        == z3.sat
    )


def test_partial_byte_overlap_survives_injective_start_addresses():
    expression, array, a, b = _query((0, 1, 2))
    augmented = _certify(expression)
    address_a, address_b = 2 * array[a], 2 * array[b]
    assert (
        _check(
            augmented,
            a == 0,
            b == 1,
            address_a < address_b + 4,
            address_b < address_a + 4,
        )
        == z3.sat
    )


def test_custom_context_reversed_equations_and_nested_conjunctions():
    expression, _, _, _ = _query(context=z3.Context())
    atoms = expression.children()
    reversed_cells = [atom.arg(1) == atom.arg(0) for atom in atoms[:3]]
    original = z3.And(z3.And(*reversed_cells), z3.And(*atoms[3:]))
    _certify(original)


def test_conflicting_original_cells_are_declined():
    expression, array, _, _ = _query()
    assert snapshot_table_lemmas(z3.And(expression, array[0] == 8)) == ()


def test_quantified_reads_are_not_lifted_out_of_binder():
    expression, array, a, b = _query()
    original = z3.And(
        *expression.children()[:3], z3.Exists([a, b], array[a] != array[b])
    )
    assert snapshot_table_lemmas(original) == ()


def test_unrelated_arrays_do_not_share_injectivity():
    expression, array, a, _ = _query()
    other = z3.Array("other", z3.IntSort(), z3.IntSort())
    original = z3.And(expression, other[a] == array[0])
    augmented = _certify(original)
    assert _check(augmented, a == 1, other[a] == 9) == z3.sat


def test_many_reads_keep_linear_range_facts_without_quadratic_pair_lemmas():
    expression, array, _, _ = _query()
    count = snapshot_lemmas._MAX_INJECTIVITY_READS + 1
    indices = [z3.Int(f"many_index_{i}") for i in range(count)]
    original = z3.And(
        *expression.children()[:3],
        *(array[index] == z3.Int(f"many_value_{i}") for i, index in enumerate(indices)),
    )
    assert len(snapshot_table_lemmas(original)) == count
