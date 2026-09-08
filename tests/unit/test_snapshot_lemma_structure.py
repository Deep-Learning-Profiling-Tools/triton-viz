"""Structural work bounds around the existing literal snapshot-cell shortcut.

The literal-cell and empty-read behavior predates these tests (f5a8466).
These tests cover the subsequent candidate gate, selective certificates, and
child-wrapper reuse without using elapsed time as a correctness assertion.
"""

from collections import Counter

import pytest
import z3

from triton_viz.clients.race_detector.snapshot_lemmas import (
    SnapshotLemmaCache,
    snapshot_table_lemmas,
)


def _array(name):
    return z3.Array(name, z3.IntSort(), z3.IntSort())


def _same_as_standalone(cache, conditions):
    expected = snapshot_table_lemmas(z3.And(*conditions))
    actual = cache.lemmas(conditions)
    assert [lemma.sexpr() for lemma in actual] == [lemma.sexpr() for lemma in expected]
    return actual


@pytest.mark.parametrize("wrapper", ["or", "implies", "quantifier", "none"])
def test_absent_top_level_cells_skip_read_summaries(monkeypatch, wrapper):
    table = _array("candidate_table")
    index = z3.Int("candidate_index")
    cells = z3.And(*(table[i] == i for i in range(64)))
    read = table[index] > 0
    guard = z3.Bool("candidate_guard")
    conditions = {
        "or": [z3.Or(guard, cells), read],
        "implies": [z3.Implies(guard, cells), read],
        "quantifier": [z3.Exists([index], z3.And(cells, read))],
        "none": [z3.And(guard, read)],
    }[wrapper]
    cache = SnapshotLemmaCache()

    def unexpected(*args):
        raise AssertionError("a formula without asserted cells needs no read scan")

    monkeypatch.setattr(cache, "_summary", unexpected)
    monkeypatch.setattr(cache, "_fact_summary", unexpected)
    monkeypatch.setattr(cache, "_certify", unexpected)
    assert cache.lemmas(conditions) == ()


@pytest.mark.parametrize("read_kind", ["none", "unrelated", "bound", "literal"])
def test_absent_applicable_reads_skip_fact_assembly_and_certification(
    monkeypatch, read_kind
):
    table, other = _array("read_table"), _array("unrelated_table")
    index = z3.Int("read_index")
    cells = z3.And(*(table[i] == i for i in range(64)))
    condition = {
        "none": index > 0,
        "unrelated": other[index] > 0,
        "bound": z3.Exists([index], table[index] > 0),
        "literal": table[0] > 0,
    }[read_kind]
    cache = SnapshotLemmaCache()

    def unexpected(*args):
        raise AssertionError("no current table read needs a certificate")

    monkeypatch.setattr(cache, "_fact_summary", unexpected)
    monkeypatch.setattr(cache, "_certify", unexpected)
    monkeypatch.setattr(z3.IntNumRef, "as_long", unexpected)
    assert cache.lemmas([cells, condition]) == ()


def test_only_current_arrays_decode_complete_original_cell_facts(monkeypatch):
    a, b, unused = (_array(name) for name in ("a_table", "b_table", "unused_table"))
    index, other = z3.Ints("selected_index selected_other")
    # Interleave tables and split a table between roots. Every cell of the
    # selected array must participate, including a duplicate value at key 1.
    cells = [
        z3.And(b[0] == 5, a[0] == 2, unused[0] == 999),
        z3.And(a[2] == 0, b[1] == 6, a[1] == 2),
    ]
    cache = SnapshotLemmaCache()
    original = cache._certify
    certified = []

    def certify(roots):
        pending, seen, found = list(roots), set(), []
        while pending:
            facts = pending.pop()
            if facts in seen:
                continue
            seen.add(facts)
            if facts.cell is None:
                pending.extend(facts.children)
            else:
                found.append(facts.cell)
        certified.append(found)
        return original(roots)

    monkeypatch.setattr(cache, "_certify", certify)
    reads_a = [a[index] >= 0, a[other] >= 0]
    assert len(_same_as_standalone(cache, [*cells, *reads_a])) == 2
    assert len(certified) == 1
    assert {cell[0] for cell in certified[0]} == {a}
    assert {cell[1].as_long() for cell in certified[0]} == {0, 1, 2}

    # Adding a read of B certifies B, reuses all of A, and restores original
    # fact order (B before A), rather than discovery order of the read set.
    reads_b = [b[index] >= 0, b[other] >= 0]
    assert len(_same_as_standalone(cache, [*cells, *reads_a, *reads_b])) == 5
    assert len(certified) == 2
    assert {cell[0] for cell in certified[1]} == {b}

    # An unrelated mutation leaves A's complete premises unchanged.
    changed = [z3.And(b[0] == 99, a[0] == 2, unused[0] == 999), cells[1]]
    assert len(_same_as_standalone(cache, [*changed, *reads_a])) == 2
    assert len(certified) == 2
    # An added conflicting cell for A must invalidate its previous certificate.
    assert _same_as_standalone(cache, [*changed, a[1] == 8, *reads_a]) == ()
    assert len(certified) == 3
    assert len(certified[-1]) == 4


def test_irrelevant_literal_values_are_not_decoded(monkeypatch):
    relevant, unused = _array("decode_relevant"), _array("decode_unused")
    index = z3.Int("decode_index")
    conditions = [relevant[0] == 7, unused[0] == 999, relevant[index] > 0]
    original = z3.IntNumRef.as_long
    decoded = []

    def as_long(value):
        result = original(value)
        decoded.append(result)
        return result

    monkeypatch.setattr(z3.IntNumRef, "as_long", as_long)
    assert len(SnapshotLemmaCache().lemmas(conditions)) == 1
    assert Counter(decoded) == Counter([0, 7])


def test_each_ast_children_tuple_is_built_once_across_all_passes(monkeypatch):
    table = _array("child_table")
    index, other = z3.Ints("child_index child_other")
    cells = z3.And(*(table[i] == i for i in range(128)))
    shared = table[index + 1]
    pair = z3.And(shared >= 0, shared == table[other], index != other)
    conditions = [z3.And(cells, pair), pair]
    expected = snapshot_table_lemmas(z3.And(*conditions))
    original = z3.ExprRef.children
    scans = Counter()

    def children(node):
        scans[node] += 1
        return original(node)

    monkeypatch.setattr(z3.ExprRef, "children", children)
    cache = SnapshotLemmaCache()
    actual = cache.lemmas(conditions)
    assert [lemma.sexpr() for lemma in actual] == [lemma.sexpr() for lemma in expected]
    first_scans = scans.copy()
    assert first_scans and set(first_scans.values()) == {1}
    assert all(new is old for new, old in zip(cache.lemmas(conditions), actual))
    assert scans == first_scans


def test_literal_shortcut_preserves_complex_array_and_nested_index_reads():
    table, second = _array("nested_table"), _array("second_table")
    index = z3.Int("nested_index")
    composite = z3.Store(second, table[index], table[second[index]])
    conditions = [
        second[0] == 2,
        z3.And(table[0] == 4, second[1] == 3, table[1] == 5),
        composite[0] == 9,
        composite[index] > 0,
        z3.If(table[index] == 4, second[index] > 0, table[second[index]] > 0),
    ]
    lemmas = _same_as_standalone(SnapshotLemmaCache(), conditions)
    assert len(lemmas) == 5


def test_context_switch_clears_candidate_and_child_caches():
    cache = SnapshotLemmaCache()
    first = z3.Context()
    second = z3.Context()
    for context in (first, second, first):
        integer = z3.IntSort(ctx=context)
        table = z3.Array("context_table", integer, integer)
        index = z3.Int("context_index", ctx=context)
        assert len(cache.lemmas([table[0] == 1, table[index] > 0])) == 1
        assert all(node.ctx == context for node in cache._children)
        assert all(node.ctx == context for node in cache._cells)
        assert all(node.ctx == context for node, _ in cache._candidates)
        assert all(node.ctx == context for node in cache._fact_summaries)
