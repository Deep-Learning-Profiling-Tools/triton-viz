"""The sparse HB closure is equivalent to the original symbolic recurrence."""

import random

import pytest
import z3

from triton_viz.clients.race_detector import hb_common


def _dense_closure(edges):
    """Original layered Floyd-Warshall implementation, used only as an oracle."""
    reach = [list(row) for row in edges]
    for k in range(len(reach)):
        reach = [
            [
                z3.Or(reach[i][j], z3.And(reach[i][k], reach[k][j]))
                for j in range(len(reach))
            ]
            for i in range(len(reach))
        ]
    return reach


def _closure(edges):
    return hb_common.build_transitive_hb(
        list(range(len(edges))), lambda i, j: edges[i][j]
    )


def _assert_equivalent(actual, expected, *, ctx=None):
    assert len(actual) == len(expected)
    differences = []
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        differences.extend(a != b for a, b in zip(actual_row, expected_row))
    if differences:
        solver = z3.Solver(ctx=ctx)
        solver.add(z3.Or(*differences))
        assert solver.check() == z3.unsat


def test_empty_graph_never_calls_edge_function():
    def unreachable(*args):
        raise AssertionError("an empty graph has no initial edges")

    assert hb_common.build_transitive_hb([], unreachable) == []


@pytest.mark.parametrize("self_edge", [False, True])
def test_constant_self_edge_is_preserved_without_reflexive_completion(self_edge):
    result = _closure([[z3.BoolVal(self_edge)]])
    assert z3.is_true(result[0][0]) == self_edge
    assert z3.is_false(result[0][0]) != self_edge


def test_initial_edges_are_evaluated_once_in_original_order():
    calls = []

    def edge(i, j):
        calls.append((i, j))
        return z3.BoolVal(False)

    _ = hb_common.build_transitive_hb([0, 1, 2], edge)
    assert calls == [(i, j) for i in range(3) for j in range(3)]


def test_sparse_disconnected_graph_constructs_only_reachable_path(monkeypatch):
    p, q = z3.Bools("sparse_p sparse_q")
    edges = [[z3.BoolVal(False) for _ in range(24)] for _ in range(24)]
    edges[0][1], edges[1][2] = p, q
    calls = {"and": 0, "or": 0}

    def count_and(*args):
        calls["and"] += 1
        return z3.And(*args)

    def count_or(*args):
        calls["or"] += 1
        return z3.Or(*args)

    monkeypatch.setattr(hb_common, "And", count_and)
    monkeypatch.setattr(hb_common, "Or", count_or)
    result = _closure(edges)
    assert calls == {"and": 1, "or": 0}
    assert result[0][2].eq(z3.And(p, q))
    assert all(z3.is_false(result[i][i]) for i in range(24))
    assert z3.is_false(result[2][0])
    assert z3.is_false(result[0][23])


def test_initial_simplification_removes_only_identity_false_edges(monkeypatch):
    p = z3.Bool("initial_guard")
    edges = [
        [z3.And(p, z3.Not(p)), z3.And(p, True)],
        [z3.Or(False, False), z3.BoolVal(False)],
    ]

    def no_path_combination(*args):
        raise AssertionError("the initial graph has no paths of length two")

    monkeypatch.setattr(hb_common, "And", no_path_combination)
    monkeypatch.setattr(hb_common, "Or", no_path_combination)
    result = _closure(edges)
    assert z3.is_false(result[0][0])
    assert result[0][1].eq(p)
    assert z3.is_false(result[1][0])
    assert z3.is_false(result[1][1])


def test_true_edges_and_duplicate_conditions_fold_without_losing_guards():
    p = z3.Bool("guarded_path")
    f, t = z3.BoolVal(False), z3.BoolVal(True)
    edges = [[f, p, p, t], [f, f, p, f], [f, f, f, t], [f, f, f, f]]
    result = _closure(edges)
    _assert_equivalent(result, _dense_closure(edges))
    assert result[0][2].eq(p)
    assert z3.is_true(result[0][3])
    assert result[1][3].eq(p)


def test_unconditional_path_replaces_existing_conditional_edge():
    p = z3.Bool("conditional_shortcut")
    f, t = z3.BoolVal(False), z3.BoolVal(True)
    edges = [[f, t, p], [f, f, t], [f, f, f]]
    result = _closure(edges)
    assert z3.is_true(result[0][2])
    _assert_equivalent(result, _dense_closure(edges))


def test_acyclicity_rejects_forced_cycle_and_retains_conditional_cut():
    p, q = z3.Bools("acyclic_p acyclic_q")
    f = z3.BoolVal(False)
    result = _closure([[f, p], [q, f]])
    acyclic = z3.And(*(z3.Not(result[i][i]) for i in range(2)))
    _assert_equivalent([[acyclic]], [[z3.Not(z3.And(p, q))]])
    solver = z3.Solver()
    solver.add(acyclic, p, q)
    assert solver.check() == z3.unsat
    solver = z3.Solver()
    solver.add(acyclic, p, z3.Not(q))
    assert solver.check() == z3.sat


def test_conditional_cycle_and_self_loops_keep_diagonal_obligations():
    p, q, r, s = z3.Bools("cycle_p cycle_q cycle_r self_guard")
    f = z3.BoolVal(False)
    edges = [[s, p, f], [f, f, q], [r, f, f]]
    result = _closure(edges)
    _assert_equivalent(result, _dense_closure(edges))
    cycle = z3.And(p, q, r)
    _assert_equivalent(
        [[result[i][i] for i in range(3)]],
        [[z3.Or(s, cycle), cycle, cycle]],
    )


def test_conditional_sync_between_components_keeps_both_directions():
    a, b, release, acquire, rf, back = z3.Bools(
        "active_a active_b release acquire rf back"
    )
    f = z3.BoolVal(False)
    edges = [
        [f, a, f, f],
        [f, f, z3.And(release, acquire, rf), f],
        [f, f, f, b],
        [z3.And(back, z3.Not(rf)), f, f, f],
    ]
    _assert_equivalent(_closure(edges), _dense_closure(edges))


def test_arithmetic_and_array_edges_keep_their_full_conditions():
    index, limit = z3.Ints("closure_index closure_limit")
    snapshot = z3.Array("closure_snapshot", z3.IntSort(), z3.IntSort())
    f = z3.BoolVal(False)
    edges = [
        [f, z3.And(index >= 0, index < limit), f],
        [f, f, z3.Select(snapshot, index) == 7],
        [index == limit, f, f],
    ]
    _assert_equivalent(_closure(edges), _dense_closure(edges))


@pytest.mark.parametrize("seed,size", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
def test_small_symbolic_graphs_match_original_layered_recurrence(seed, size):
    random_source = random.Random(seed)
    p, q, r = z3.Bools(f"random_p_{seed} random_q_{seed} random_r_{seed}")
    choices = [
        z3.BoolVal(False),
        z3.BoolVal(True),
        p,
        q,
        r,
        z3.Not(p),
        z3.And(p, q),
        z3.Or(q, r),
    ]
    edges = [[random_source.choice(choices) for _ in range(size)] for _ in range(size)]
    _assert_equivalent(_closure(edges), _dense_closure(edges))


def test_non_default_z3_context_survives_constants_cycles_and_paths():
    ctx = z3.Context()
    p, q = z3.Bools("custom_p custom_q", ctx=ctx)
    f, t = z3.BoolVal(False, ctx=ctx), z3.BoolVal(True, ctx=ctx)
    edges = [[f, p, f], [t, f, q], [f, t, f]]
    result = _closure(edges)
    assert all(entry.ctx == ctx for row in result for entry in row)
    _assert_equivalent(result, _dense_closure(edges), ctx=ctx)
