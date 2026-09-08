"""Known-False shortcuts reuse normalization without changing proof premises."""

from collections import Counter

import pytest
import z3

from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector.snapshot_lemmas import SnapshotLemmaCache

from .test_conflict_precheck_cache import _store_solver
from .test_snapshot_precheck_integration import _conflict


def _unexpected(*args, **kwargs):
    raise AssertionError("known False must not prepare lemmas or an auxiliary solver")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("snapshot_cached", [False, True])
@pytest.mark.parametrize("custom_context", [False, True])
def test_false_original_condition_skips_lemma_scans_and_solver(
    monkeypatch, cached, snapshot_cached, custom_context
):
    context = z3.Context() if custom_context else None
    integer = z3.IntSort(ctx=context)
    table = z3.Array("false_original_table", integer, integer)
    index = z3.Int("false_original_index", ctx=context)
    conditions = [table[0] == 1, index < index]
    cache = cs._PureSelectExpressionCache() if cached else None
    snapshot = SnapshotLemmaCache() if snapshot_cached else None
    monkeypatch.setattr(cs, "snapshot_table_lemmas", _unexpected)
    monkeypatch.setattr(SnapshotLemmaCache, "lemmas", _unexpected)
    monkeypatch.setattr(cs.z3, "SolverFor", _unexpected)
    assert cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cache,
        snapshot_cache=snapshot,
    )


@pytest.mark.parametrize("cached", [False, True])
def test_prepared_normalization_is_reused_and_lemmas_use_original_conditions(
    monkeypatch, cached
):
    conditions, _, _ = _conflict()
    # Keep an intentionally unnormalized source node. The certificate must
    # receive this original node even though its normalized form is available.
    conditions[0] = z3.And(z3.BoolVal(True), conditions[0])
    originals = tuple(conditions)
    normalize_calls = Counter()
    normalize = z3.simplify
    cache = cs._PureSelectExpressionCache() if cached else None
    snapshot = SnapshotLemmaCache()
    extract = snapshot.lemmas
    observed_lemmas = []

    def counted_normalize(expression):
        normalize_calls[expression.get_id()] += 1
        return normalize(expression)

    def checked_lemmas(source):
        assert len(source) == len(originals)
        assert all(first is second for first, second in zip(source, originals))
        result = extract(source)
        observed_lemmas.extend(result)
        return result

    monkeypatch.setattr(cs.z3, "simplify", counted_normalize)
    monkeypatch.setattr(snapshot, "lemmas", checked_lemmas)
    assert cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cache,
        snapshot_cache=snapshot,
    )
    assert observed_lemmas
    # The native simplifier sees each original once, including without a
    # normalization cache. Rewritten conjunction simplification is separate.
    assert all(normalize_calls[condition.get_id()] == 1 for condition in originals)
    assert all(first is second for first, second in zip(conditions, originals))


@pytest.mark.parametrize("mode", ["general", "pure", "cached-pure"])
def test_false_relaxation_skips_solver_after_actual_select_abstraction(
    monkeypatch, mode
):
    table = z3.Array("false_relaxation_table", z3.IntSort(), z3.IntSort())
    value = table[z3.Int("false_relaxation_index")]
    predicate = value < 0
    conditions = [predicate, z3.Not(predicate)]
    assert not any(z3.is_false(z3.simplify(condition)) for condition in conditions)
    cache = cs._PureSelectExpressionCache() if mode == "cached-pure" else None
    lemma_calls = []
    original = cs.snapshot_table_lemmas

    def counted_lemmas(expression):
        lemma_calls.append(expression)
        return original(expression)

    monkeypatch.setattr(cs, "snapshot_table_lemmas", counted_lemmas)
    monkeypatch.setattr(cs.z3, "SolverFor", _unexpected)
    assert cs.conflict_impossible(
        conditions, simplify_first=mode != "general", expression_cache=cache
    )
    # The existing weak-first policy reaches pure-Select contradictions before
    # lemma preparation. The general path retains its original preparation.
    assert len(lemma_calls) == int(mode == "general")


def test_disabled_cached_snapshot_path_still_does_no_normalization(monkeypatch):
    cache = cs._PureSelectExpressionCache()
    condition = z3.BoolVal(False)
    assert z3.is_false(cache.normalize(condition))
    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
    monkeypatch.setattr(cache, "normalize", _unexpected)
    monkeypatch.setattr(cs, "snapshot_table_lemmas", _unexpected)
    monkeypatch.setattr(cs.z3, "SolverFor", _unexpected)
    assert not cs.conflict_impossible(
        [condition], simplify_first=True, expression_cache=cache
    )


def test_disabled_uncached_path_retains_its_existing_literal_false_answer(monkeypatch):
    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
    monkeypatch.setattr(cs, "snapshot_table_lemmas", _unexpected)
    monkeypatch.setattr(cs.z3, "SolverFor", _unexpected)
    assert cs.conflict_impossible([z3.BoolVal(False)], simplify_first=True)


@pytest.mark.parametrize("cached", [False, True])
def test_unsupported_normalization_preserves_the_existing_exception_boundary(cached):
    if cached:
        assert not cs.conflict_impossible(
            [object()],
            simplify_first=True,
            expression_cache=cs._PureSelectExpressionCache(),
        )
    else:
        with pytest.raises(z3.Z3Exception):
            cs.conflict_impossible([object()], simplify_first=True)


def test_general_path_does_not_gain_an_extra_normalization_or_smt_attempt(
    monkeypatch,
):
    condition = z3.Bool("general_precheck_condition")
    relaxation_inputs = []

    def unchanged_relaxation(conditions, *args, **kwargs):
        relaxation_inputs.append((conditions, kwargs))
        return None

    monkeypatch.setattr(cs, "_ENABLE_SNAPSHOT_PRECHECK", False)
    monkeypatch.setattr(cs.z3, "simplify", _unexpected)
    monkeypatch.setattr(cs.z3, "SolverFor", _unexpected)
    monkeypatch.setattr(cs, "_linear_relaxation", unchanged_relaxation)
    assert not cs.conflict_impossible([condition])
    assert len(relaxation_inputs) == 1
    assert relaxation_inputs[0][0][0] is condition
    assert relaxation_inputs[0][1]["normalized_conditions"] is None


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("answer", [z3.sat, z3.unknown])
def test_nonfalse_relaxation_keeps_one_original_budgeted_solver(
    monkeypatch, cached, answer
):
    table = z3.Array("nonfalse_table", z3.IntSort(), z3.IntSort())
    condition = table[z3.Int("nonfalse_index")] == 7
    calls = []

    class ProbeSolver:
        def set(self, **kwargs):
            calls.append(("set", kwargs))

        def add(self, expression):
            assert not z3.is_false(expression)
            calls.append(("add", None))

        def check(self):
            calls.append(("check", None))
            return answer

    def create(logic):
        calls.append(("create", logic))
        return ProbeSolver()

    monkeypatch.setattr(cs.z3, "SolverFor", create)
    assert not cs.conflict_impossible(
        [condition],
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache() if cached else None,
    )
    assert calls == [
        ("create", "QF_LIA"),
        ("set", {"timeout": 500}),
        ("add", None),
        ("check", None),
    ]


def test_literal_false_shortcut_does_not_replace_independent_feasibility():
    solver = _store_solver((z3.BoolVal(False),))
    assert solver.find_races() == []
    assert not solver.check_feasibility()
    assert not solver.enum_used


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("with_cells", [False, True])
def test_weak_failure_reuses_successful_normalization_for_original_guarded_path(
    monkeypatch, cached, with_cells
):
    table = z3.Array("failed_weak_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("failed_weak_index")
    conditions = [table[index] >= 0]
    if with_cells:
        conditions += [table[0] == 0, table[1] == 1, index >= 0, index < 2]
    original_simplify = z3.simplify
    normalization_calls = Counter()
    solver_attempts = []

    def counted_normalize(expression):
        normalization_calls[expression.get_id()] += 1
        return original_simplify(expression)

    class ProbeSolver:
        def set(self, **kwargs):
            assert kwargs == {"timeout": 500}

        def add(self, expression):
            assert not z3.is_false(expression)

        def check(self):
            solver_attempts.append(None)
            if len(solver_attempts) == 1:
                raise z3.Z3Exception("the existing optional weak check failed")
            return z3.sat

    monkeypatch.setattr(cs.z3, "simplify", counted_normalize)
    monkeypatch.setattr(cs.z3, "SolverFor", lambda logic: ProbeSolver())
    assert not cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache() if cached else None,
        snapshot_cache=SnapshotLemmaCache(),
    )
    assert len(solver_attempts) == 2
    assert all(normalization_calls[condition.get_id()] == 1 for condition in conditions)


def test_failed_uncached_weak_normalization_retries_the_original_guarded_path(
    monkeypatch,
):
    table = z3.Array("normalization_retry_table", z3.IntSort(), z3.IntSort())
    condition = table[z3.Int("normalization_retry_index")] == 7
    original_simplify = z3.simplify
    attempts = []

    def fail_first(expression):
        if expression is condition:
            attempts.append(None)
            if len(attempts) == 1:
                raise z3.Z3Exception("optional preparation failed")
        return original_simplify(expression)

    monkeypatch.setattr(cs.z3, "simplify", fail_first)
    assert not cs.conflict_impossible([condition], simplify_first=True)
    assert len(attempts) == 2
