"""Deferred snapshot facts retain the original check and fallback budgets."""

import pytest
import z3

from triton_viz.clients.race_detector import conflict_simplification as cs
from triton_viz.clients.race_detector.snapshot_lemmas import SnapshotLemmaCache

from .test_conflict_simplification import _formula


def _range_query():
    table = z3.Array("lazy_table", z3.IntSort(), z3.IntSort())
    index = z3.Int("lazy_index")
    return [table[0] == 2, table[1] == 3, index >= 0, index < 2, table[index] >= 4]


def _observe_checks(monkeypatch, forced=()):
    original = z3.SolverFor
    checks = []

    class ObservedSolver:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)
            self.timeout = None

        def set(self, **kwargs):
            self.timeout = kwargs.get("timeout")
            self.inner.set(**kwargs)

        def add(self, *expressions):
            self.inner.add(*expressions)

        def check(self):
            ordinal = len(checks)
            answer = forced[ordinal] if ordinal < len(forced) else self.inner.check()
            checks.append((self.timeout, answer))
            return answer

    monkeypatch.setattr(cs.z3, "SolverFor", ObservedSolver)
    return checks


@pytest.mark.parametrize("cached", [False, True])
def test_weak_unsat_avoids_snapshot_extraction(monkeypatch, cached):
    table = z3.Array("lazy_unused_table", z3.IntSort(), z3.IntSort())
    index, limit = z3.Ints("lazy_unused_index lazy_limit")
    conditions = [table[index] >= 0, limit >= 7, limit < 7]
    checks = _observe_checks(monkeypatch)
    snapshot_cache = SnapshotLemmaCache() if cached else None

    def reject_extraction(*args):
        raise AssertionError("UNSAT of original necessary conditions needs no lemma")

    if cached:
        monkeypatch.setattr(snapshot_cache, "lemmas", reject_extraction)
    else:
        monkeypatch.setattr(cs, "snapshot_table_lemmas", reject_extraction)
    assert cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache() if cached else None,
        snapshot_cache=snapshot_cache,
    )
    assert checks == [(500, z3.unsat)]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("force_unknown", [False, True])
def test_inconclusive_weak_query_keeps_guarded_check_and_its_full_budget(
    monkeypatch, cached, force_unknown
):
    conditions = _range_query()
    before = tuple(conditions)
    checks = _observe_checks(monkeypatch, (z3.unknown,) if force_unknown else ())
    assert cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache() if cached else None,
        snapshot_cache=SnapshotLemmaCache() if cached else None,
    )
    assert checks == [(500, z3.unknown if force_unknown else z3.sat), (500, z3.unsat)]
    assert len(conditions) == len(before)
    assert all(current is original for current, original in zip(conditions, before))


@pytest.mark.parametrize("answer", [z3.sat, z3.unknown])
def test_empty_lemmas_reuse_the_original_check_without_an_added_budget(
    monkeypatch, answer
):
    table = z3.Array("lazy_no_cells", z3.IntSort(), z3.IntSort())
    index = z3.Int("lazy_no_cells_index")
    checks = _observe_checks(monkeypatch, (answer,))
    assert not cs.conflict_impossible(
        [table[index] == 7],
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache(),
        snapshot_cache=SnapshotLemmaCache(),
    )
    assert checks == [(500, answer)]


def test_unsupported_weak_relaxation_does_not_skip_the_original_guarded_path(
    monkeypatch,
):
    original = cs._linear_relaxation
    calls = []

    def unsupported_first(conditions, *args, **kwargs):
        calls.append(tuple(conditions))
        return None if len(calls) == 1 else original(conditions, *args, **kwargs)

    monkeypatch.setattr(cs, "_linear_relaxation", unsupported_first)
    checks = _observe_checks(monkeypatch)
    conditions = _range_query()
    assert cs.conflict_impossible(conditions, simplify_first=True)
    assert len(calls) == 2 and len(calls[1]) > len(calls[0])
    assert checks == [(500, z3.unsat)]


@pytest.mark.parametrize("exception", [TypeError, z3.Z3Exception])
def test_failed_optional_lemma_extraction_retains_the_inconclusive_answer(
    monkeypatch, exception
):
    def unsupported(*args):
        raise exception("unsupported optional snapshot facts")

    monkeypatch.setattr(cs, "snapshot_table_lemmas", unsupported)
    checks = _observe_checks(monkeypatch, (z3.unknown,))
    assert not cs.conflict_impossible(_range_query(), simplify_first=True)
    assert checks == [(500, z3.unknown)]


@pytest.mark.parametrize("exception", [TypeError, z3.Z3Exception])
@pytest.mark.parametrize("stage", ["_linear_relaxation", "_relaxation_impossible"])
@pytest.mark.parametrize("with_cells", [False, True])
def test_weak_attempt_exception_keeps_the_original_check_with_its_full_budget(
    monkeypatch, exception, stage, with_cells
):
    if with_cells:
        conditions = _range_query()
    else:
        table = z3.Array("lazy_retry_no_cells", z3.IntSort(), z3.IntSort())
        index = z3.Int("lazy_retry_no_cells_index")
        conditions = [table[index] >= 7, table[index] < 7]
    original = getattr(cs, stage)
    calls = []

    def failed_first(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise exception("optional weak attempt failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(cs, stage, failed_first)
    checks = _observe_checks(monkeypatch)
    assert cs.conflict_impossible(conditions, simplify_first=True)
    assert len(calls) == 2
    assert checks == [(500, z3.unsat)]


def test_radix_path_still_derives_lemmas_before_its_only_check(monkeypatch):
    conditions, correspondence, _ = _formula()
    checks = _observe_checks(monkeypatch)
    original = cs.snapshot_table_lemmas
    extractions = []

    def observed(expression):
        assert not checks
        extractions.append(expression)
        return original(expression)

    monkeypatch.setattr(cs, "snapshot_table_lemmas", observed)
    assert cs.conflict_impossible(conditions, correspondence)
    assert len(extractions) == 1
    assert checks == [(500, z3.unsat)]
