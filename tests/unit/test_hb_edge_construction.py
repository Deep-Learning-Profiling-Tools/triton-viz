"""Structural HB shortcuts preserve conditional edges and scope validation."""

from dataclasses import replace

import pytest
import z3

import triton_viz.clients.race_detector.two_copy_symbolic_hb_solver as tc_module
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_two_copy_symbolic_hb_solver import _scalar_load, _scalar_store


@pytest.fixture
def edge_solver():
    def make(*, fence_order=True, fence_seqs=(), dependency=False):
        active_load, active_store = z3.Bools("edge_active_load edge_active_store")
        load = _scalar_load(z3.IntVal(100), event_id=0, program_seq=0, mask=active_load)
        store = _scalar_store(
            z3.IntVal(100), event_id=1, program_seq=2, mask=active_store
        )
        if dependency:
            store.dep_loads = (load.event_id,)
        return TwoCopySymbolicHBSolver(
            [load, store],
            grid=(2, 1, 1),
            fence_order=fence_order,
            fence_seqs=fence_seqs,
        )

    return make


def _assert_equivalent(actual, expected):
    solver = z3.Solver()
    solver.add(actual != expected)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize(
    "fence_order,fence_seqs,dependency,ordered",
    [
        (False, (), False, True),
        (True, (), False, False),
        (True, (1,), False, True),
        (True, (), True, True),
        (True, (0, 2), False, False),
    ],
)
def test_missing_reads_through_preserves_all_conditional_po_edges(
    edge_solver, fence_order, fence_seqs, dependency, ordered
):
    core = edge_solver(
        fence_order=fence_order, fence_seqs=fence_seqs, dependency=dependency
    )
    assert not core.reads_through
    differences = []
    for first in core.events:
        for second in core.events:
            actual = core._edge(first, second)
            original = z3.Or(
                core._program_order(first, second),
                core._synchronizes_with(first, second),
            )
            expected = (
                z3.And(first.active, second.active)
                if ordered
                and first.copy == second.copy
                and first.program_seq < second.program_seq
                else z3.BoolVal(False)
            )
            differences.extend((actual != original, actual != expected))
    _assert_equivalent(z3.Or(*differences), z3.BoolVal(False))


def test_missing_reads_through_skips_sw_and_disjunction_construction(
    edge_solver, monkeypatch
):
    core = edge_solver(dependency=True)

    def forbidden(*args):
        raise AssertionError("absent reads-through must bypass SW and Or")

    monkeypatch.setattr(core, "_synchronizes_with", forbidden)
    monkeypatch.setattr(tc_module, "Or", forbidden)
    for first in core.events:
        for second in core.events:
            assert isinstance(core._edge(first, second), z3.BoolRef)


def test_existing_reads_through_keeps_cross_copy_cta_scope_condition(edge_solver):
    core = edge_solver()
    first = replace(core.events[0], sem="release", scope="cta")
    second = replace(core.events[-1], sem="acquire", scope="gpu")
    assert first.copy != second.copy
    rt = z3.Bool("edge_reads_through")
    core.reads_through[(first.idx, second.idx)] = rt
    expected = z3.And(rt, *(a == b for a, b in zip(first.pid, second.pid)))
    _assert_equivalent(core._edge(first, second), expected)


@pytest.mark.parametrize("rt_value", [False, True, None])
@pytest.mark.parametrize("invalid_endpoint", ["writer", "reader"])
def test_existing_reads_through_still_rejects_unknown_scope(
    edge_solver, rt_value, invalid_endpoint
):
    core = edge_solver()
    first = replace(
        core.events[0],
        sem="release",
        scope="unsupported-scope" if invalid_endpoint == "writer" else "gpu",
    )
    second = replace(
        core.events[-1],
        sem="acquire",
        scope="unsupported-scope" if invalid_endpoint == "reader" else "gpu",
    )
    # Presence matters even when the stored reads-through formula is False.
    core.reads_through[(first.idx, second.idx)] = (
        z3.Bool("invalid_scope_rt") if rt_value is None else z3.BoolVal(rt_value)
    )
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="unsupported memory scope"):
        core._edge(first, second)
