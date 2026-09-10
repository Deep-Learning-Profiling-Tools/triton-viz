"""Solver-level tests for the enumeration fallback (the concretization
ladder's last rung): a Z3-undecided race query is re-asked as an
exhaustive concrete-pid case split at the launch extent.

The fallback is forced deterministically by monkeypatching
``_race_query_is_sat`` to raise on every symbolic attempt, so each test
exercises the enumeration path itself, not Z3's timeout behavior:

  * equivalence on a race-free system — every case UNSAT, no reports,
    ``enum_used`` set (the caller degrades the claim to launch scope);
  * a real race is still found — some case SAT, with a usable model
    whose witness pids are in-extent by construction;
  * fail-closed refusals — no fallback grid, or a case count above
    ``ENUM_MAX_CASES``, re-raise the original unknown unchanged.
"""

from __future__ import annotations

import pytest
from z3 import IntVal

from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.clients.symbolic_engine import SymbolicExpr
from triton_viz.core.data import Store

DATA_BASE = 1 << 21


def _store(addr, *, event_id, program_seq, elem_size=4):
    return AccessEventRecord(
        op_type=Store,
        access_mode="write",
        addr_expr=addr,
        local_constraints=(),
        active=True,
        reads=False,
        writes=True,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def _force_unknown(monkeypatch):
    def _raise(solver, a, b):
        raise UnsupportedSymbolicRaceQuery(f"forced unknown for {a.name} vs {b.name}")

    monkeypatch.setattr(
        TwoCopySymbolicHBSolver, "_race_query_is_sat", staticmethod(_raise)
    )


def _disjoint_records():
    # store[DATA_BASE + 4*pid]: footprints disjoint across instances
    pid = SymbolicExpr.PID0
    return [_store(IntVal(DATA_BASE) + pid * 4, event_id=0, program_seq=0)]


def _colliding_records():
    # store[DATA_BASE] from every instance: cross-instance WAW
    return [_store(IntVal(DATA_BASE), event_id=0, program_seq=0)]


def test_enum_decides_race_free_at_the_extent(monkeypatch):
    _force_unknown(monkeypatch)
    solver = TwoCopySymbolicHBSolver(
        _disjoint_records(), grid=(4, 1, 1), enum_fallback_grid=(4, 1, 1)
    )
    assert solver.find_races() == []
    assert solver.enum_used


def test_enum_finds_the_race_with_in_extent_pids(monkeypatch):
    _force_unknown(monkeypatch)
    solver = TwoCopySymbolicHBSolver(
        _colliding_records(), grid=(4, 1, 1), enum_fallback_grid=(4, 1, 1)
    )
    reports = solver.find_races()
    assert solver.enum_used
    assert reports, "the collision must survive enumeration"
    rep = reports[0]
    for pid in (rep.witness_grid_a, rep.witness_grid_b):
        assert pid is not None
        assert 0 <= pid[0] < 4 and pid[1] == 0 and pid[2] == 0


def test_no_fallback_grid_reraises_the_unknown(monkeypatch):
    _force_unknown(monkeypatch)
    solver = TwoCopySymbolicHBSolver(_disjoint_records(), grid=(4, 1, 1))
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="forced unknown"):
        solver.find_races()
    assert not solver.enum_used


def test_case_cap_reraises_the_unknown(monkeypatch):
    _force_unknown(monkeypatch)
    # 100 instances -> 9900 ordered cross cases, far above ENUM_MAX_CASES
    solver = TwoCopySymbolicHBSolver(
        _disjoint_records(), grid=(100, 1, 1), enum_fallback_grid=(100, 1, 1)
    )
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="forced unknown"):
        solver.find_races()


def test_symbolic_decision_never_enumerates():
    solver = TwoCopySymbolicHBSolver(
        _disjoint_records(), grid=(4, 1, 1), enum_fallback_grid=(4, 1, 1)
    )
    assert solver.find_races() == []
    assert not solver.enum_used
