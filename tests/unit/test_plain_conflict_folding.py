"""Local formula folding must preserve arbitrary activities and HB edges."""

from types import SimpleNamespace

import pytest
import z3

from triton_viz.clients.race_detector.hb_common import conflicting_access_modes
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)


def _event(index, mode, *, atomic=False, width=4, scope="gpu"):
    active = z3.Bool(f"fold_active_{index}")
    conditional = z3.Bool(f"fold_write_{index}")
    return SimpleNamespace(
        idx=index,
        active=active,
        reads=active if mode in ("read", "rmw", "conditional") else z3.BoolVal(False),
        writes=(
            z3.And(active, conditional)
            if mode == "conditional"
            else active
            if mode in ("write", "rmw")
            else z3.BoolVal(False)
        ),
        addr=z3.Int(f"fold_addr_{index}"),
        elem_size=width,
        is_atomic=atomic,
        scope=scope,
        pid=tuple(z3.Int(f"fold_pid_{index}_{axis}") for axis in range(3)),
    )


@pytest.mark.parametrize("first_mode", ["read", "write", "conditional", "rmw", "none"])
@pytest.mark.parametrize("second_mode", ["read", "write", "conditional", "rmw", "none"])
@pytest.mark.parametrize("edges", ["false", "true", "conditional"])
def test_folded_race_formula_is_equivalent_without_extra_assumptions(
    first_mode, second_mode, edges
):
    first, second = _event(0, first_mode), _event(1, second_mode, width=2)
    forward = (
        z3.Bool("fold_forward")
        if edges == "conditional"
        else z3.BoolVal(edges == "true")
    )
    reverse = z3.Bool("fold_reverse") if edges == "conditional" else z3.BoolVal(False)
    core = object.__new__(TwoCopySymbolicHBSolver)
    core.hb = [[z3.BoolVal(False), forward], [reverse, z3.BoolVal(False)]]
    reference = z3.And(
        first.active,
        second.active,
        core._byte_overlap(first, second),
        conflicting_access_modes(first, second),
        z3.Not(forward),
        z3.Not(reverse),
    )
    solver = z3.Solver()
    solver.add(reference != core._race_expr(first, second))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("scopes", [("gpu", "gpu"), ("cta", "gpu"), ("cta", "cta")])
@pytest.mark.parametrize("width", [2, 4])
def test_atomic_scope_and_partial_byte_exemptions_keep_general_formula(scopes, width):
    first = _event(0, "rmw", atomic=True, scope=scopes[0])
    second = _event(1, "conditional", atomic=True, width=width, scope=scopes[1])
    core = object.__new__(TwoCopySymbolicHBSolver)
    assert core._ordinary_access_mode(first) is None
    assert core._ordinary_access_mode(second) is None
    reference = z3.And(
        first.active,
        second.active,
        core._byte_overlap(first, second),
        conflicting_access_modes(first, second),
    )
    solver = z3.Solver()
    solver.add(reference != core._conflict(first, second))
    assert solver.check() == z3.unsat
