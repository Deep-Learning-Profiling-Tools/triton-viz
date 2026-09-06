"""Literal access modes retain activity and never hide conditional writes."""

import pytest
import z3

import triton_viz.clients.race_detector.two_copy_symbolic_hb_solver as tc

from .test_two_copy_symbolic_hb_solver import (
    _cas_record,
    _rmw_record,
    _scalar_load,
    _scalar_store,
)


@pytest.mark.parametrize("mode", [True, False, None])
@pytest.mark.parametrize("enabled", [True, False, None])
def test_mode_folding_is_equivalent_for_literal_and_symbolic_activity(mode, enabled):
    ctx = z3.Context()
    active = (
        z3.Bool("mode_active", ctx) if enabled is None else z3.BoolVal(enabled, ctx)
    )
    condition = (
        z3.Bool("mode_condition", ctx) if mode is None else z3.BoolVal(mode, ctx)
    )
    actual = tc.TwoCopySymbolicHBSolver._active_access_mode(active, condition)
    assert actual.ctx == ctx
    assert z3.is_false(z3.simplify(actual != z3.And(active, condition)))
    if mode is True:
        assert actual is active
    elif mode is False:
        assert actual is condition


def test_plain_lowering_keeps_masks_and_premises_with_conditional_modes():
    active, local, premise, condition = z3.Bools("mode_mask mode_local mode_prem mode")
    load = _scalar_load(z3.IntVal(100), event_id=0, program_seq=0, mask=local)
    store = _scalar_store(z3.IntVal(100), event_id=1, program_seq=1, mask=local)
    conditional = _scalar_store(z3.IntVal(100), event_id=2, program_seq=2, mask=local)
    load.reads = None
    store.writes = None
    conditional.writes = condition
    for record in (load, store, conditional):
        record.active = active
        record.premises = (premise,)
    core = tc.TwoCopySymbolicHBSolver([load, store, conditional], grid=(2, 1, 1))
    for event in core.events:
        assert z3.is_false(z3.simplify(event.active != z3.And(active, local, premise)))
        if event.record is load:
            assert event.reads is event.active
            assert z3.is_false(event.writes)
        else:
            assert z3.is_false(event.reads)
            expected = event.active
            if event.record is conditional:
                expected = z3.And(event.active, condition)
            assert event.writes.eq(expected)


def test_read_only_pairs_skip_mode_construction_for_cross_and_intra(monkeypatch):
    record = _scalar_load(z3.IntVal(100), event_id=0, mask=z3.Bool("read_mask"))
    core = tc.TwoCopySymbolicHBSolver([record], grid=(2, 1, 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("literal read-only pairs need no mode or address formula")

    monkeypatch.setattr(tc, "conflicting_access_modes", forbidden)
    monkeypatch.setattr(tc, "conflict_precheck_features", forbidden)
    monkeypatch.setattr(core, "_race_query_is_sat", forbidden)
    assert core._conflict_precheck(*core.events)
    assert core._conflict_precheck(
        *core.events, same_instance=True, lane_cond=z3.BoolVal(True)
    )
    assert core.find_races() == []


@pytest.mark.parametrize("writer_kind", ["conditional", "rmw", "cas"])
def test_read_only_shortcut_does_not_skip_possible_writes(writer_kind):
    load = _scalar_load(z3.IntVal(100), event_id=0, program_seq=0)
    if writer_kind == "conditional":
        writer = _scalar_store(z3.IntVal(100), event_id=1, program_seq=1)
        writer.writes = z3.Bool("conditional_write")
    elif writer_kind == "rmw":
        writer = _rmw_record(z3.IntVal(100), event_id=1, program_seq=1)
    else:
        writer = _cas_record(
            z3.IntVal(100),
            z3.IntVal(0),
            z3.IntVal(1),
            z3.Int("mode_cas_old"),
            event_id=1,
            program_seq=1,
        )
    writer.active = z3.Bool("writer_active")
    core = tc.TwoCopySymbolicHBSolver([load, writer], grid=(2, 1, 1))
    first, second = core.events[0], core.events[-1]
    assert not z3.is_false(second.writes)
    assert not core._conflict_precheck(first, second)
    assert core.find_races()


@pytest.mark.parametrize("scope,exempt", [("gpu", True), ("cta", False)])
def test_atomic_exemption_path_still_checks_scope(scope, exempt):
    record = _rmw_record(z3.IntVal(100), event_id=0, program_seq=0)
    record.scope = scope
    core = tc.TwoCopySymbolicHBSolver([record], grid=(2, 1, 1))
    assert core._conflict_precheck(*core.events) is exempt
    assert bool(core.find_races()) is not exempt
