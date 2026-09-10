"""Counter scope failures retry the complete, feasible launch domain."""

from dataclasses import replace

import pytest
from z3 import BoolVal, Int

from triton_viz.clients.common.ttir_reader import parse_ttir
from triton_viz.clients.race_detector.compiled import client as client_module
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.hb_common import (
    CountingScopeUnsupported,
    UnsupportedSymbolicRaceQuery,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_t1_rmw_static import _module


def _batch_ttir(
    *,
    increment=2,
    masked=False,
    foreign_write=False,
    shared_store=False,
    grid_gate=None,
    duplicate_lanes=False,
):
    before = [
        "%true = arith.constant true",
        "%zero = arith.constant 0 : i32",
        f"%inc = arith.constant {increment} : i32",
        "%pid = tt.get_program_id x : i32",
    ]
    mask = "%true"
    if masked:
        before.append("%first = arith.cmpi eq, %pid, %zero : i32")
        mask = "%first"
    if foreign_write:
        before.append("tt.store %head_ptr, %zero : !tt.ptr<i32>")
    after = []
    if shared_store or grid_gate is not None:
        mask_suffix = ""
        if grid_gate is not None:
            after.extend(
                [
                    "%np = tt.get_num_programs x : i32",
                    "%four = arith.constant 4 : i32",
                    f"%grid_gate = arith.cmpi {grid_gate}, %np, %four : i32",
                ]
            )
            mask_suffix = ", %grid_gate"
        after.append(f"tt.store %buf_ptr, %pid{mask_suffix} : !tt.ptr<i32>")
    lane = "%lane"
    lane_lines = []
    if duplicate_lanes:
        lane_lines = [
            "%two = arith.constant dense<2> : tensor<2xi32>",
            "%same_lane = arith.divsi %lane, %two : tensor<2xi32>",
        ]
        lane = "%same_lane"
    return _module(
        "%head_ptr: !tt.ptr<i32>, %buf_ptr: !tt.ptr<i32>",
        *before,
        f"%old = tt.atomic_rmw add, relaxed, gpu, %head_ptr, %inc, {mask} : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%lane = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>",
        *lane_lines,
        "%ticket = tt.splat %old : i32 -> tensor<2xi32>",
        f"%idx = arith.addi %ticket, {lane} : tensor<2xi32>",
        "%base = tt.splat %buf_ptr : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>",
        "%ptrs = tt.addptr %base, %idx : tensor<2x!tt.ptr<i32>>, tensor<2xi32>",
        "%values = tt.splat %pid : i32 -> tensor<2xi32>",
        "tt.store %ptrs, %values : tensor<2x!tt.ptr<i32>>",
        *after,
    )


def _tensors(init=0):
    return {
        "head_ptr": GlobalTensor(
            data_ptr=0x2000,
            elem_size=4,
            numel=1,
            init_values=None if init is None else (init,),
        ),
        "buf_ptr": GlobalTensor(data_ptr=0x10000, elem_size=4, numel=64),
    }


def _analyze(text=None, *, init=0, grid=(4,)):
    graph = parse_ttir(text or _batch_ttir(), multipath=True)
    detector = CompiledRaceDetector(confirm_races=False, ladder_level="L2")
    verdict = detector.analyze_graph(graph, {}, _tensors(init), grid)
    return detector, verdict


def test_generalized_counter_scope_is_distinct_from_actual_overflow():
    enc = encode_graph(parse_ttir(_batch_ttir()), {}, _tensors())
    with pytest.raises(CountingScopeUnsupported):
        TwoCopySymbolicHBSolver(
            enc.records, grid=symbolic_grid(enc, (4,)), arange_dict=enc.arange_dict
        )
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=(4, 1, 1), arange_dict=enc.arange_dict
    )
    assert len(solver._counting) == 1
    assert solver.find_races() == []
    assert solver.check_feasibility()
    # The same increment really wraps at this larger fixed extent.
    with pytest.raises(UnsupportedSymbolicRaceQuery) as exc:
        TwoCopySymbolicHBSolver(
            enc.records, grid=(2**30, 1, 1), arange_dict=enc.arange_dict
        )
    assert not isinstance(exc.value, CountingScopeUnsupported)


def test_batch_ticket_gets_launch_proof_without_grid_fragility():
    detector, verdict = _analyze()
    assert detector.last_global_status == "ok"
    assert detector.last_global_provenance == "proved@T1-launch"
    assert verdict["proved_scope"] == "this-params-this-grid"
    assert not verdict["grid_fragile"]
    assert detector.last_grid_fragile == []


@pytest.mark.parametrize("init,grid", [(2**31 - 1, (4,)), (0, (2**30,))])
def test_fixed_launch_wraparound_still_abstains(init, grid):
    detector, verdict = _analyze(init=init, grid=grid)
    assert detector.last_global_status == "unsupported"
    assert verdict["verdict"] == "abstain"
    assert verdict["proved_scope"] is None


@pytest.mark.parametrize(
    "change", ["masked", "foreign_write", "unknown_init", "zero_increment"]
)
def test_fixed_launch_does_not_bypass_other_counting_guards(change):
    options = {change: True} if change in {"masked", "foreign_write"} else {}
    if change == "zero_increment":
        options["increment"] = 0
    detector, verdict = _analyze(
        _batch_ttir(**options), init=None if change == "unknown_init" else 0
    )
    assert detector.last_global_status == "unsupported"
    assert verdict["verdict"] == "abstain"


def test_without_a_launch_domain_counter_rejection_stays_unsupported():
    detector, verdict = _analyze(grid=None)
    assert detector.last_global_status == "unsupported"
    assert verdict["proved_scope"] is None


@pytest.mark.parametrize("mutation", ["shared_store", "duplicate_lanes"])
def test_retry_checks_cross_instance_and_duplicate_lane_conflicts(mutation):
    detector, verdict = _analyze(_batch_ttir(**{mutation: True}))
    assert detector.last_global_status == "races"
    assert verdict["verdict"] == "race"
    assert detector.last_global_reports
    if mutation == "duplicate_lanes":
        assert any(
            r.witness_grid_a == r.witness_grid_b for r in detector.last_global_reports
        )


@pytest.mark.parametrize("comparison,expected", [("eq", "race"), ("ne", "race-free")])
def test_retry_pins_num_programs_in_existing_records(comparison, expected):
    detector, verdict = _analyze(_batch_ttir(grid_gate=comparison))
    assert verdict["verdict"] == expected
    if expected == "race-free":
        assert verdict["proved_scope"] == "this-params-this-grid"


@pytest.mark.parametrize("failure", ["infeasible", "unknown"])
def test_retry_cannot_certify_infeasible_or_unknown_base(monkeypatch, failure):
    original = client_module.encode_graph
    if failure == "infeasible":

        def contradictory(*args, **kwargs):
            enc = original(*args, **kwargs)
            return replace(
                enc, assumptions=tuple(enc.assumptions) + (Int("grid_0") != 4,)
            )

        monkeypatch.setattr(client_module, "encode_graph", contradictory)
    else:

        def unknown(self, extra=()):
            raise UnsupportedSymbolicRaceQuery("test feasibility unknown")

        monkeypatch.setattr(TwoCopySymbolicHBSolver, "check_feasibility", unknown)
    detector, verdict = _analyze()
    assert detector.last_global_status == (
        "vacuous" if failure == "infeasible" else "unsupported"
    )
    assert verdict["verdict"] == "abstain"
    assert verdict["proved_scope"] is None


def test_retry_retains_original_assumptions_and_content_qualification(monkeypatch):
    original = client_module.encode_graph

    def qualified(*args, **kwargs):
        enc = original(*args, **kwargs)
        return replace(
            enc,
            assumptions=tuple(enc.assumptions) + (BoolVal(True),),
            content_qualified=True,
        )

    monkeypatch.setattr(client_module, "encode_graph", qualified)
    detector, verdict = _analyze()
    assert detector.last_global_status == "ok"
    assert verdict["proved_scope"] == "this-params-this-grid"
    assert verdict["content_qualified"]
    assert "+content" in detector.last_global_provenance
