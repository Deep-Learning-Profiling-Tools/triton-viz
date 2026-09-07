"""A simultaneous positional path must survive a second permuted path."""

import pytest

from triton_viz.clients.common.ttir_reader import parse_ttir
from .test_t1_rmw_static import _module


def _graph(combine):
    return parse_ttir(
        _module(
            "%x: !tt.ptr<f32>, %cond: i1",
            "%r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>",
            "%xs = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>",
            "%p = tt.addptr %xs, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>",
            "%v = tt.load %p : tensor<4x!tt.ptr<f32>>",
            "%reshaped = tt.reshape %v : tensor<4xf32> -> tensor<2x2xf32>",
            "%transposed = tt.trans %reshaped {order = array<i32: 1, 0>} : tensor<2x2xf32> -> tensor<2x2xf32>",
            "%rotated = tt.reshape %transposed : tensor<2x2xf32> -> tensor<4xf32>",
            "%cs = tt.splat %cond : i1 -> tensor<4xi1>",
            "%zero = arith.constant dense<0.0> : tensor<4xf32>",
            "%loaded_cond = arith.cmpf ogt, %v, %zero : tensor<4xf32>",
            combine,
            "tt.store %p, %out : tensor<4x!tt.ptr<f32>>",
        )
    )


@pytest.mark.parametrize("operands", ["%v, %rotated", "%rotated, %v"])
def test_simultaneous_direct_and_permuted_paths_keep_direct_dependency(operands):
    graph = _graph(f"%out = arith.addf {operands} : tensor<4xf32>")
    assert graph.accesses[-1].deps == (0,)


def test_pure_permutation_does_not_gain_positional_dependency():
    graph = _graph("%out = arith.addf %rotated, %rotated : tensor<4xf32>")
    assert graph.accesses[-1].deps == ()


@pytest.mark.parametrize("arms", ["%v, %rotated", "%rotated, %v"])
def test_select_cannot_borrow_inactive_direct_arm(arms):
    graph = _graph(f"%out = arith.select %cs, {arms} : tensor<4xi1>, tensor<4xf32>")
    assert graph.accesses[-1].deps == ()


@pytest.mark.parametrize("arms", ["%v, %zero", "%zero, %v"])
def test_select_cannot_borrow_dependency_absent_from_other_arm(arms):
    graph = _graph(f"%out = arith.select %cs, {arms} : tensor<4xi1>, tensor<4xf32>")
    assert graph.accesses[-1].deps == ()


def test_select_keeps_dependency_present_in_both_arms():
    graph = _graph("%out = arith.select %cs, %v, %v : tensor<4xi1>, tensor<4xf32>")
    assert graph.accesses[-1].deps == (0,)


def test_select_condition_is_an_unconditional_positional_operand():
    graph = _graph(
        "%out = arith.select %loaded_cond, %rotated, %zero : tensor<4xi1>, tensor<4xf32>"
    )
    assert graph.accesses[-1].deps == (0,)


def test_unrecognized_select_syntax_cannot_invent_dependencies():
    graph = _graph("%out = arith.select %cs,  %v, %zero : tensor<4xi1>, tensor<4xf32>")
    assert graph.accesses[-1].deps == ()


@pytest.mark.parametrize(
    "operation,racy",
    [
        ("arith.addf %v, %rotated : tensor<4xf32>", False),
        ("arith.addf %rotated, %rotated : tensor<4xf32>", True),
        ("arith.select %cs, %v, %rotated : tensor<4xi1>, tensor<4xf32>", True),
    ],
)
def test_dependency_reaches_solver_without_hiding_permuted_conflicts(operation, racy):
    from triton_viz.clients.race_detector.compiled.global_records import (
        GlobalTensor,
        encode_graph,
    )
    from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
        TwoCopySymbolicHBSolver,
    )

    graph = _graph(f"%out = {operation}")
    encoded = encode_graph(
        graph, {"cond": 1}, {"x": GlobalTensor(data_ptr=4096, elem_size=4, numel=4)}
    )
    solver = TwoCopySymbolicHBSolver(
        encoded.records, grid=(1,), arange_dict=encoded.arange_dict, fence_order=True
    )
    assert bool(solver.find_races()) is racy
