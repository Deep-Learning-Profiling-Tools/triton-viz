"""The addition represented by dot C retains D3; matrix inputs do not.

These are parser, production frontend, and shared solver regressions. They
do not assert a GPU lowering or make floating-point dot values symbolic.
"""

from types import SimpleNamespace

import pytest

from triton_viz.clients.common.ttir_reader import UnsupportedTTIR, parse_ttir
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.clients.sanitizer.compiled.oob import (
    LaunchContext,
    TensorMeta,
    check_graph,
)


_TYPE = "tensor<16x16xf32>"
_PTR = "tensor<16x16x!tt.ptr<f32>>"


def _dot(a="%one", b="%one", c="%c", suffix=", inputPrecision = tf32"):
    return f"tt.dot {a}, {b}, {c}{suffix} : {_TYPE} * {_TYPE} -> {_TYPE}"


def _module(*middle, result="%dot", shifted=False, load_mask="", store_mask=""):
    lines = [
        "%c16 = arith.constant 16 : i32",
        f"%zero = arith.constant dense<0.0> : {_TYPE}",
        f"%one = arith.constant dense<1.0> : {_TYPE}",
        "%false = arith.constant dense<false> : tensor<16x16xi1>",
        "%r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>",
        "%rx = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>",
        "%ry = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>",
        "%cs = tt.splat %c16 : i32 -> tensor<16x1xi32>",
        "%xm = arith.muli %rx, %cs : tensor<16x1xi32>",
        "%xb = tt.broadcast %xm : tensor<16x1xi32> -> tensor<16x16xi32>",
        "%yb = tt.broadcast %ry : tensor<1x16xi32> -> tensor<16x16xi32>",
        "%off = arith.addi %xb, %yb : tensor<16x16xi32>",
        "%limit = arith.constant dense<128> : tensor<16x16xi32>",
        "%active = arith.cmpi slt, %off, %limit : tensor<16x16xi32>",
        f"%base = tt.splat %c_ptr : !tt.ptr<f32> -> {_PTR}",
        f"%p = tt.addptr %base, %off : {_PTR}, tensor<16x16xi32>",
        f"%c = tt.load %p{load_mask} : {_PTR}",
        *middle,
    ]
    if shifted:
        lines.extend(
            [
                "%shift = arith.constant dense<1> : tensor<16x16xi32>",
                f"%ps = tt.addptr %p, %shift : {_PTR}, tensor<16x16xi32>",
            ]
        )
    pointer = "%ps" if shifted else "%p"
    lines.extend([f"tt.store {pointer}, {result}{store_mask} : {_PTR}", "tt.return"])
    body = "\n    ".join(lines)
    return (
        "module {\n"
        "  tt.func public @k(%c_ptr: !tt.ptr<f32>) attributes {noinline = false} {\n"
        f"    {body}\n"
        "  }\n}\n"
    )


def _solve(graph):
    encoding = encode_graph(
        graph, {}, {"c_ptr": GlobalTensor(data_ptr=0x10000, elem_size=4, numel=257)}
    )
    solver = TwoCopySymbolicHBSolver(
        encoding.records,
        grid=(1, 1, 1),
        arange_dict=encoding.arange_dict,
        fence_order=True,
        fence_seqs=tuple(graph.fences),
    )
    assert solver.check_feasibility(), "a vacuous fixture is not a regression"
    return solver.find_races()


@pytest.mark.parametrize("multipath", [False, True])
@pytest.mark.parametrize(
    "suffix",
    [
        "",
        ", inputPrecision = tf32",
        ", inputPrecision = tf32x3",
        ", inputPrecision = ieee {maxNumImpreciseAcc = 8 : i32}",
        " {maxNumImpreciseAcc = 8 : i32}",
    ],
)
def test_supported_dot_c_orders_same_position(suffix, multipath):
    graph = parse_ttir(_module("%dot = " + _dot(suffix=suffix)), multipath=multipath)
    assert graph.accesses[-1].deps == (0,)
    assert not _solve(graph)


@pytest.mark.parametrize(
    "dot",
    [
        _dot(a="%c", c="%zero"),
        _dot(b="%c", c="%zero"),
        _dot(a="%c", b="%c", c="%zero"),
    ],
)
def test_multiplicand_paths_cannot_order(dot):
    graph = parse_ttir(_module("%dot = " + dot))
    assert graph.accesses[-1].deps == ()
    assert [report.race_type.name for report in _solve(graph)] == ["WAR"]


@pytest.mark.parametrize("a,b", [("%c", "%one"), ("%one", "%c"), ("%c", "%c")])
def test_shared_multiplicand_source_does_not_erase_c_path(a, b):
    graph = parse_ttir(_module("%dot = " + _dot(a=a, b=b)))
    assert graph.accesses[-1].deps == (0,)
    assert not _solve(graph)


def test_c_path_does_not_revive_a_different_multiplicand_source():
    graph = parse_ttir(
        _module(
            f"%other = tt.load %p : {_PTR}",
            "%dot = " + _dot(a="%c", c="%other"),
        )
    )
    assert graph.accesses[-1].deps == (1,)
    assert [report.race_type.name for report in _solve(graph)] == ["WAR"]


@pytest.mark.parametrize(
    "transform",
    [
        f"tt.trans %c {{order = array<i32: 1, 0>}} : {_TYPE} -> {_TYPE}",
        f"tt.reshape %c : {_TYPE} -> {_TYPE}",
        f"tt.broadcast %c : {_TYPE} -> {_TYPE}",
        f"tt.unknown_transform %c : {_TYPE} -> {_TYPE}",
    ],
)
def test_c_does_not_revive_nonpositional_provenance(transform):
    graph = parse_ttir(_module("%ct = " + transform, "%dot = " + _dot(c="%ct")))
    assert graph.accesses[-1].deps == ()
    assert _solve(graph)


def test_elementwise_c_and_chained_c_dots_retain_provenance():
    graph = parse_ttir(
        _module(
            f"%ce = arith.mulf %c, %one : {_TYPE}",
            "%d1 = " + _dot(c="%ce"),
            "%dot = " + _dot(c="%d1"),
            f"%cast = arith.truncf %dot : {_TYPE} to tensor<16x16xf16>",
            result="%cast",
        )
    )
    assert graph.accesses[-1].deps == (0,)
    assert not _solve(graph)


def test_chained_dot_through_a_loses_c_provenance():
    graph = parse_ttir(_module("%d1 = " + _dot(), "%dot = " + _dot(a="%d1", c="%zero")))
    assert graph.accesses[-1].deps == ()
    assert _solve(graph)


def test_independent_dot_plus_add_matches_fused_c_dependency():
    graph = parse_ttir(
        _module(
            "%product = " + _dot(c="%zero"),
            f"%dot = arith.addf %product, %c : {_TYPE}",
        )
    )
    assert graph.accesses[-1].deps == (0,)
    assert not _solve(graph)


def test_c_does_not_invent_provenance_across_loop_iter_args():
    graph = parse_ttir(
        _module(
            "%lo = arith.constant 0 : i32",
            "%hi = arith.constant 1 : i32",
            f"%loop = scf.for %i = %lo to %hi step %hi iter_args(%acc = %c) -> ({_TYPE}) {{",
            "  %dot = " + _dot(c="%acc"),
            f"  scf.yield %dot : {_TYPE}",
            "}",
            result="%loop",
        )
    )
    assert graph.accesses[-1].deps == ()
    assert _solve(graph)


@pytest.mark.parametrize("mask", ["", ", %active"])
def test_shifted_store_still_races_at_distinct_positions(mask):
    graph = parse_ttir(
        _module("%dot = " + _dot(), shifted=True, load_mask=mask, store_mask=mask)
    )
    assert graph.accesses[-1].deps == (0,)
    assert [report.race_type.name for report in _solve(graph)] == ["WAR"]


@pytest.mark.parametrize(
    "load_mask,store_mask",
    [(", %false, %zero", ""), ("", ", %false"), (", %active", ", %active")],
)
def test_load_and_store_activity_is_preserved(load_mask, store_mask):
    graph = parse_ttir(
        _module("%dot = " + _dot(), load_mask=load_mask, store_mask=store_mask)
    )
    assert graph.accesses[-1].deps == (0,)
    assert not _solve(graph)


@pytest.mark.parametrize(
    "dot",
    [
        _dot().replace("tt.dot ", "tt.dot_scaled "),
        f"tt.dot_scaled %one scale %c, %one, %zero lhs = e4m3 rhs = e4m3 : {_TYPE}, {_TYPE} * {_TYPE} -> {_TYPE}",
        _dot().replace("tt.dot ", "tt.dot_unknown "),
        _dot().replace("%c, inputPrecision", "%c, %one, inputPrecision"),
        _dot().replace("inputPrecision = tf32", "unknownAttribute = 1"),
        _dot(suffix=" {unknownAttribute = 1 : i32}"),
        f'"tt.dot"(%one, %one, %c) : ({_TYPE}, {_TYPE}, {_TYPE}) -> {_TYPE}',
    ],
)
def test_unrecognized_dot_forms_remain_nonpositional(dot):
    graph = parse_ttir(_module("%dot = " + dot))
    assert graph.accesses[-1].deps == ()
    assert _solve(graph)


def test_dot_value_remains_unknown_for_masks_and_addresses():
    text = _module(
        "%dot = " + _dot(),
        f"%mask = arith.cmpf ogt, %dot, %zero : {_TYPE}",
        store_mask=", %mask",
    )
    graph = parse_ttir(text)
    assert graph.accesses[-1].deps == (0,)
    assert graph.accesses[-1].mask is None
    assert graph.accesses[-1].mask_dropped
    text = text.replace(
        "tt.store %p, %dot, %mask",
        f"%bad = tt.addptr %p, %dot : {_PTR}, tensor<16x16xi32>\n"
        "    tt.store %bad, %dot, %mask",
    )
    with pytest.raises(UnsupportedTTIR, match="data-dependent"):
        parse_ttir(text)


def test_compiled_client_and_shared_sanitizer_consume_the_same_graph():
    text = _module("%dot = " + _dot())
    detector = CompiledRaceDetector()
    detector.post_warmup_callback(None, SimpleNamespace(asm={"ttir": text}))
    detector.finalize()
    (graph,) = detector.last_ttir_graphs
    assert graph is not None
    assert detector.last_ttir_unsupported == [None]
    assert graph.accesses[-1].deps == (0,)
    assert not _solve(graph)
    context = LaunchContext(
        grid=(1, 1, 1),
        params={},
        tensors={
            "c_ptr": TensorMeta(
                numel=257, elem_bits=32, data_ptr=0x10000, contiguous=True
            )
        },
    )
    assert check_graph(graph, context) == []
