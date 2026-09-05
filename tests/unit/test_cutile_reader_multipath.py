"""Multipath mode of the CuTile IR reader (Route 3, the ladder's L2).

``parse_cutile_ir(ir, name, multipath=True)`` lifts the single-path
model's two structural refusals, several ``for`` loops and structured
``if`` blocks, and is byte-identical otherwise. Snippets follow the
cuda-tile 1.5.0 final-IR grammar the captured corpus prints (``for $i in
range(a, b, c) (with )`` / ``do`` / body / ``continue``; ``if(cond=$c)``
/ ``then`` / ``():`` / body / ``yield ...`` or ``return`` / ``else``).
"""

from dataclasses import replace

import pytest

from triton_viz.clients.common.cutile_ir_reader import parse_cutile_ir
from triton_viz.clients.common.ttir_reader import (
    Cmp,
    Const,
    LoopVar,
    Not,
    Param,
    Pid,
    Select,
    UnsupportedTTIR,
)
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

_HDR = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()], n: Tile[int32,()], m: Tile[int32,()]):
$token: Token = make_token()
$0: Tile[int32,()] = assume_bounded(x=x_1, lower_bound=0, upper_bound=None)
x{x_0, $0, x_2}: Array[float32,(?):(1)] = make_tensor_view(base_ptr=x_0, shape=($0), dynamic_strides=())
$1: Tile[int32,()] = tile_bid(axis=0)
$2{x_0, $0, x_2}: PartitionView[Array[float32,(?):(1)],tile_shape=(64,),order=(0,),padding_mode=PaddingMode.UNDETERMINED] = make_partition_view(array=x{x_0, $0, x_2})
$3: Tile[float32,(64)] = typed_const(value=0)
$c0: const Tile[int32,()] = typed_const(value=0)
$c1: const Tile[int32,()] = typed_const(value=1)
"""
_ARITH = 'raw_binary_arith(lhs={a}, rhs={b}, fn="{fn}", rounding_mode=None, flush_to_zero=False)'
_STORE = "tile_store(view=$2{{x_0, $0, x_2}}, index=({idx}), tile=$3, token=$token, latency=None, allow_tma=None, memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE)"


def _ir(*body):
    return _HDR + "\n".join(body) + "\nreturn\n"


def _mp(text):
    return parse_cutile_ir(text, "t", multipath=True)


def _t1(g, params, numel=1 << 14, grid=(4, 1, 1)):
    tensors = {
        "x": GlobalTensor(data_ptr=1 << 40, numel=numel, elem_size=4, contiguous=True)
    }
    enc = encode_graph(g, {"x_1": numel, "x_2": 1, **params}, tensors, multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, grid),
        arange_dict=enc.arange_dict,
        enum_fallback_grid=grid,
    )
    return solver.find_races()


def _pids(rep):
    return rep.witness_grid_a[0], rep.witness_grid_b[0]


# ───────────────────── fixtures ─────────────────────

SEQ_LOOPS = _ir(
    "for $i in range($c0, n, $c1) (with )",
    "do ($i: Tile[int32,()])",
    "    ($i: Tile[int32,()]):",
    "    $5: Tile[int32,()] = " + _ARITH.format(a="$1", b="n", fn="mul"),
    "    $6: Tile[int32,()] = " + _ARITH.format(a="$5", b="$i", fn="add"),
    "    $7: Token = " + _STORE.format(idx="$6"),
    "    continue ",
    "for $j in range($c0, m, $c1) (with )",
    "do ($j: Tile[int32,()])",
    "    ($j: Tile[int32,()]):",
    "    $8: Token = " + _STORE.format(idx="$j"),
    "    continue ",
)

NESTED_LOOPS = _ir(
    "for $i in range($c0, n, $c1) (with )",
    "do ($i: Tile[int32,()])",
    "    ($i: Tile[int32,()]):",
    "    $5: Tile[int32,()] = " + _ARITH.format(a="$1", b="n", fn="mul"),
    "    $6: Tile[int32,()] = " + _ARITH.format(a="$5", b="$i", fn="add"),
    "    $7: Tile[int32,()] = " + _ARITH.format(a="$6", b="m", fn="mul"),
    "    for $j in range($c0, m, $c1) (with )",
    "    do ($j: Tile[int32,()])",
    "        ($j: Tile[int32,()]):",
    "        $8: Tile[int32,()] = " + _ARITH.format(a="$7", b="$j", fn="add"),
    "        $9: Token = " + _STORE.format(idx="$8"),
    "        continue ",
    "    continue ",
)

SINGLE_LOOP = _ir(
    "for $i in range($c0, n, $c1) (with )",
    "do ($i: Tile[int32,()])",
    "    ($i: Tile[int32,()]):",
    "    $5: Tile[int32,()] = " + _ARITH.format(a="$1", b="n", fn="mul"),
    "    $6: Tile[int32,()] = " + _ARITH.format(a="$5", b="$i", fn="add"),
    "    $7: Token = " + _STORE.format(idx="$6"),
    "    continue ",
)


def _if_expr(then_val):
    return _ir(
        '$8: Tile[bool_,()] = raw_cmp(lhs=$1, rhs=$c0, fn="eq")',
        "$9: Tile[int32,()] = if(cond=$8)",
        "then",
        "    ():",
        f"    yield {then_val}",
        "else",
        "    ():",
        "    yield $1",
        "$10: Token = " + _STORE.format(idx="$9"),
    )


GUARD = _ir(
    '$8: Tile[bool_,()] = raw_cmp(lhs=$1, rhs=n, fn="ge")',
    "if(cond=$8)",
    "then",
    "    ():",
    "    return",
    "else",
    "    ():",
    "    yield ",
    "$10: Token = " + _STORE.format(idx="$c0"),
)

ARMS = _ir(
    '$8: Tile[bool_,()] = raw_cmp(lhs=$1, rhs=$c0, fn="eq")',
    "if(cond=$8)",
    "then",
    "    ():",
    "    $10: Token = " + _STORE.format(idx="$c0"),
    "    yield ",
    "else",
    "    ():",
    "    $11: Token = " + _STORE.format(idx="$1"),
    "    yield ",
)

LOADED_GUARD = _ir(
    "$u: Tile[int32,()] = mystery_op(x=$1)",
    '$8: Tile[bool_,()] = raw_cmp(lhs=$u, rhs=$c0, fn="eq")',
    "if(cond=$8)",
    "then",
    "    ():",
    "    return",
    "else",
    "    ():",
    "    yield ",
    "$10: Token = " + _STORE.format(idx="$1"),
)

WHILE_LOOP = _ir(
    "$20: Tile[int32,()] = loop (with k.0: Tile[int32,()] = $c0)",
    "do (k.0: Tile[int32,()])",
    "    (k.0: Tile[int32,()]):",
    '    $21: Tile[bool_,()] = raw_cmp(lhs=k.0, rhs=n, fn="lt")',
    "    if(cond=$21)",
    "    then",
    "        ():",
    "        yield ",
    "    else",
    "        ():",
    "        break k.0",
    "    $22: Token = " + _STORE.format(idx="k.0"),
    "    $23: Tile[int32,()] = " + _ARITH.format(a="k.0", b="$c1", fn="add"),
    "    continue $23",
)


# ───────────────────── single-path unchanged ─────────────────────


@pytest.mark.parametrize(
    "text, kind, needle",
    [
        (SEQ_LOOPS, "nested-loop", "multiple/nested loops"),
        (NESTED_LOOPS, "nested-loop", "multiple/nested loops"),
        (_if_expr("$c0"), "control-flow", "`if` block structure"),
        (GUARD, "control-flow", "unrecognized statement"),  # the pre-change message
        (WHILE_LOOP, "control-flow", "while-form"),
    ],
)
def test_single_path_refusals_unchanged(text, kind, needle):
    with pytest.raises(UnsupportedTTIR, match=needle) as ei:
        parse_cutile_ir(text, "t")
    assert ei.value.kind == kind


def test_single_loop_parses_identically_in_both_modes():
    a = parse_cutile_ir(SINGLE_LOOP, "t")
    b = _mp(SINGLE_LOOP)
    assert not a.multipath and b.multipath
    assert a.loop is not None and a.loops == []
    # the only differences are the multipath bookkeeping fields
    assert replace(a, multipath=True, loops=[a.loop]) == replace(
        b, accesses=[replace(x, loops=()) for x in b.accesses]
    )
    assert b.accesses[0].loops == ("$i",)


# ───────────────────── loops ─────────────────────


def test_sequential_loops_get_their_own_iterators():
    g = _mp(SEQ_LOOPS)
    assert [lp.loop_ssa for lp in g.loops] == ["$i", "$j"]
    assert g.loop is None
    first, second = g.accesses
    assert first.loops == ("$i",) and second.loops == ("$j",)
    assert g.loops[0].upper == Param("n") and g.loops[1].upper == Param("m")


def test_nested_loops_outer_first():
    g = _mp(NESTED_LOOPS)
    assert [lp.loop_ssa for lp in g.loops] == ["$i", "$j"]
    (store,) = g.accesses
    assert store.loops == ("$i", "$j") and store.in_loop


def test_nested_loops_prove_disjoint_tiles_and_report_the_shared_ones():
    # tile (pid*n + i)*m + j: disjoint across pids for every n, m
    assert _t1(_mp(NESTED_LOOPS), {"n": 2, "m": 3}) == []
    # the second sequential loop writes tiles 0..m-1 from every pid
    reports = _t1(_mp(SEQ_LOOPS), {"n": 2, "m": 3})
    assert reports and all(a != b for a, b in map(_pids, reports))


def test_yield_or_break_inside_a_for_body_refuses():
    text = _ir(
        "for $i in range($c0, n, $c1) (with )",
        "do ($i: Tile[int32,()])",
        "    ($i: Tile[int32,()]):",
        "    $7: Token = " + _STORE.format(idx="$i"),
        "    break ",
    )
    with pytest.raises(UnsupportedTTIR, match="inside a `for` body") as ei:
        _mp(text)
    assert ei.value.kind == "control-flow"


def test_while_form_loop_still_refused_at_l2():
    with pytest.raises(UnsupportedTTIR, match="while-form") as ei:
        _mp(WHILE_LOOP)
    assert ei.value.kind == "control-flow"


# ───────────────────── if blocks ─────────────────────

PID_IS_ZERO = Cmp("eq", Pid(0), Const(0))


def test_if_expression_binds_a_select_over_the_yields():
    g = _mp(_if_expr("$c0"))
    (store,) = g.accesses
    assert store.path is None and not store.guarded
    # offset = (Select(pid == 0, 0, pid) * 64 + arange) * stride
    sel = store.offset.a.a.a
    assert sel == Select(PID_IS_ZERO, Const(0), Pid(0))
    assert _t1(g, {"n": 1, "m": 1}) == []
    # then-arm yielding 1 makes pid 0 collide with pid 1
    reports = _t1(_mp(_if_expr("$c1")), {"n": 1, "m": 1})
    assert reports and all(set(_pids(r)) == {0, 1} for r in reports)


def test_early_return_guard_becomes_the_continuation_predicate():
    g = _mp(GUARD)
    (store,) = g.accesses
    assert store.path == Not(Cmp("sge", Pid(0), Param("n"))) and not store.guarded
    # every pid < n writes tile 0: one survivor proves, two race
    assert _t1(g, {"n": 1, "m": 1}) == []
    reports = _t1(g, {"n": 2, "m": 1})
    assert reports and all(max(_pids(r)) < 2 for r in reports)


def test_both_arms_carry_their_conditions():
    g = _mp(ARMS)
    then_store, else_store = g.accesses
    assert then_store.path == PID_IS_ZERO and else_store.path == Not(PID_IS_ZERO)
    assert not then_store.guarded and not else_store.guarded
    # pid 0 writes tile 0 in the then-arm; pid k >= 1 writes tile k: disjoint
    assert _t1(g, {"n": 1, "m": 1}) == []


def test_unmodelable_condition_widens_the_continuation():
    g = _mp(LOADED_GUARD)
    (store,) = g.accesses
    assert store.guarded and store.path is None
    enc = encode_graph(
        g,
        {"x_1": 1 << 14, "x_2": 1, "n": 1, "m": 1},
        {
            "x": GlobalTensor(
                data_ptr=1 << 40, numel=1 << 14, elem_size=4, contiguous=True
            )
        },
        multipath=True,
    )
    assert enc.uncertain_event_ids == {0}


def test_if_inside_a_loop_body_conjoins_the_iteration_and_the_condition():
    text = _ir(
        "for $i in range($c0, n, $c1) (with )",
        "do ($i: Tile[int32,()])",
        "    ($i: Tile[int32,()]):",
        '    $8: Tile[bool_,()] = raw_cmp(lhs=$i, rhs=$c0, fn="eq")',
        "    if(cond=$8)",
        "    then",
        "        ():",
        "        $10: Token = " + _STORE.format(idx="$1"),
        "        yield ",
        "    else",
        "        ():",
        "        yield ",
        "    continue ",
    )
    g = _mp(text)
    (store,) = g.accesses
    assert store.loops == ("$i",)
    assert store.path == Cmp("eq", LoopVar("$i"), Const(0))
    assert _t1(g, {"n": 3, "m": 1}) == []
