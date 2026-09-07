"""cuTile token order is an SSA partial order, including control-flow results.

The loop pins protect the serial-iteration proof required by the solver's
single-iteration same-instance queries. Unsupported recurrences refuse rather
than manufacturing program-order edges or silently ignoring other iterations.
"""

import pytest

from triton_viz.clients.common.cutile_ir_reader import parse_cutile_ir
from triton_viz.clients.common.ttir_reader import Cmp, Const, Param, UnsupportedTTIR
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

_HDR = """\
(x_0: Tile[pointer[int32],()], n: Tile[int32,()], gate: Tile[int32,()]):
$root: Token = make_token()
$zero: const Tile[int32,()] = typed_const(value=0)
$one: const Tile[int32,()] = typed_const(value=1)
$two: const Tile[int32,()] = typed_const(value=2)
$yes: const Tile[bool_,()] = typed_const(value=True)
$no: const Tile[bool_,()] = typed_const(value=False)
$cond: Tile[bool_,()] = raw_cmp(lhs=gate, rhs=$one, fn="eq")
"""


def _store(name, token, mask="$yes"):
    return f"{name}: Token = store_pointer(pointer=x_0, value=$one, mask={mask}, token={token}, latency=None)"


def _load(name, token):
    return f"{name}.value: Tile[int32,()], {name}: Token = load_pointer(pointer=x_0, mask=$yes, padding_value=$zero, token={token}, latency=None)"


def _graph(*body, multipath=False):
    return parse_cutile_ir(
        _HDR + "\n".join(body) + "\nreturn\n", "t", multipath=multipath
    )


def _races(g, **params):
    enc = encode_graph(
        g,
        {"n": 2, "gate": 1, **params},
        {"x": GlobalTensor(data_ptr=1 << 40, numel=1, elem_size=4, contiguous=True)},
        multipath=g.multipath,
    )
    return TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (1, 1, 1)),
        arange_dict=enc.arange_dict,
        fence_order=True,
        fence_seqs=enc.fence_seqs,
        token_order=enc.token_order,
    ).find_races()


def test_empty_order_is_explicit_and_independent_same_instance_stores_race():
    g = _graph(_store("$a", "$root"), _store("$b", "$root"))
    assert g.token_order == {} and g.fences == []
    assert _races(g)


def test_chain_and_join_preserve_partial_order_without_a_full_cut():
    g = _graph(
        _store("$a", "$root"),
        _store("$b", "$root"),
        "$joined: Token = join_tokens(tokens=($a, $b))",
        _store("$c", "$joined"),
    )
    assert g.token_order == {(0, 2): None, (1, 2): None}
    assert _races(g), "the two independent predecessors still conflict"
    chain = _graph(_store("$a", "$root"), _store("$b", "$a"), _store("$c", "$b"))
    assert chain.token_order == {(0, 1): None, (0, 2): None, (1, 2): None}
    assert _races(chain) == []


def test_masked_intermediate_operation_still_transmits_token_ancestry():
    g = _graph(_store("$a", "$root"), _store("$b", "$a", "$no"), _store("$c", "$b"))
    assert g.token_order[(0, 2)] is None
    assert _races(g) == []


def _phi(then_token="$a", else_token="$root"):
    return (
        "$selected: Token = if(cond=$cond)",
        "then",
        "    ():",
        f"    yield {then_token}",
        "else",
        "    ():",
        f"    yield {else_token}",
    )


def test_branch_phi_dependency_is_guarded_by_the_actual_selected_operand():
    g = _graph(
        _store("$a", "$root"), *_phi(), _store("$b", "$selected"), multipath=True
    )
    assert g.token_order == {(0, 1): Cmp("eq", Param("gate"), Const(1))}
    assert _races(g, gate=1) == []
    assert _races(g, gate=0)


def test_phi_of_same_token_on_both_arms_has_unconditional_ancestry():
    g = _graph(
        _store("$a", "$root"),
        *_phi("$a", "$a"),
        _store("$b", "$selected"),
        multipath=True,
    )
    assert g.token_order == {(0, 1): None}
    assert _races(g, gate=0) == []


@pytest.mark.parametrize(
    "condition",
    [
        "$cond: Tile[bool_,()] = mystery_op(x=gate)",
        '$lane: Tile[int32,(2)] = tile_arange()\n$cond: Tile[bool_,(2)] = raw_cmp(lhs=$lane, rhs=$one, fn="eq")',
    ],
)
def test_unknown_or_lane_token_phi_refuses(condition):
    with pytest.raises(UnsupportedTTIR) as exc:
        _graph(condition, _store("$a", "$root"), *_phi(), multipath=True)
    assert exc.value.kind == "token-order"


def test_branch_local_token_cannot_escape_without_a_result():
    with pytest.raises(UnsupportedTTIR) as exc:
        _graph(
            "if(cond=$cond)",
            "then",
            "    ():",
            "    " + _store("$a", "$root"),
            "    yield ",
            "else",
            "    ():",
            "    yield ",
            _store("$b", "$a"),
            multipath=True,
        )
    assert exc.value.kind == "token-order"


@pytest.mark.parametrize(
    "body",
    [
        (_store("$a", "$missing"),),
        (_store("$a", "$root").replace(", token=$root", ""),),
        ("$mystery: Token = unknown_token_op(x=$root)",),
        ("$mystery: Tile[int32,()] = unknown_token_op(tokens=($root))",),
        ("$mystery: Tile[int32,()] = unknown_token_op(operands=([$root]))",),
        ("$joined: Token = join_tokens(tokens=($root, $missing))",),
    ],
)
def test_missing_or_unsupported_token_shapes_refuse(body):
    with pytest.raises(UnsupportedTTIR) as exc:
        _graph(*body)
    assert exc.value.kind == "token-order"


def _loop(body, *, upper="n", initial="$root", continued="$a"):
    return (
        f"$done: Token = for $i in range($zero, {upper}, $one) (with $carry: Token = {initial})",
        "do ($i: Tile[int32,()], $carry: Token)",
        "    ($i: Tile[int32,()], $carry: Token):",
        *("    " + line for line in body),
        f"    continue {continued}",
    )


def test_serial_loop_preserves_entry_and_exit_dependencies_with_zero_trip_bypass():
    g = _graph(
        _store("$before", "$root"),
        *_loop([_store("$a", "$carry")], initial="$before"),
        _store("$after", "$done"),
    )
    assert g.token_order[(0, 2)] is None
    assert g.token_order[(1, 2)] == Cmp("slt", Const(0), Param("n"))
    assert _races(g, n=0) == []
    assert _races(g, n=2) == []


def test_statically_zero_trip_skips_dead_unsupported_token_body():
    g = _graph(
        _store("$before", "$root"),
        *_loop(
            ["$a: Token = unknown_token_op(x=$carry)"], upper="$zero", initial="$before"
        ),
        _store("$after", "$done"),
    )
    assert len(g.accesses) == 2 and g.token_order == {(0, 1): None}
    assert _races(g) == []


def test_one_trip_needs_no_recurrence_but_parallel_body_stores_still_race():
    g = _graph(
        *_loop(
            [_store("$a", "$root"), _store("$b", "$root")],
            upper="$one",
            continued="$root",
        )
    )
    assert g.token_order == {} and _races(g)


@pytest.mark.parametrize("upper", ["n", "$two"])
def test_independent_write_iterations_refuse(upper):
    with pytest.raises(UnsupportedTTIR, match="serial memory boundary") as exc:
        _graph(
            *_loop(
                [
                    _store("$a", "$root"),
                    "$joined: Token = join_tokens(tokens=($carry, $a))",
                ],
                upper=upper,
                continued="$joined",
            )
        )
    assert exc.value.kind == "token-order"


def test_parallel_stores_joined_at_a_serial_iteration_boundary_remain_unordered_inside():
    g = _graph(
        *_loop(
            [
                _store("$a", "$carry"),
                _store("$b", "$carry"),
                "$joined: Token = join_tokens(tokens=($a, $b))",
            ],
            continued="$joined",
        )
    )
    assert g.token_order == {} and _races(g)


def test_readonly_for_can_have_independent_iterations():
    g = _graph(*_loop([_load("$a", "$root")], continued="$carry"))
    assert g.token_order == {} and _races(g) == []


def test_readonly_loop_cannot_reset_its_carried_token():
    with pytest.raises(UnsupportedTTIR, match="drops or swaps") as exc:
        _graph(*_loop([_load("$a", "$root")], continued="$root"))
    assert exc.value.kind == "token-order"


def test_loop_results_preserve_corresponding_slots():
    g = _graph(
        _store("$before", "$root"),
        "$first: Token, $second: Token = for $i in range($zero, n, $one) (with $c1: Token = $before, $c2: Token = $root)",
        "do ($i: Tile[int32,()], $c1: Token, $c2: Token)",
        "    ($i: Tile[int32,()], $c1: Token, $c2: Token):",
        "    " + _load("$a", "$c1"),
        "    continue $a, $c2",
        _store("$after", "$second"),
    )
    assert (0, 2) not in g.token_order and (1, 2) not in g.token_order
    assert _races(g)
