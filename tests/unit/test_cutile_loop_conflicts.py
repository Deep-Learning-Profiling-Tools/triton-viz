"""Deferred cuTile loop checks use verified allocation separation.

The same-instance solver shares loop indices. A loop summary therefore needs
both directions of cross-iteration token order, or proof that the unresolved
pair cannot conflict. These pins deliberately include hazards invisible when
the two accesses are restricted to the same iteration.
"""

from dataclasses import replace

import pytest

from triton_viz.clients.common.cutile_ir_reader import parse_cutile_ir
from triton_viz.clients.common.ttir_reader import UnsupportedTTIR
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
)

_HDR = """\
(x_0: Tile[pointer[int32],()], y_0: Tile[pointer[int32],()], n: Tile[int32,()]):
$root: Token = make_token()
$zero: const Tile[int32,()] = typed_const(value=0)
$one: const Tile[int32,()] = typed_const(value=1)
$yes: const Tile[bool_,()] = typed_const(value=True)
"""


def _parse(body, level=0):
    return parse_cutile_ir(
        _HDR + body + "\nreturn\n", "loop_conflicts", multipath=level >= 2
    )


def _mixed(*, level=0, load_base="x", read_shift="$one", mask="$yes"):
    # load x[k+1] and store y[k] have no same-iteration conflict even if
    # x and y alias. In different iterations their addresses can coincide.
    return _parse(
        f"""\
$done: Token = for $k in range($zero, n, $one) (with $carry: Token = $root)
do ($k: Tile[int32,()], $carry: Token)
    ($k: Tile[int32,()], $carry: Token):
    $next: Tile[int32,()] = raw_binary_arith(lhs=$k, rhs={read_shift}, fn="add", rounding_mode=None, flush_to_zero=False)
    $read_ptr: Tile[pointer[int32],()] = pointer_offset(pointer={load_base}_0, offset=$next)
    $read: Tile[int32,()], $loaded: Token = load_pointer(pointer=$read_ptr, mask={mask}, padding_value=$zero, token=$root, latency=None)
    $write_ptr: Tile[pointer[int32],()] = pointer_offset(pointer=y_0, offset=$k)
    $stored: Token = store_pointer(pointer=$write_ptr, value=$read, mask=$yes, token=$carry, latency=None)
    continue $stored
""",
        level,
    )


def _tensors(mode="separate"):
    x = GlobalTensor(data_ptr=4096, numel=16, elem_size=4)
    y = GlobalTensor(data_ptr=8192, numel=16, elem_size=4)
    if mode == "adjacent":
        y = replace(y, data_ptr=4160)
    elif mode == "partial_overlap":
        y = replace(y, data_ptr=4104)
    elif mode == "same":
        y = replace(x)
    elif mode == "shifted":
        # Different view pointers, the same allocation. The read in
        # iteration k+1 aliases the shifted store in iteration k.
        x = replace(x, storage_data_ptr=4096, storage_nbytes=128)
        y = GlobalTensor(
            data_ptr=4104,
            numel=16,
            elem_size=4,
            storage_data_ptr=4096,
            storage_nbytes=128,
        )
    elif mode == "disjoint_views_shared_storage":
        x = replace(x, numel=4, storage_data_ptr=4096, storage_nbytes=128)
        y = GlobalTensor(
            data_ptr=4160,
            numel=4,
            elem_size=4,
            storage_data_ptr=4096,
            storage_nbytes=128,
        )
    return {"x": x, "y": y}


def _verdict(graph, tensors, *, level=0, n=3):
    detector = CompiledRaceDetector(
        confirm_races=False, differential_check=False, ladder_level=level
    )
    result = detector.analyze_graph(graph, {"n": n}, tensors, (1, 1, 1))
    return result, detector


def _assert_token_abstention(result):
    assert result["verdict"] == "abstain", result
    assert result["unsupported_kind"] == "token-order", result


def test_mixed_loop_records_only_the_uncovered_pair_without_inventing_order():
    graph = _mixed()
    assert {(c.loop_ssa, c.first, c.second) for c in graph.loop_token_conflicts} == {
        ("$k", 0, 1)
    }
    assert graph.token_order == {}, "a loaded value does not create a token edge"


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("mode", ["separate", "adjacent"])
def test_separate_allocations_discharge_mixed_loop_obligation(level, mode):
    graph = _mixed(level=level)
    result, _ = _verdict(graph, _tensors(mode), level=level)
    assert result["verdict"] == "race-free", result


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("mode", ["same", "shifted", "partial_overlap"])
def test_actual_aliases_with_cross_iteration_hazards_refuse(level, mode):
    graph = _mixed(level=level)
    tensors = _tensors(mode)
    assert not CompiledRaceDetector._t0_premises_hold_for_launch(graph, tensors)
    result, _ = _verdict(graph, tensors, level=level)
    _assert_token_abstention(result)


def test_shared_allocation_with_disjoint_views_may_conservatively_refuse():
    # This bounded implementation proves allocation separation, not exact
    # footprint separation. Disjoint views do not justify skipping the check.
    result, _ = _verdict(_mixed(), _tensors("disjoint_views_shared_storage"))
    _assert_token_abstention(result)


@pytest.mark.parametrize(
    "bad_x",
    [
        GlobalTensor(data_ptr=4096, numel=16, elem_size=4, contiguous=False),
        GlobalTensor(data_ptr=4096, numel=16, elem_size=4, storage_data_ptr=4096),
        GlobalTensor(
            data_ptr=4096,
            numel=16,
            elem_size=4,
            storage_data_ptr=4096,
            storage_nbytes=4,
        ),
    ],
)
def test_unverified_allocation_bounds_cannot_discharge_an_obligation(bad_x):
    tensors = {**_tensors(), "x": bad_x}
    assert not CompiledRaceDetector._t0_premises_hold_for_launch(_mixed(), tensors)
    result, _ = _verdict(_mixed(), tensors)
    assert result["verdict"] == "abstain", result
    with pytest.raises(UnsupportedTTIR):
        encode_graph(_mixed(), {"n": 3}, tensors)


def test_t0_different_formals_are_conditional_on_the_existing_nonalias_gate():
    graph = _mixed()
    assert CompiledRaceDetector._t0_premises_hold_for_launch(graph, _tensors())
    assert not CompiledRaceDetector._t0_premises_hold_for_launch(
        graph, _tensors("shifted")
    )
    assert encode_graph_t0(graph), "the symbolic T0 case can use its nonalias premise"
    # Grouping must not erase an unresolved same-formal obligation.
    with pytest.raises(UnsupportedTTIR) as exc:
        encode_graph_t0(_mixed(load_base="y"))
    assert exc.value.kind == "token-order"


def _two_chains(*, one_direction=False, level=0):
    joined = (
        "    $joined: Token = join_tokens(tokens=($written_x, $written_y))\n"
        if one_direction
        else ""
    )
    continued_x = "$joined" if one_direction == "reverse" else "$written_x"
    continued_y = "$joined" if one_direction is True else "$written_y"
    return _parse(
        f"""\
$done_x: Token, $done_y: Token = for $k in range($zero, n, $one) (with $cx: Token = $root, $cy: Token = $root)
do ($k: Tile[int32,()], $cx: Token, $cy: Token)
    ($k: Tile[int32,()], $cx: Token, $cy: Token):
    $written_x: Token = store_pointer(pointer=x_0, value=$one, mask=$yes, token=$cx, latency=None)
    $written_y: Token = store_pointer(pointer=y_0, value=$one, mask=$yes, token=$cy, latency=None)
{joined}    continue {continued_x}, {continued_y}
""",
        level,
    )


@pytest.mark.parametrize("one_direction", [False, True, "reverse"])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_distinct_serial_chains_require_separation_in_both_directions(
    one_direction, level
):
    graph = _two_chains(one_direction=one_direction, level=level)
    assert {(c.first, c.second) for c in graph.loop_token_conflicts} == {(0, 1)}
    clean, _ = _verdict(graph, _tensors(), level=level)
    assert clean["verdict"] == "race-free", clean
    alias, _ = _verdict(graph, _tensors("same"), level=level)
    _assert_token_abstention(alias)


def _unchained_store(level=0):
    return _parse(
        """\
$done: Token = for $k in range($zero, n, $one) (with $carry: Token = $root)
do ($k: Tile[int32,()], $carry: Token)
    ($k: Tile[int32,()], $carry: Token):
    $stored: Token = store_pointer(pointer=y_0, value=$one, mask=$yes, token=$root, latency=None)
    $joined: Token = join_tokens(tokens=($carry, $stored))
    continue $joined
""",
        level,
    )


def test_store_self_pair_cannot_be_omitted_from_cross_iteration_checks():
    graph = _unchained_store()
    assert {(c.first, c.second) for c in graph.loop_token_conflicts} == {(0, 0)}
    result, _ = _verdict(graph, _tensors(), n=2)
    _assert_token_abstention(result)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("n", [0, 1])
def test_launch_zero_or_single_trip_discharges_cross_iteration_obligations(level, n):
    graph = _unchained_store(level)
    result, _ = _verdict(graph, _tensors(), level=level, n=n)
    assert result["verdict"] == "race-free", result


@pytest.mark.parametrize("level", [0, 1, 2])
def test_single_trip_still_reports_an_unordered_same_iteration_conflict(level):
    graph = _mixed(level=level, read_shift="$zero")
    result, _ = _verdict(graph, _tensors("same"), level=level, n=1)
    assert result["verdict"] == "race", result


def test_same_allocation_masked_access_does_not_get_an_unimplemented_footprint_proof():
    # The mask makes the load inactive. Keeping a conservative refusal is
    # permitted here: the new discharge deliberately uses allocation bounds.
    graph = _mixed(mask="$zero")
    result, _ = _verdict(graph, _tensors("same"))
    _assert_token_abstention(result)
