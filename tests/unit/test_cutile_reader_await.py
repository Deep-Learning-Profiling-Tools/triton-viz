"""The await abstraction in the CuTile IR reader (spec C1, mirrored from the
TTIR reader's ``scf.while`` contract; ungated, every ladder level).

The cuda-tile 1.5.0 final IR prints a spin loop as a token-only
``loop (with $t: Token = ...)`` whose body polls one location, compares the
observation and ``break``s out of an ``if``; the reader collapses it to one
``awaited`` access carrying the EXIT predicate. Data-carrying while-form
loops keep their control-flow refusal byte for byte, and every clause of
the shape contract refuses with kind ``spin-shape``.
"""

import pytest

from triton_viz.clients.common.cutile_ir_reader import parse_cutile_ir
from triton_viz.clients.common.ttir_reader import (
    Cmp,
    Const,
    Not,
    Observed,
    UnsupportedTTIR,
)
from triton_viz.clients.race_detector.compiled.global_records import encode_graph

_HDR = """\
(flag_0: Tile[pointer[int32],()], flag_1: Tile[int32,()], flag_2: Tile[int32,()], data_0: Tile[pointer[int32],()], data_1: Tile[int32,()], data_2: Tile[int32,()]):
$token: Token = make_token()
$0: Tile[int32,()] = assume_bounded(x=flag_1, lower_bound=0, upper_bound=None)
$1: Tile[int32,()] = assume_bounded(x=data_1, lower_bound=0, upper_bound=None)
data{data_0, $1, data_2}: Array[int32,(?):(1)] = make_tensor_view(base_ptr=data_0, shape=($1), dynamic_strides=())
$12: Tile[int32,()] = tile_bid(axis=0)
$32: Tile[int32,(1)] = tile_arange()
$131: const Tile[int32,()] = typed_const(value=0)
$145: Tile[uint64,(1)] = tile_astype(x=$32)
$146: Tile[uint64,()] = tile_astype(x=$0)
$147: Tile[uint64,(1)] = tile_reshape(x=$146)
$148: Tile[bool_,(1)] = raw_cmp(lhs=$145, rhs=$147, fn="lt")
$149: Tile[pointer[int32],(1)] = tile_reshape(x=flag_0)
$150: Tile[pointer[int32],(1)] = pointer_offset(pointer=$149, offset=$145)
$151: Tile[int32,(1)] = tile_reshape(x=$131)
$160: const Tile[int32,()] = typed_const(value=1)
$161: const Tile[int32,()] = typed_const(value=7)
"""
_LOOP_OPEN = (
    "$token.9: Token, $token.13: Token = loop (with $token.8: Token = $token, $token.12: Token = $token)",
    "do ($token.8: Token, $token.12: Token)",
    "    ($token.8: Token, $token.12: Token):",
    "    $token.14: Token = join_tokens(tokens=($token.8, $token.12))",
)
_POLL = "    $152: Tile[int32,(1)], $153: Token = tile_atomic_rmw(pointer=$150, update={upd}, mask=$148, token=$token.14, mode=AtomicRMWMode.{mode}, memory_order=MemoryOrder.{order}, memory_scope=MemoryScope.DEVICE)"
_TEST = (
    "    $157: Tile[int32,()] = tile_reshape(x=$152)",
    '    $162: Tile[bool_,()] = raw_cmp(lhs=$157, rhs=$160, fn="ne")',
)
_EXIT_ELSE = (
    "    if(cond=$162)",
    "    then",
    "        ():",
    "        yield ",
    "    else",
    "        ():",
    "        break $153, $153",
    "    continue $153, $153",
)
_EXIT_THEN = (
    "    if(cond=$162)",
    "    then",
    "        ():",
    "        break $153, $153",
    "    else",
    "        ():",
    "        yield ",
    "    continue $153, $153",
)
_TAIL = (
    "$177: const Tile[int32,()] = typed_const(value=0)",
    "$190{data_0, $1, data_2}: PartitionView[Array[int32,(?):(1)],tile_shape=(64,),order=(0,),padding_mode=PaddingMode.UNDETERMINED] = make_partition_view(array=data{data_0, $1, data_2})",
    "$token.15: Token = join_tokens(tokens=($token, $token.13))",
    "$191: Tile[int32,(64)], $192: Token = tile_load(view=$190{data_0, $1, data_2}, index=($177), token=$token.15, latency=None, allow_tma=None, memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE)",
    "return",
)


def _ir(*body, poll=None, exit=_EXIT_ELSE, test=_TEST):
    poll = poll or _POLL.format(upd="$151", mode="ADD_INT", order="ACQUIRE")
    return _HDR + "\n".join([*_LOOP_OPEN, poll, *test, *body, *exit, *_TAIL]) + "\n"


@pytest.mark.parametrize("multipath", [False, True])
def test_spin_loop_becomes_one_awaited_access(multipath):
    g = parse_cutile_ir(_ir(), "t", multipath=multipath)
    assert [a.kind for a in g.accesses] == ["atomic_rmw", "load"]
    poll = g.accesses[0]
    assert poll.awaited and poll.atomic is not None
    assert poll.atomic.sem == "acquire" and poll.atomic.scope == "gpu"
    assert poll.atomic_val == Const(0)
    # continue while old != 1: the exit predicate is Not(old != 1)
    assert isinstance(poll.exit_pred, Not)
    cv = poll.exit_pred.a
    assert isinstance(cv, Cmp) and cv.pred == "ne"
    assert cv.a == Observed(0) and cv.b == Const(1)
    assert not g.accesses[1].awaited
    enc = encode_graph(g, {"flag_1": 1, "data_1": 64}, _tensors())
    assert enc.assumes_termination


def test_then_arm_break_keeps_the_condition_as_the_exit_predicate():
    g = parse_cutile_ir(_ir(exit=_EXIT_THEN), "t")
    poll = g.accesses[0]
    assert poll.awaited
    assert isinstance(poll.exit_pred, Cmp) and poll.exit_pred.a == Observed(0)


def test_plain_load_poll_is_the_await_shape_too():
    poll = "    $152: Tile[int32,(1)], $153: Token = load_pointer(pointer=$150, mask=$148, padding_value=$151, token=$token.14, latency=None)"
    g = parse_cutile_ir(_ir(poll=poll), "t")
    assert g.accesses[0].kind == "load" and g.accesses[0].awaited


def test_relaxed_spin_keeps_its_order():
    g = parse_cutile_ir(
        _ir(poll=_POLL.format(upd="$151", mode="ADD_INT", order="RELAXED")), "t"
    )
    assert g.accesses[0].awaited and g.accesses[0].atomic.sem == "relaxed"


def test_mutating_re_read_refuses():
    for upd, mode in (("$160", "ADD_INT"), ("$151", "EXCHANGE")):
        with pytest.raises(UnsupportedTTIR, match="MUTATES") as ei:
            parse_cutile_ir(
                _ir(poll=_POLL.format(upd=upd, mode=mode, order="ACQUIRE")), "t"
            )
        assert ei.value.kind == "spin-shape"


def test_store_inside_the_spin_refuses():
    store = "    $170: Token = store_pointer(pointer=$150, value=$151, mask=$148, token=$token.14, latency=None)"
    with pytest.raises(UnsupportedTTIR, match="store inside a spin loop") as ei:
        parse_cutile_ir(_ir(store), "t")
    assert ei.value.kind == "spin-shape"


def test_two_polls_refuse():
    second = "    $170: Tile[int32,(1)], $171: Token = tile_atomic_rmw(pointer=$150, update=$151, mask=$148, token=$153, mode=AtomicRMWMode.ADD_INT, memory_order=MemoryOrder.ACQUIRE, memory_scope=MemoryScope.DEVICE)"
    with pytest.raises(UnsupportedTTIR, match="exactly one location") as ei:
        parse_cutile_ir(_ir(second), "t")
    assert ei.value.kind == "spin-shape"


def test_exit_test_must_break_on_exactly_one_arm():
    both = tuple(ln.replace("yield ", "break $153, $153") for ln in _EXIT_ELSE)
    with pytest.raises(UnsupportedTTIR, match="exactly one arm") as ei:
        parse_cutile_ir(_ir(exit=both), "t")
    assert ei.value.kind == "spin-shape"


def test_exit_test_must_compare_the_observation_with_an_invariant():
    test = (
        "    $157: Tile[int32,()] = tile_reshape(x=$152)",
        '    $162: Tile[bool_,()] = raw_cmp(lhs=$157, rhs=$157, fn="ne")',
    )
    with pytest.raises(UnsupportedTTIR, match="loop-invariant") as ei:
        parse_cutile_ir(_ir(test=test), "t")
    assert ei.value.kind == "spin-shape"


def test_cas_poll_refuses_as_atomic_cas():
    poll = "    $152: Tile[int32,(1)], $153: Token = tile_atomic_cas(pointer=$150, expected=$151, desired=$160, mask=$148, token=$token.14, memory_order=MemoryOrder.ACQUIRE, memory_scope=MemoryScope.DEVICE)"
    with pytest.raises(UnsupportedTTIR) as ei:
        parse_cutile_ir(_ir(poll=poll), "t")
    assert ei.value.kind == "atomic-cas"


def test_data_carrying_while_form_keeps_the_control_flow_refusal():
    text = (
        _HDR
        + "\n".join(
            [
                "$20: Tile[int32,()] = loop (with k.0: Tile[int32,()] = $131)",
                "do (k.0: Tile[int32,()])",
                "    (k.0: Tile[int32,()]):",
                '    $21: Tile[bool_,()] = raw_cmp(lhs=k.0, rhs=$161, fn="lt")',
                "    if(cond=$21)",
                "    then",
                "        ():",
                "        yield ",
                "    else",
                "        ():",
                "        break k.0",
                "    continue k.0",
                "return",
            ]
        )
        + "\n"
    )
    for mp in (False, True):
        with pytest.raises(UnsupportedTTIR, match="while-form") as ei:
            parse_cutile_ir(text, "t", multipath=mp)
        assert ei.value.kind == "control-flow"


def _tensors():
    from triton_viz.clients.race_detector.compiled.global_records import GlobalTensor

    return {
        "flag": GlobalTensor(data_ptr=1 << 40, numel=1, elem_size=4, contiguous=True),
        "data": GlobalTensor(data_ptr=1 << 41, numel=64, elem_size=4, contiguous=True),
    }


# ── the benchmark's cuTile twins through the evaluation's cuTile track ──
# Every spin row of the litmus suite decides like its Triton twin at L2
# (their role split is an `if`, which the cuTile reader models only in
# multipath mode); the CAS-polling mutex rows refuse by name.

_SPIN_ROWS = {
    "trb016_pc_wait_no": "race-free",
    "trb016_pc_wait_or_poll_no": "race-free",
    "trb016_pc_wait_xor_poll_no": "race-free",
    "trb016_pc_wait_atomic_reset_no": "race-free",
    "trb016_pc_wait_cta_scope_yes": "race",
    "trb016_pc_wait_relaxed_spin_yes": "race",
    "trb016_pc_wait_cta_reset_yes": "race",
    "trb016_pc_wait_flag_read_yes": "race",
    "trb016_pc_wait_relaxed_writer_yes": "race",
    "trb018_lookback_no": "race-free",
    "trb018_lookback_cta_yes": "race",
    "trb025_comm_comp_no": "race-free",
    "trb025_relaxed_poll_yes": "race",
    "trb025_poll_initial_yes": "race",
    "trb025_role_skip_yes": "race",
}


@pytest.fixture(scope="module")
def _cutile_bench():
    from evaluation.kernels import load

    return {s.name: s for s in load("tritonracebench_cutile").specs}


@pytest.mark.parametrize("row", sorted(_SPIN_ROWS))
def test_benchmark_spin_twins_decide_like_their_triton_rows(row, _cutile_bench):
    from evaluation.harness import _static_track_cutile
    from triton_viz.clients.race_detector.ladder import LadderLevel

    spec = _cutile_bench[row]
    assert spec.expected == _SPIN_ROWS[row]
    res = _static_track_cutile(spec, 0, LadderLevel.L2)
    if _SPIN_ROWS[row] == "race-free":
        assert res["status"] == "ok", res["reason"]
        assert res["assumes_termination"] is True
        assert res["provenance"].endswith("+assumes-termination"), res["provenance"]
    else:
        assert res["status"] == "races", res["reason"]
        assert res["n_reports"] >= 1


@pytest.mark.parametrize(
    "row",
    [
        "trb017_mutex_cas_no",
        "trb017_mutex_plain_unlock_yes",
        "trb017_mutex_relaxed_cas_yes",
    ],
)
def test_benchmark_cas_spin_twins_refuse_by_name(row, _cutile_bench):
    from evaluation.harness import _static_track_cutile
    from triton_viz.clients.race_detector.ladder import LadderLevel

    res = _static_track_cutile(_cutile_bench[row], 0, LadderLevel.L2)
    assert res["status"] == "unsupported"
    assert res["reason"].startswith("atomic-cas:"), res["reason"]
