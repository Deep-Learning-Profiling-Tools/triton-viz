"""Pins for the CuTile IR reader front-end (clients/common/cutile_ir_reader).

Self-contained IR snippets (grammar-faithful to cuda-tile 1.5.0's final
CuTile IR text) exercise the semantic mapping: tile-space addressing to
affine terms with implicit-clip masks, the raw-pointer gather/scatter and
atomic paths, the boolean-xor floor-division lowering, the counted form
of the while-form `loop` construct, and the abstention discipline for
integer xor and for every other while-form loop. End-to-end pins run the
parsed graph through encode_graph and the two-copy solver — proof AND
detection directions.
"""

import pytest

from triton_viz.clients.common.cutile_ir_reader import parse_cutile_ir
from triton_viz.clients.common.ttir_reader import Const, UnsupportedTTIR
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

_STORE_IR = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()]):
$token: Token = make_token()
$0: Tile[int32,()] = assume_bounded(x=x_1, lower_bound=0, upper_bound=None)
x{x_0, $0, x_2}: Array[float32,(?):(1)] = make_tensor_view(base_ptr=x_0, shape=($0), dynamic_strides=())
$1: Tile[int32,()] = tile_bid(axis=0)
$2{x_0, $0, x_2}: PartitionView[Array[float32,(?):(1)],tile_shape=(64,),order=(0,),padding_mode=PaddingMode.UNDETERMINED] = make_partition_view(array=x{x_0, $0, x_2})
$3: Tile[float32,(64)] = typed_const(value=0)
$4: Token = tile_store(view=$2{x_0, $0, x_2}, index=(INDEX), tile=$3, token=$token, latency=None, allow_tma=None, memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE)
return
"""


def _solve(ir: str, n: int = 256, grid: tuple = (4, 1, 1)):
    g = parse_cutile_ir(ir, "t")
    params = {"x_1": n, "x_2": 1}
    tensors = {
        "x": GlobalTensor(data_ptr=1 << 40, numel=n, elem_size=4, contiguous=True)
    }
    enc = encode_graph(g, params, tensors)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, grid),
        arange_dict=enc.arange_dict,
        fence_order=True,
        fence_seqs=enc.fence_seqs,
        token_order=enc.token_order,
    )
    return g, solver.find_races()


def test_tile_store_partitioned_by_bid_proves_clean():
    g, found = _solve(_STORE_IR.replace("INDEX", "$1"))
    assert [a.kind for a in g.accesses] == ["store"]
    assert g.accesses[0].base_param == "x"
    assert g.accesses[0].mask is not None, "implicit OOB clip must be a mask"
    assert g.pid_axes == {0}
    assert found == []


def test_fresh_client_solves_a_cutile_graph_without_finalize():
    """The evaluation's cuTile track drives ``_solve_one_graph`` on a fresh
    client (cuda.tile has no interpreter, so the launch callbacks and
    finalize() never run). Regression (2026-09-05, after Route 2): the
    "+content" qualifier was initialized only in finalize(), so every
    solved cuTile row died with an AttributeError and refused."""
    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    det = CompiledRaceDetector(confirm_races=False, differential_check=False)
    assert det.last_global_content_qualified is False
    g = parse_cutile_ir(_STORE_IR.replace("INDEX", "$1"), "t")
    tensors = {
        "x": GlobalTensor(data_ptr=1 << 40, numel=256, elem_size=4, contiguous=True)
    }
    outcome = det._solve_one_graph(g, {"x_1": 256, "x_2": 1}, tensors, (4, 1, 1))
    assert outcome[0] == "proved", outcome
    assert det.last_global_content_qualified is False
    # the public entry: the same outcome settled into a verdict, with the
    # attributes stamped, on a client that never saw a launch callback
    det2 = CompiledRaceDetector(confirm_races=False, differential_check=False)
    v = det2.analyze_graph(g, {"x_1": 256, "x_2": 1}, tensors, (4, 1, 1))
    assert det2.last_global_status == "ok"
    assert det2.last_global_provenance in ("proved@T0", "proved@T1")
    assert v["verdict"] == "race-free" and v["ladder_level"] == "L0"
    assert v["conditional"] == () and v["content_qualified"] is False
    # a parse refusal settles as an abstention with the reason and kind
    v3 = CompiledRaceDetector(confirm_races=False).analyze_graph(
        None, {}, {}, (1, 1, 1), parse_reason="nested-loop: line 3: nested"
    )
    assert v3["verdict"] == "abstain" and v3["unsupported_kind"] == "nested-loop"


def test_tile_store_constant_index_races():
    # every program writes tile 0 — the detection direction must fire
    _, found = _solve(_STORE_IR.replace("INDEX", "0"))
    assert len(found) == 1
    assert found[0].race_type.name == "WAW"


_ATOMIC_IR = """\
(h_0: Tile[pointer[int32],()], h_1: Tile[int32,()], h_2: Tile[int32,()]):
$token: Token = make_token()
$b: Tile[int32,()] = tile_bid(axis=0)
$ar: Tile[int32,(64)] = tile_arange()
$c64: const Tile[int32,()] = typed_const(value=64)
$off0: Tile[int32,(64)] = raw_binary_arith(lhs=$b, rhs=$c64, fn="mul", rounding_mode=None, flush_to_zero=False)
$off: Tile[int32,(64)] = raw_binary_arith(lhs=$off0, rhs=$ar, fn="add", rounding_mode=None, flush_to_zero=False)
$m: Tile[bool_,(64)] = raw_cmp(lhs=$off, rhs=h_1, fn="lt")
$pr: Tile[pointer[int32],(1)] = tile_reshape(x=h_0)
$pb: Tile[pointer[int32],(64)] = tile_broadcast(x=$pr)
$p: Tile[pointer[int32],(64)] = pointer_offset(pointer=$pb, offset=$off)
$one: const Tile[int32,()] = typed_const(value=1)
$oneb: Tile[int32,(64)] = tile_broadcast(x=$one)
$old: Tile[int32,(64)], $tk: Token = tile_atomic_rmw(pointer=$p, update=$oneb, mask=$m, token=$token, mode=AtomicRMWMode.ADD_INT, memory_order=MemoryOrder.ACQ_REL, memory_scope=MemoryScope.DEVICE)
return
"""


def test_atomic_rmw_via_pointer_offset():
    g = parse_cutile_ir(_ATOMIC_IR, "t")
    (ev,) = g.accesses
    assert ev.kind == "atomic_rmw"
    assert ev.base_param == "h"
    assert ev.atomic is not None and ev.atomic.rmw_op == "add"
    assert ev.atomic.sem == "acq_rel" and ev.atomic.scope == "gpu"
    assert ev.mask is not None and not ev.mask_dropped
    assert ev.atomic_val == Const(1)


_FLOOR_FIX_IR = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()]):
$token: Token = make_token()
$ar: Tile[int32,(64)] = tile_arange()
$C: const Tile[int32,()] = typed_const(value=8)
$r: Tile[int32,(64)] = raw_binary_arith(lhs=$ar, rhs=$C, fn="c_mod", rounding_mode=None, flush_to_zero=False)
$z: const Tile[int32,()] = typed_const(value=0)
$s1: Tile[bool_,(64)] = raw_cmp(lhs=$r, rhs=$z, fn="lt")
$s2: Tile[bool_,()] = raw_cmp(lhs=$C, rhs=$z, fn="lt")
$x: Tile[bool_,(64)] = raw_binary_bitwise(lhs=$s1, rhs=$s2, fn="xor")
$n0: Tile[bool_,(64)] = raw_cmp(lhs=$r, rhs=$z, fn="ne")
$fx: Tile[bool_,(64)] = raw_binary_bitwise(lhs=$x, rhs=$n0, fn="and_")
$rc: Tile[int32,(64)] = raw_binary_arith(lhs=$r, rhs=$C, fn="add", rounding_mode=None, flush_to_zero=False)
$rr: Tile[int32,(64)] = raw_where(cond=$fx, x=$rc, y=$r)
$pr: Tile[pointer[float32],(1)] = tile_reshape(x=x_0)
$pb: Tile[pointer[float32],(64)] = tile_broadcast(x=$pr)
$p: Tile[pointer[float32],(64)] = pointer_offset(pointer=$pb, offset=$rr)
$v: Tile[float32,(64)] = typed_const(value=0)
$m: Tile[bool_,(64)] = raw_cmp(lhs=$rr, rhs=x_1, fn="lt")
$st: Token = store_pointer(pointer=$p, value=$v, mask=$m, token=$token, latency=None)
return
"""


def test_bool_xor_floor_division_lowering_stays_modeled():
    # python floor-mod lowers to c_mod + a sign-fix select whose
    # disagreement test is a BOOLEAN xor — the whole chain must stay a
    # modeled term (no mask_dropped, no indirect-address abstention)
    g = parse_cutile_ir(_FLOOR_FIX_IR, "t")
    (ev,) = g.accesses
    assert ev.kind == "store"
    assert not ev.mask_dropped
    assert ev.mask is not None


def test_integer_xor_in_address_abstains():
    ir = _FLOOR_FIX_IR.replace(
        "$rr: Tile[int32,(64)] = raw_where(cond=$fx, x=$rc, y=$r)",
        '$rr: Tile[int32,(64)] = raw_binary_bitwise(lhs=$r, rhs=$rc, fn="xor")',
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(ir, "t")
    assert exc.value.kind == "indirect-address"


def test_load_pointer_records_read_event():
    ir = _FLOOR_FIX_IR.replace(
        "$st: Token = store_pointer(pointer=$p, value=$v, mask=$m, token=$token, latency=None)",
        "$ld: Tile[float32,(64)], $lt: Token = load_pointer(pointer=$p, mask=$m, padding_value=$v, token=$token, latency=None)",
    )
    g = parse_cutile_ir(ir, "t")
    (ev,) = g.accesses
    assert ev.kind == "load" and ev.base_param == "x"


def test_while_form_loop_abstains_as_control_flow():
    ir = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()]):
$token: Token = make_token()
$z: Tile[float32,()] = typed_const(value=0)
$a: Tile[float32,()] = loop (with acc.0: Tile[float32,()] = $z)
return
"""
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(ir, "t")
    assert exc.value.kind == "control-flow"


# ── exact lowering of integer bitwise addressing ─────────────────
# cuda.tile emits xor/shift/mask directly in an address (the bitonic and
# radix networks). The reader lowers them into the affine fragment ONLY
# when the second operand is a known integer, and a lowering that consumed
# a scalar param's captured value marks the graph param_pinned so the tier
# selector never claims the ANY-params (T0) scope from it.

_BITWISE_IR = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()], j: Tile[int32,()]):
$token: Token = make_token()
$ar: Tile[int32,(64)] = tile_arange()
$jr: Tile[int32,(1)] = tile_reshape(x=j)
$jb: Tile[int32,(64)] = tile_broadcast(x=$jr)
$xo: Tile[int32,(64)] = raw_binary_bitwise(lhs=$ar, rhs=$jb, fn="OP")
$pr: Tile[pointer[float32],(1)] = tile_reshape(x=x_0)
$pb: Tile[pointer[float32],(64)] = tile_broadcast(x=$pr)
$p: Tile[pointer[float32],(64)] = pointer_offset(pointer=$pb, offset=$xo)
$v: Tile[float32,(64)] = typed_const(value=0)
$st: Token = store_pointer(pointer=$p, value=$v, mask=None, token=$token, latency=None)
return
"""


def _eval_term(term, lane, params):
    """Interpret an address term the way the encoder does: Bin // and % are
    C-style truncation toward zero (remsi)."""
    from triton_viz.clients.common.ttir_reader import Arange, Bin, Const, Param

    if isinstance(term, Const):
        return term.value
    if isinstance(term, Param):
        return params[term.name]
    if isinstance(term, Arange):
        return term.start + lane
    if isinstance(term, Bin):
        a, b = _eval_term(term.a, lane, params), _eval_term(term.b, lane, params)
        if term.op == "+":
            return a + b
        if term.op == "-":
            return a - b
        if term.op == "*":
            return a * b
        quotient = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
        return quotient if term.op == "//" else a - b * quotient
    raise AssertionError(f"unexpected term {term!r}")


@pytest.mark.parametrize("op,value", [("xor", 4), ("and_", 7), ("rshift", 3)])
def test_integer_bitwise_address_lowers_exactly_from_a_pinned_param(op, value):
    line = 'raw_binary_bitwise(lhs=$ar, rhs=$jb, fn="OP")'
    if op == "rshift":
        line = 'raw_bitwise_shift(lhs=$ar, rhs=$jb, fn="OP")'
    ir = _BITWISE_IR.replace(
        'raw_binary_bitwise(lhs=$ar, rhs=$jb, fn="OP")', line
    ).replace("OP", op)
    g = parse_cutile_ir(ir, "t", params={"j": value})
    (ev,) = g.accesses
    assert g.param_pinned, "a launch value went into the address"
    native = {
        "xor": lambda i: i ^ value,
        "and_": lambda i: i & value,
        "rshift": lambda i: i >> value,
    }[op]
    for lane in range(64):
        assert _eval_term(ev.offset, lane, {"j": value}) == native(lane), lane


def test_integer_bitwise_address_without_captured_params_abstains():
    # T0 has no launch: the operand stays symbolic and the refusal stands.
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(_BITWISE_IR.replace("OP", "xor"), "t")
    assert exc.value.kind == "indirect-address"


def test_non_power_of_two_operands_keep_abstaining():
    for op, value in (("xor", 6), ("and_", 6)):
        with pytest.raises(UnsupportedTTIR) as exc:
            parse_cutile_ir(_BITWISE_IR.replace("OP", op), "t", params={"j": value})
        assert exc.value.kind == "indirect-address"


def test_literal_mask_lowers_without_pinning_the_graph():
    # A literal needs no launch, so the ANY-params tier stays available.
    ir = _BITWISE_IR.replace(
        "$jb: Tile[int32,(64)] = tile_broadcast(x=$jr)",
        "$jc: const Tile[int32,()] = typed_const(value=7)\n"
        "$jb: Tile[int32,(64)] = tile_broadcast(x=$jc)",
    ).replace("OP", "and_")
    g = parse_cutile_ir(ir, "t")
    (ev,) = g.accesses
    assert not g.param_pinned
    for lane in range(64):
        assert _eval_term(ev.offset, lane, {}) == lane & 7


def test_t0_gate_refuses_a_param_pinned_graph():
    from triton_viz.clients.race_detector.compiled.global_records import (
        t0_linearity_gate,
    )

    g = parse_cutile_ir(_BITWISE_IR.replace("OP", "xor"), "t", params={"j": 4})
    assert g.param_pinned and not t0_linearity_gate(g)


# ── the counted form of the while-form `loop` construct ──────────
# cuda.tile lowers a python `while i < N:` over an integer counter into a
# `loop` whose body opens with the test and exits through it. That exact
# shape carries the same trip count as `for i in range(i0, N, step)`, so
# the reader lifts it into the one LoopInfo slot; every other while-form
# loop keeps the control-flow refusal.

_LOOP_PREAMBLE = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()], N: Tile[int32,()]):
$token: Token = make_token()
$0: Tile[int32,()] = assume_bounded(x=x_1, lower_bound=0, upper_bound=None)
x{x_0, $0, x_2}: Array[float32,(?):(1)] = make_tensor_view(base_ptr=x_0, shape=($0), dynamic_strides=())
$1: Tile[int32,()] = tile_bid(axis=0)
$2{x_0, $0, x_2}: PartitionView[Array[float32,(?):(1)],tile_shape=(64,),order=(0,),padding_mode=PaddingMode.UNDETERMINED] = make_partition_view(array=x{x_0, $0, x_2})
$3: Tile[float32,(64)] = typed_const(value=0)
$i0: const Tile[int32,()] = typed_const(value=0)
$one: const Tile[int32,()] = typed_const(value=1)
"""

# The body is written once and shared by both loop forms, so a graph
# difference can only come from the loop lift itself. The store chains the
# carried token, which is what gives the loop its serial memory boundary
# (a store in a token-free loop body abstains in both forms alike).
_LOOP_BODY = """\
    $m1: Tile[int32,()] = raw_binary_arith(lhs=$1, rhs=N, fn="mul", rounding_mode=None, flush_to_zero=False)
    $idx: Tile[int32,()] = raw_binary_arith(lhs=$m1, rhs=i.0, fn="add", rounding_mode=None, flush_to_zero=False)
    $sto: Token = tile_store(view=$2{x_0, $0, x_2}, index=($idx), tile=$3, token=$tok.0, latency=None, allow_tma=None, memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE)
    $an: Tile[float32,(64)] = raw_binary_arith(lhs=acc.0, rhs=$3, fn="add", rounding_mode=None, flush_to_zero=False)
"""

_COUNTED_WHILE_IR = (
    _LOOP_PREAMBLE
    + """\
$out: Tile[float32,(64)], $iout: Tile[int32,()], $tok.1: Token = loop (with acc.0: Tile[float32,(64)] = $3, i.0: Tile[int32,()] = $i0, $tok.0: Token = $token)
do (acc.0: Tile[float32,(64)], i.0: Tile[int32,()], $tok.0: Token)
    (acc.0: Tile[float32,(64)], i.0: Tile[int32,()], $tok.0: Token):
    $c: Tile[bool_,()] = raw_cmp(lhs=i.0, rhs=N, fn="lt")
    if(cond=$c)
    then
        ():
        yield
    else
        ():
        break acc.0, i.0, $tok.0
"""
    + _LOOP_BODY
    + """\
    $in: Tile[int32,()] = raw_binary_arith(lhs=i.0, rhs=$one, fn="add", rounding_mode=None, flush_to_zero=False)
    continue $an, $in, $sto
return
"""
)

_COUNTED_FOR_IR = (
    _LOOP_PREAMBLE
    + """\
$out: Tile[float32,(64)], $tok.1: Token = for i.0 in range($i0, N, $one) (with acc.0: Tile[float32,(64)] = $3, $tok.0: Token = $token)
do (i.0: Tile[int32,()], acc.0: Tile[float32,(64)], $tok.0: Token)
    (i.0: Tile[int32,()], acc.0: Tile[float32,(64)], $tok.0: Token):
"""
    + _LOOP_BODY
    + """\
    continue $an, $sto
return
"""
)


def _solve_loop(ir: str, n: int = 512, grid: tuple = (4, 1, 1), bound: int = 2):
    g = parse_cutile_ir(ir, "t", multipath=True)
    params = {"x_1": n, "x_2": 1, "N": bound}
    tensors = {
        "x": GlobalTensor(data_ptr=1 << 40, numel=n, elem_size=4, contiguous=True)
    }
    enc = encode_graph(g, params, tensors)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, grid),
        arange_dict=enc.arange_dict,
        fence_order=True,
        fence_seqs=enc.fence_seqs,
        token_order=enc.token_order,
    )
    return g, solver.find_races()


def test_counted_while_lifts_to_the_range_of_its_test():
    from triton_viz.clients.common.ttir_reader import Param

    g = parse_cutile_ir(_COUNTED_WHILE_IR, "t", multipath=True)
    (loop,) = g.loops
    assert loop.loop_ssa == "i.0" and loop.induction_var == "i.0"
    assert loop.lower == Const(0)
    assert loop.upper == Param("N")
    assert loop.step == Const(1)
    (ev,) = g.accesses
    assert ev.kind == "store" and ev.loops == ("i.0",)


def test_counted_while_and_the_equivalent_for_build_the_same_graph():
    w = parse_cutile_ir(_COUNTED_WHILE_IR, "t", multipath=True)
    f = parse_cutile_ir(_COUNTED_FOR_IR, "t", multipath=True)
    assert [(a.kind, a.base_param, a.offset, a.mask, a.loops) for a in w.accesses] == [
        (a.kind, a.base_param, a.offset, a.mask, a.loops) for a in f.accesses
    ]
    assert [(lp.loop_ssa, lp.lower, lp.upper, lp.step) for lp in w.loops] == [
        (lp.loop_ssa, lp.lower, lp.upper, lp.step) for lp in f.loops
    ]


def test_counted_while_partitioned_by_bid_proves_clean():
    g, found = _solve_loop(_COUNTED_WHILE_IR)
    assert [a.kind for a in g.accesses] == ["store"]
    assert found == []


def test_counted_while_writing_the_same_tiles_still_reports_the_race():
    # the negative control for the lift: dropping the bid from the tile
    # index makes every program write tiles 0..N-1, and the loop lift must
    # not launder that into a proof
    racy = _COUNTED_WHILE_IR.replace("index=($idx)", "index=(i.0)")
    _g, found = _solve_loop(racy)
    assert found, "a counted while that write-shares its tiles must race"


def test_counted_while_with_a_second_exit_keeps_refusing():
    # an early `break` inside the body means the trip count is no longer
    # the test's range: the zero-trip rule would DELETE accesses that a
    # real early exit still performs
    early = _COUNTED_WHILE_IR.replace(
        "    $in: Tile[int32,()] = raw_binary_arith(lhs=i.0, rhs=$one",
        "    if(cond=$c)\n"
        "    then\n"
        "        ():\n"
        "        break acc.0, i.0, $tok.0\n"
        "    else\n"
        "        ():\n"
        "        yield \n"
        "    $in: Tile[int32,()] = raw_binary_arith(lhs=i.0, rhs=$one",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(early, "t", multipath=True)
    assert exc.value.kind == "control-flow"


def test_counted_while_with_a_runtime_step_keeps_refusing():
    runtime = _COUNTED_WHILE_IR.replace(
        'lhs=i.0, rhs=$one, fn="add"', 'lhs=i.0, rhs=N, fn="add"'
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(runtime, "t", multipath=True)
    assert exc.value.kind == "control-flow"


def test_counted_while_with_a_loop_carried_bound_keeps_refusing():
    carried_bound = _COUNTED_WHILE_IR.replace(
        'raw_cmp(lhs=i.0, rhs=N, fn="lt")', 'raw_cmp(lhs=i.0, rhs=acc.0, fn="lt")'
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(carried_bound, "t", multipath=True)
    assert exc.value.kind == "control-flow"


def test_counted_while_break_must_repeat_the_carried_slots():
    swapped = _COUNTED_WHILE_IR.replace(
        "        break acc.0, i.0, $tok.0", "        break acc.0, acc.0, $tok.0"
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(swapped, "t", multipath=True)
    assert exc.value.kind == "control-flow"


def test_counted_while_zero_trip_has_no_footprint():
    zero = _COUNTED_WHILE_IR.replace(
        '$c: Tile[bool_,()] = raw_cmp(lhs=i.0, rhs=N, fn="lt")',
        '$c: Tile[bool_,()] = raw_cmp(lhs=i.0, rhs=$i0, fn="lt")',
    )
    g = parse_cutile_ir(zero, "t", multipath=True)
    assert g.accesses == []


# ── Route 2: loaded values as snapshot Selects ───────────────────
# Under the L2 reader mode an integer load with a MODELED mask binds a
# Loaded term instead of DataDep, so an address built on a loaded value
# is a Select over the launch's pre-launch contents. L0/L1 keep DataDep
# and every refusal fires byte for byte, as before.

_SCATTER_IR = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()], idx_0: Tile[pointer[int32],()], idx_1: Tile[int32,()], idx_2: Tile[int32,()]):
$token: Token = make_token()
$ar: Tile[int32,(4)] = tile_arange()
$bid: Tile[int32,()] = tile_bid(axis=0)
$c4: const Tile[int32,()] = typed_const(value=4)
$base: Tile[int32,()] = raw_binary_arith(lhs=$bid, rhs=$c4, fn="mul", rounding_mode=None, flush_to_zero=False)
$br: Tile[int32,(1)] = tile_reshape(x=$base)
$bb: Tile[int32,(4)] = tile_broadcast(x=$br)
$off: Tile[int32,(4)] = raw_binary_arith(lhs=$bb, rhs=$ar, fn="add", rounding_mode=None, flush_to_zero=False)
$ipr: Tile[pointer[int32],(1)] = tile_reshape(x=idx_0)
$ipb: Tile[pointer[int32],(4)] = tile_broadcast(x=$ipr)
$ip: Tile[pointer[int32],(4)] = pointer_offset(pointer=$ipb, offset=$off)
$m: Tile[bool_,(4)] = raw_cmp(lhs=$off, rhs=idx_1, fn="lt")
$zi: Tile[int32,(4)] = typed_const(value=0)
$v: Tile[int32,(4)], $lt: Token = load_pointer(pointer=$ip, mask=$m, padding_value=$zi, token=$token, latency=None)
$xpr: Tile[pointer[float32],(1)] = tile_reshape(x=x_0)
$xpb: Tile[pointer[float32],(4)] = tile_broadcast(x=$xpr)
$xp: Tile[pointer[float32],(4)] = pointer_offset(pointer=$xpb, offset=$v)
$fv: Tile[float32,(4)] = typed_const(value=0)
$st: Token = store_pointer(pointer=$xp, value=$fv, mask=$m, token=$token, latency=None)
return
"""


def _solve_scatter(table, *, ir=_SCATTER_IR, multipath=True, snapshot=True, n=16):
    g = parse_cutile_ir(ir, "t", multipath=multipath)
    tensors = {
        "x": GlobalTensor(data_ptr=1 << 40, numel=n, elem_size=4, contiguous=True),
        "idx": GlobalTensor(
            data_ptr=(1 << 40) + 8192,
            numel=n,
            elem_size=4,
            contiguous=True,
            snapshot=tuple(table) if snapshot else None,
            snapshot_reason="" if snapshot else "not captured",
        ),
    }
    enc = encode_graph(
        g,
        {"x_1": n, "x_2": 1, "idx_1": n, "idx_2": 1},
        tensors,
        multipath=multipath,
    )
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (4, 1, 1)),
        arange_dict=enc.arange_dict,
        fence_order=True,
        fence_seqs=enc.fence_seqs,
        token_order=enc.token_order,
    )
    return g, enc, solver.find_races()


def test_scatter_over_a_permutation_proves_content_qualified():
    from triton_viz.clients.common.ttir_reader import mentions_loaded

    g, enc, found = _solve_scatter(range(16))
    assert mentions_loaded(g.accesses[1].offset), "the address IS the loaded value"
    assert enc.content_qualified, "the proof rests on this launch's contents"
    assert found == []


def test_scatter_over_a_duplicated_index_races():
    table = list(range(16))
    table[7] = 3  # two programs now write element 3
    _g, _enc, found = _solve_scatter(table)
    assert len(found) == 1


def test_scatter_without_a_snapshot_refuses_by_name():
    with pytest.raises(UnsupportedTTIR) as exc:
        _solve_scatter(range(16), snapshot=False)
    assert exc.value.kind == "indirect-address"
    assert "no usable snapshot" in str(exc.value)


def test_scatter_single_path_keeps_the_old_refusal():
    # L0/L1 bind DataDep, so the message is the pre-Route-2 one, verbatim
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(_SCATTER_IR, "t", multipath=False)
    assert (exc.value.kind, str(exc.value)) == (
        "indirect-address",
        "line 18: pointer offset: data-dependent (loaded value)",
    )


def test_float_loaded_value_in_an_address_stays_datadep():
    # a float pointee is outside the Int model: no snapshot Select for it
    ir = _SCATTER_IR.replace("pointer[int32]", "pointer[float32]").replace(
        "$v: Tile[int32,(4)]", "$v: Tile[float32,(4)]"
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(ir, "t", multipath=True)
    assert exc.value.kind == "indirect-address"
    assert "loaded value" in str(exc.value)


def test_loaded_value_under_a_dropped_mask_stays_datadep():
    # only a MODELED mask keeps a masked-off lane (which holds `other` or
    # an undefined value) apart from the snapshot value
    ir = _SCATTER_IR.replace(
        '$m: Tile[bool_,(4)] = raw_cmp(lhs=$off, rhs=idx_1, fn="lt")',
        "$u: Tile[int32,(4)] = mystery_op(x=$off)\n"
        '$m: Tile[bool_,(4)] = raw_cmp(lhs=$u, rhs=idx_1, fn="lt")',
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_cutile_ir(ir, "t", multipath=True)
    assert exc.value.kind == "indirect-address"


_TILE_LOAD_SCATTER_IR = """\
(x_0: Tile[pointer[float32],()], x_1: Tile[int32,()], x_2: Tile[int32,()], idx_0: Tile[pointer[int32],()], idx_1: Tile[int32,()], idx_2: Tile[int32,()]):
$token: Token = make_token()
$0: Tile[int32,()] = assume_bounded(x=idx_1, lower_bound=0, upper_bound=None)
idx{idx_0, $0, idx_2}: Array[int32,(?):(1)] = make_tensor_view(base_ptr=idx_0, shape=($0), dynamic_strides=())
$1: Tile[int32,()] = tile_bid(axis=0)
$2{idx_0, $0, idx_2}: PartitionView[Array[int32,(?):(1)],tile_shape=(4,),order=(0,),padding_mode=PaddingMode.UNDETERMINED] = make_partition_view(array=idx{idx_0, $0, idx_2})
$3: Tile[int32,(4)], $4: Token = tile_load(view=$2{idx_0, $0, idx_2}, index=($1), token=$token, latency=None, allow_tma=None, memory_order=MemoryOrder.WEAK, memory_scope=MemoryScope.NONE)
$5: Tile[pointer[float32],(1)] = tile_reshape(x=x_0)
$6: Tile[pointer[float32],(4)] = tile_broadcast(x=$5)
$7: Tile[pointer[float32],(4)] = pointer_offset(pointer=$6, offset=$3)
$8: Tile[float32,(4)] = typed_const(value=0)
$9: Token = store_pointer(pointer=$7, value=$8, mask=None, token=$token, latency=None)
return
"""


def _first_loaded(term):
    from triton_viz.clients.common.ttir_reader import Loaded

    if isinstance(term, Loaded):
        return term
    for attr in ("a", "b", "cond", "t", "f", "offset", "mask", "other"):
        sub = getattr(term, attr, None)
        if sub is not None:
            hit = _first_loaded(sub)
            if hit is not None:
                return hit
    return None


def test_tile_load_takes_other_from_the_views_padding_mode():
    # cuTile has no `other` operand: a clipped lane's value is the
    # partition view's padding mode. ZERO names a value; UNDETERMINED
    # leaves the lane unspecified, which is the widening direction.
    undetermined = parse_cutile_ir(_TILE_LOAD_SCATTER_IR, "t", multipath=True)
    zero = parse_cutile_ir(
        _TILE_LOAD_SCATTER_IR.replace("PaddingMode.UNDETERMINED", "PaddingMode.ZERO"),
        "t",
        multipath=True,
    )
    assert _first_loaded(undetermined.accesses[1].offset).other is None
    assert _first_loaded(zero.accesses[1].offset).other == Const(0)


def test_load_pointer_takes_other_from_its_padding_value():
    with_other = parse_cutile_ir(_SCATTER_IR, "t", multipath=True)
    assert _first_loaded(with_other.accesses[1].offset).other == Const(0)
    without = parse_cutile_ir(
        _SCATTER_IR.replace("padding_value=$zi", "padding_value=None"),
        "t",
        multipath=True,
    )
    assert _first_loaded(without.accesses[1].offset).other is None


def test_param_pinned_only_when_the_rewrite_reaches_the_claim():
    """radix_sort's shape: ``(key >> bit) & 1`` is COUNTED, never used to
    address. The rewrite is exact only for this launch's ``bit``, but the
    footprint does not depend on it, so the ANY-params tier stays open."""
    from triton_viz.clients.race_detector.compiled.global_records import (
        t0_linearity_gate,
    )

    ir = (
        _BITWISE_IR.replace("OP", "and_")
        .replace("pointer[float32]", "pointer[int32]")
        .replace(
            "$p: Tile[pointer[int32],(64)] = pointer_offset(pointer=$pb, offset=$xo)",
            "$p: Tile[pointer[int32],(64)] = pointer_offset(pointer=$pb, offset=$ar)",
        )
        .replace("$v: Tile[float32,(64)] = typed_const(value=0)\n", "")
        .replace("value=$v", "value=$xo")
    )
    g = parse_cutile_ir(ir, "t", params={"j": 7})
    assert not g.param_pinned, "the launch value never reaches the footprint"
    assert t0_linearity_gate(g)
    # the same rewrite IN the address does close the ANY-params tier
    pinned = parse_cutile_ir(_BITWISE_IR.replace("OP", "and_"), "t", params={"j": 7})
    assert pinned.param_pinned and not t0_linearity_gate(pinned)
