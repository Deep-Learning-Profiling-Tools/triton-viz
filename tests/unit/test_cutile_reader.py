"""Pins for the CuTile IR reader front-end (clients/common/cutile_ir_reader).

Self-contained IR snippets (grammar-faithful to cuda-tile 1.5.0's final
CuTile IR text) exercise the semantic mapping: tile-space addressing to
affine terms with implicit-clip masks, the raw-pointer gather/scatter and
atomic paths, the boolean-xor floor-division lowering, and the abstention
discipline for integer xor and while-form loops. End-to-end pins run the
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
        enc.records, grid=symbolic_grid(enc, grid), arange_dict=enc.arange_dict
    )
    return g, solver.find_races()


def test_tile_store_partitioned_by_bid_proves_clean():
    g, found = _solve(_STORE_IR.replace("INDEX", "$1"))
    assert [a.kind for a in g.accesses] == ["store"]
    assert g.accesses[0].base_param == "x"
    assert g.accesses[0].mask is not None, "implicit OOB clip must be a mask"
    assert g.pid_axes == {0}
    assert found == []


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
