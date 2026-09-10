"""A pointer tile through ``tt.expand_dims``: the address's lane variables
must follow the new dimension exactly like the mask's.

``conv_state_base = ptr + feats`` (1-D pointers) then
``conv_state_base[None, :] + (tokens * stride)[:, None]`` with the mask
``(tokens < L)[:, None] & (feats < D)[None, :]``: Triton expands the
POINTER tensor. The reader once left a ``PtrValue``'s offset untagged, so
the store's address used the 1-D ``feats`` variable while its mask used
the 2-D one; the two program copies could then differ in the mask's lane
while sharing an address, a phantom intra-instance WAW (aiter's
causal_conv1d update kernels, reported as definite races by the first
Route 2 change-surface run).
"""

from triton_viz.clients.common.ttir_reader import Arange, parse_ttir
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_t1_rmw_static import _module

POINTER_EXPAND = _module(
    "%out_ptr: !tt.ptr<f32>, %L: i32, %D: i32",
    "%c1 = arith.constant 1.0 : f32",
    "%c128 = arith.constant 128 : i32",
    "%pid = tt.get_program_id x : i32",
    "%toks = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>",
    "%feats = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>",
    "%pb = arith.muli %pid, %c128 : i32",
    "%pbs = tt.splat %pb : i32 -> tensor<8xi32>",
    "%foff = arith.addi %pbs, %feats : tensor<8xi32>",
    "%bs = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>",
    "%base = tt.addptr %bs, %foff : tensor<8x!tt.ptr<f32>>, tensor<8xi32>",
    "%base2 = tt.expand_dims %base {axis = 0 : i32} : tensor<8x!tt.ptr<f32>> -> tensor<1x8x!tt.ptr<f32>>",
    "%base2b = tt.broadcast %base2 : tensor<1x8x!tt.ptr<f32>> -> tensor<2x8x!tt.ptr<f32>>",
    "%c64 = arith.constant 64 : i32",
    "%c64s = tt.splat %c64 : i32 -> tensor<2xi32>",
    "%to = arith.muli %toks, %c64s : tensor<2xi32>",
    "%to2 = tt.expand_dims %to {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>",
    "%to2b = tt.broadcast %to2 : tensor<2x1xi32> -> tensor<2x8xi32>",
    "%p = tt.addptr %base2b, %to2b : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>",
    "%Ls = tt.splat %L : i32 -> tensor<2xi32>",
    "%mt = arith.cmpi slt, %toks, %Ls : tensor<2xi32>",
    "%mt2 = tt.expand_dims %mt {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>",
    "%mt2b = tt.broadcast %mt2 : tensor<2x1xi1> -> tensor<2x8xi1>",
    "%Ds = tt.splat %D : i32 -> tensor<8xi32>",
    "%mf = arith.cmpi slt, %feats, %Ds : tensor<8xi32>",
    "%mf2 = tt.expand_dims %mf {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>",
    "%mf2b = tt.broadcast %mf2 : tensor<1x8xi1> -> tensor<2x8xi1>",
    "%m = arith.andi %mt2b, %mf2b : tensor<2x8xi1>",
    "%vs = tt.splat %c1 : f32 -> tensor<2x8xf32>",
    "tt.store %p, %vs, %m : tensor<2x8x!tt.ptr<f32>>",
)


def _dims(term, ssa):
    out = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Arange) and t.ssa == ssa:
            out.add(t.dim)
        for attr in ("a", "b", "cond", "t", "f", "offset", "mask", "other"):
            sub = getattr(t, attr, None)
            if sub is not None:
                stack.append(sub)
    return out


def test_expanded_pointer_tile_shares_its_lane_variables_with_the_mask():
    g = parse_ttir(POINTER_EXPAND)
    (store,) = g.accesses
    # feats varies along dim 1 in BOTH the address and the mask; toks along dim 0
    assert _dims(store.offset, "%feats") == {1} == _dims(store.mask, "%feats")
    assert _dims(store.offset, "%toks") == {0} == _dims(store.mask, "%toks")
    enc = encode_graph(
        g,
        {"L": 2, "D": 8},
        {"out_ptr": GlobalTensor(data_ptr=0x100000, elem_size=4, numel=1 << 12)},
    )
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=symbolic_grid(enc, (4, 1, 1)), arange_dict=enc.arange_dict
    )
    # every (token, feature) lane of every pid has its own address: no race
    assert solver.find_races() == []
