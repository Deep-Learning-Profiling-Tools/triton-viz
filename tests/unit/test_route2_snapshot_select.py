"""Route 2: loaded values as snapshot Selects in the static frontend
(the L2 reader mode), on top of Route 3's multipath capture.

An integer ``tt.load`` with a modeled mask binds a ``Loaded`` term; the
encoder evaluates it as ``If(mask ∧ in-domain, snap[off], other-or-free)``
over the launch's pre-launch snapshot of the source tensor and marks the
verdict content-qualified. Without a usable snapshot (T0, a float / large
/ non-contiguous source, a source the kernel writes) the value is FREE:
the widening Route 3 applied (uncertain record) in mask, path and bound
positions, a refusal by name in address position. Single-path parsing is
unchanged (loaded values stay DataDep).
"""

from types import SimpleNamespace

import pytest
import torch

from triton_viz.clients.common.ttir_reader import (
    Arange,
    Loaded,
    UnsupportedTTIR,
    loaded_leaves,
    parse_ttir,
)
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_t1_rmw_static import _module


def _t(ptr, numel, snapshot=None, reason="", elem=4):
    return GlobalTensor(
        data_ptr=ptr,
        elem_size=elem,
        numel=numel,
        snapshot=snapshot,
        snapshot_reason=reason,
    )


def _mp(text):
    return parse_ttir(text, multipath=True)


def _t1(graph, params, tensors, grid=(4, 1, 1), pinned=True):
    """The T1 query with the snapshot equalities asserted; ``pinned``
    confines the pids to the launch extent (the client's launch-scoped
    rung), otherwise the grid stays symbolic (the any-grid rung, where an
    instance beyond the snapshotted table is excluded by the consumer's
    domain premise rather than admitted with an unspecified value)."""
    from z3 import IntVal

    enc = encode_graph(graph, params, tensors, multipath=True)
    if not pinned:
        solver = TwoCopySymbolicHBSolver(
            enc.records,
            grid=symbolic_grid(enc, grid),
            arange_dict=enc.arange_dict,
            extra_assumptions=enc.assumptions,
        )
        return enc, solver.find_races()
    g = symbolic_grid(enc, grid)
    pins = tuple(
        d == IntVal(grid[i]) for i, d in enumerate(g) if not isinstance(d, int)
    )
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=g,
        arange_dict=enc.arange_dict,
        enum_fallback_grid=grid,
        extra_assumptions=enc.assumptions + pins,
        launch_ceiling=True,
    )
    return enc, solver.find_races()


def _pids(rep):
    return rep.witness_grid_a[0], rep.witness_grid_b[0]


# ───────────────────── the scatter litmus (trb010) ─────────────────────

SCATTER = _module(
    "%idx_ptr: !tt.ptr<i32>, %x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
    "%c4 = arith.constant 4 : i32",
    "%pid = tt.get_program_id x : i32",
    "%base = arith.muli %pid, %c4 : i32",
    "%ar = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>",
    "%bs = tt.splat %base : i32 -> tensor<4xi32>",
    "%offs = arith.addi %bs, %ar : tensor<4xi32>",
    "%is = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>",
    "%ip = tt.addptr %is, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>",
    "%i = tt.load %ip : tensor<4x!tt.ptr<i32>>",
    "%xs = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>",
    "%xp = tt.addptr %xs, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>",
    "%v = tt.load %xp : tensor<4x!tt.ptr<i32>>",
    "%os = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>",
    "%op = tt.addptr %os, %i : tensor<4x!tt.ptr<i32>>, tensor<4xi32>",
    "tt.store %op, %v : tensor<4x!tt.ptr<i32>>",
)
PERM = tuple(range(16))  # a permutation: every instance's 4 targets distinct
DUP = tuple(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 3]
)  # pid 3 hits pid 0's slot 3


def _tensors(idx):
    return {
        "idx_ptr": _t(0x100000, 16, snapshot=idx),
        "x_ptr": _t(0x200000, 16),
        "out_ptr": _t(0x300000, 64),
    }


def test_single_path_still_refuses_the_scatter():
    with pytest.raises(UnsupportedTTIR) as ei:
        parse_ttir(SCATTER)
    assert ei.value.kind == "indirect-address"


def test_scatter_binds_a_loaded_index_and_uses_it_in_the_address():
    g = _mp(SCATTER)
    idx_load, x_load, store = g.accesses
    (leaf,) = loaded_leaves(store.offset)
    assert isinstance(leaf, Loaded)
    assert leaf.access_index == 0 and leaf.base_param == "idx_ptr"
    assert leaf.offset == idx_load.offset and leaf.mask is None


def test_permutation_scatter_proves_content_qualified_and_duplicates_race():
    g = _mp(SCATTER)
    enc, reports = _t1(g, {}, _tensors(PERM))
    assert enc.content_qualified and enc.uncertain_event_ids == set()
    assert len(enc.assumptions) == 16
    assert reports == []  # within the launch extent a permutation never collides
    enc, reports = _t1(g, {}, _tensors(DUP))
    assert reports and any(set(_pids(r)) == {0, 3} for r in reports)


def test_any_grid_query_excludes_instances_beyond_the_table():
    """The consumer store carries the index load's domain premise
    (``0 <= pid*4+lane < 16``): an instance beyond the 16-entry table
    would read its index out of bounds, which is outside the model, so
    the ANY-grid query proves the permutation instead of admitting a
    free out-of-table value that two instances could agree on."""
    g = _mp(SCATTER)
    (store,) = [a for a in g.accesses if a.kind == "store"]
    enc, reports = _t1(g, {}, _tensors(PERM), pinned=False)
    assert reports == [] and enc.uncertain_event_ids == set()
    (rec,) = [r for r in enc.records if r.event_id == 2]
    assert len(rec.local_constraints) == 3  # two bounds + the domain premise
    enc, reports = _t1(g, {}, _tensors(DUP), pinned=False)
    assert reports and all(max(_pids(r)) < 4 for r in reports)


def test_client_lands_the_scatter_on_the_content_qualified_rungs():
    jit = SimpleNamespace(arg_names=["idx_ptr", "x_ptr", "out_ptr"])

    def run(idx):
        det = CompiledRaceDetector(confirm_races=False, ladder_level=2)
        det.pre_warmup_callback(
            jit,
            torch.tensor(idx, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(64, dtype=torch.int32),
            grid=(4,),
        )
        det.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": SCATTER}))
        det.finalize()
        return det

    l0 = CompiledRaceDetector(confirm_races=False, ladder_level=0)
    l0.pre_warmup_callback(
        jit,
        torch.tensor(PERM, dtype=torch.int32),
        torch.zeros(16, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
        grid=(4,),
    )
    l0.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": SCATTER}))
    l0.finalize()
    assert l0.last_global_status == "unsupported"
    assert (l0.last_global_reason or "").startswith("indirect-address")

    det = run(PERM)
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1+content"
    assert det.last_global_verdict["content_qualified"]
    det = run(DUP)
    assert det.last_global_status == "races"
    assert det.last_global_verdict["content_qualified"]
    assert any(set(_pids(r)) == {0, 3} for r in det.last_global_reports)


def test_client_never_certifies_a_race_on_out_of_table_instances():
    """A callable grid leaves the client without a launch extent, so the
    launch-scoped requery cannot run: the any-grid verdict must stand on
    its own. With the domain premise the permutation proves; the report
    for the duplicate stays in-table."""
    jit = SimpleNamespace(arg_names=["idx_ptr", "x_ptr", "out_ptr"])

    def run(idx):
        det = CompiledRaceDetector(confirm_races=False, ladder_level=2)
        det.pre_warmup_callback(
            jit,
            torch.tensor(idx, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(64, dtype=torch.int32),
            grid=lambda meta: (4,),
        )
        det.post_warmup_callback(jit, SimpleNamespace(asm={"ttir": SCATTER}))
        det.finalize()
        return det

    det = run(PERM)
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1+content"
    det = run(DUP)
    assert det.last_global_status == "races"
    assert all(max(_pids(r)) < 4 for r in det.last_global_reports)


# ───────────────────── refusals and widening ─────────────────────


def test_address_on_a_load_without_a_snapshot_refuses_by_name():
    g = _mp(SCATTER)
    too_large = dict(
        _tensors(PERM),
        idx_ptr=_t(0x100000, 16, reason="too large (40000 elements, bound 16384)"),
    )
    with pytest.raises(UnsupportedTTIR, match="too large") as ei:
        encode_graph(g, {}, too_large, multipath=True)
    assert ei.value.kind == "snapshot-bound"
    missing = dict(
        _tensors(PERM), idx_ptr=_t(0x100000, 16, reason="float dtype torch.float32")
    )
    with pytest.raises(UnsupportedTTIR, match="no usable snapshot") as ei:
        encode_graph(g, {}, missing, multipath=True)
    assert ei.value.kind == "indirect-address"


def test_source_written_by_the_kernel_is_unusable():
    """The read-only-source premise: idx_ptr aliases a tensor the kernel
    stores to, so its pre-launch snapshot cannot stand for the loaded
    value; in address position that is a refusal."""
    g = _mp(SCATTER)
    aliased = dict(_tensors(PERM), out_ptr=_t(0x100000, 64))  # out overlaps idx
    with pytest.raises(UnsupportedTTIR, match="overlaps the source") as ei:
        encode_graph(g, {}, aliased, multipath=True)
    assert ei.value.kind == "indirect-address"


def test_t0_keeps_loaded_values_free_and_refuses_them_in_addresses():
    with pytest.raises(UnsupportedTTIR, match="no usable snapshot"):
        encode_graph_t0(_mp(SCATTER), multipath=True)


GUARD_THEN_ADDRESS = _module(
    "%idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
    "%c1 = arith.constant 1 : i32",
    "%cm1 = arith.constant -1 : i32",
    "%pid = tt.get_program_id x : i32",
    "%ip = tt.addptr %idx_ptr, %pid : !tt.ptr<i32>, i32",
    "%y = tt.load %ip : !tt.ptr<i32>",
    "%g = arith.cmpi eq, %y, %cm1 : i32",
    "scf.if %g {",
    "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
    "}",
    "%op = tt.addptr %out_ptr, %y : !tt.ptr<i32>, i32",
    "tt.store %op, %c1 : !tt.ptr<i32>",
)


def test_free_loaded_in_an_address_refuses_regardless_of_evaluation_order():
    """``y = idx[pid]; if y == -1: out[0] = 1; out[y] = 1``: the guarded
    store evaluates the loaded value FIRST (in its path), so a refusal
    keyed on "newly free during this address" would miss the second
    store's address and lower it to a free copy-local array, a definite
    race no launch produces. The refusal is structural on the address."""
    g = _mp(GUARD_THEN_ADDRESS)
    tensors = {
        "idx_ptr": _t(0x100000, 4, reason="too large (40000 elements, bound 16384)"),
        "out_ptr": _t(0x300000, 64),
    }
    with pytest.raises(UnsupportedTTIR, match="too large") as ei:
        encode_graph(g, {}, tensors, multipath=True)
    assert ei.value.kind == "snapshot-bound"
    with pytest.raises(UnsupportedTTIR, match="no usable snapshot") as ei:
        encode_graph_t0(g, multipath=True)
    assert ei.value.kind == "indirect-address"
    tensors["idx_ptr"] = _t(0x100000, 4, snapshot=(1, 2, 3, 4))
    enc, reports = _t1(g, {}, tensors)
    assert enc.content_qualified and enc.uncertain_event_ids == set()
    assert reports == []  # the guard never fires, out[1..4] distinct per pid
    tensors["idx_ptr"] = _t(0x100000, 4, snapshot=(1, -1, 3, 1))
    enc, reports = _t1(g, {}, tensors)
    assert reports and any(set(_pids(r)) == {0, 3} for r in reports)


MASKED_GUARD = _module(
    "%out_ptr: !tt.ptr<i32>, %flag_ptr: !tt.ptr<i32>, %n: i32",
    "%c1 = arith.constant 1 : i32",
    "%c0 = arith.constant 0 : i32",
    "%pid = tt.get_program_id x : i32",
    "%m = arith.cmpi slt, %pid, %n : i32",
    "%fp = tt.addptr %flag_ptr, %pid : !tt.ptr<i32>, i32",
    "%f = tt.load %fp, %m, %c0 : !tt.ptr<i32>",
    "%g = arith.cmpi ne, %f, %c0 : i32",
    "scf.if %g {",
    "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
    "}",
)


def test_masked_load_with_other_and_a_flag_guard():
    """``f = load(flag[pid], mask=pid < n, other=0); if f != 0: store out[0]``.
    With the flags snapshotted, the guard is exact: one raised flag proves,
    two race. Without a snapshot the guard is free and the store widened."""
    g = _mp(MASKED_GUARD)
    (store,) = [a for a in g.accesses if a.kind == "store"]
    (leaf,) = loaded_leaves(store.path)
    assert leaf.mask is not None and leaf.other is not None
    one = {
        "out_ptr": _t(0x300000, 64),
        "flag_ptr": _t(0x400000, 4, snapshot=(1, 0, 0, 0)),
    }
    enc, reports = _t1(g, {"n": 4}, one)
    assert enc.content_qualified and reports == []
    two = {
        "out_ptr": _t(0x300000, 64),
        "flag_ptr": _t(0x400000, 4, snapshot=(1, 0, 1, 0)),
    }
    enc, reports = _t1(g, {"n": 4}, two)
    assert reports and all(set(_pids(r)) == {0, 2} for r in reports)
    free = {
        "out_ptr": _t(0x300000, 64),
        "flag_ptr": _t(0x400000, 4, reason="too large"),
    }
    enc, reports = _t1(g, {"n": 4}, free)
    assert not enc.content_qualified and 1 in enc.uncertain_event_ids
    assert reports  # widened, never definite: the client withholds them


def test_masked_off_lanes_take_other():
    """n = 2 masks pids 2 and 3 off: they read ``other``, not the flag
    table. other = 0 keeps them out of the guard (flags (1,0,1,0) prove);
    other = 1 sends both through it (flags all zero race between 2 and 3),
    so a lowering that dropped the mask or the other would flip both."""
    g = _mp(MASKED_GUARD)
    two = {
        "out_ptr": _t(0x300000, 64),
        "flag_ptr": _t(0x400000, 4, snapshot=(1, 0, 1, 0)),
    }
    enc, reports = _t1(g, {"n": 2}, two)
    assert enc.uncertain_event_ids == set() and reports == []
    other_one = _mp(MASKED_GUARD.replace("%fp, %m, %c0", "%fp, %m, %c1"))
    none = {
        "out_ptr": _t(0x300000, 64),
        "flag_ptr": _t(0x400000, 4, snapshot=(0, 0, 0, 0)),
    }
    enc, reports = _t1(other_one, {"n": 2}, none)
    assert enc.uncertain_event_ids == set()
    assert reports and all(set(_pids(r)) == {2, 3} for r in reports)


def test_masked_load_without_other_widens_unless_the_consumer_repeats_the_mask():
    """Without ``other`` a masked-off lane holds an unspecified value. A
    guard built on it decides the store's activity for those lanes, so
    the record is uncertain (its reports are widened, never definite).
    When the consumer's own mask repeats the load's mask the unspecified
    lanes are inactive for it and the record stays exact."""
    no_other = _mp(
        MASKED_GUARD.replace("%fp, %m, %c0 : !tt.ptr<i32>", "%fp, %m : !tt.ptr<i32>")
    )
    (store,) = [a for a in no_other.accesses if a.kind == "store"]
    (leaf,) = loaded_leaves(store.path)
    assert leaf.mask is not None and leaf.other is None
    flags = {
        "out_ptr": _t(0x300000, 64),
        "flag_ptr": _t(0x400000, 4, snapshot=(1, 0, 0, 0)),
    }
    enc, reports = _t1(no_other, {"n": 1}, flags)
    assert enc.content_qualified and 1 in enc.uncertain_event_ids
    assert reports  # pids 1..3 are masked off: their guard is unspecified
    # the scatter with the index load and the store under the SAME mask
    masked_scatter = _mp(
        SCATTER.replace(
            "%i = tt.load %ip : tensor<4x!tt.ptr<i32>>",
            "%ns = tt.splat %n : i32 -> tensor<4xi32>\n"
            "    %m = arith.cmpi slt, %offs, %ns : tensor<4xi1>\n"
            "    %i = tt.load %ip, %m : tensor<4x!tt.ptr<i32>>",
        )
        .replace(
            "tt.store %op, %v : tensor<4x!tt.ptr<i32>>",
            "tt.store %op, %v, %m : tensor<4x!tt.ptr<i32>>",
        )
        .replace("%out_ptr: !tt.ptr<i32>", "%out_ptr: !tt.ptr<i32>, %n: i32")
    )
    (store,) = [a for a in masked_scatter.accesses if a.kind == "store"]
    (leaf,) = loaded_leaves(store.offset)
    assert leaf.mask is not None and leaf.other is None and store.mask == leaf.mask
    enc, reports = _t1(masked_scatter, {"n": 8}, _tensors(PERM))
    assert enc.uncertain_event_ids == set() and reports == []
    enc, reports = _t1(masked_scatter, {"n": 16}, _tensors(DUP))
    assert enc.uncertain_event_ids == set()
    assert reports and any(set(_pids(r)) == {0, 3} for r in reports)


def test_loaded_index_tile_follows_expand_dims():
    """``rows = load(row_ptr + offs_m)`` then ``out + rows[:, None] * S +
    offs_n[None, :]``: the Loaded's lane arange is retagged with the
    consumer's dimension exactly like the address's own lane_ranges."""
    text = _module(
        "%row_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>, %S: i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%c2 = arith.constant 2 : i32",
        "%om = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>",
        "%on = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>",
        "%pb = arith.muli %pid, %c2 : i32",
        "%pbs = tt.splat %pb : i32 -> tensor<2xi32>",
        "%rowi = arith.addi %pbs, %om : tensor<2xi32>",
        "%rs = tt.splat %row_ptr : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>",
        "%rp = tt.addptr %rs, %rowi : tensor<2x!tt.ptr<i32>>, tensor<2xi32>",
        "%rows = tt.load %rp : tensor<2x!tt.ptr<i32>>",
        "%r2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>",
        "%Ss = tt.splat %S : i32 -> tensor<2x1xi32>",
        "%rS = arith.muli %r2, %Ss : tensor<2x1xi32>",
        "%rSb = tt.broadcast %rS : tensor<2x1xi32> -> tensor<2x2xi32>",
        "%n2 = tt.expand_dims %on {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>",
        "%n2b = tt.broadcast %n2 : tensor<1x2xi32> -> tensor<2x2xi32>",
        "%o = arith.addi %rSb, %n2b : tensor<2x2xi32>",
        "%os = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<2x2x!tt.ptr<i32>>",
        "%op = tt.addptr %os, %o : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>",
        "%vs = tt.splat %c1 : i32 -> tensor<2x2xi32>",
        "tt.store %op, %vs : tensor<2x2x!tt.ptr<i32>>",
    )
    g = _mp(text)
    (store,) = [a for a in g.accesses if a.kind == "store"]
    (leaf,) = loaded_leaves(store.offset)
    lane_ranges = [t for t in _walk(leaf.offset) if isinstance(t, Arange)]
    assert lane_ranges and all(a.dim == 0 for a in lane_ranges)
    # rows = [0,1,2,3,...] -> distinct rows per pid: proved; rows all 0 -> race
    tensors = {
        "row_ptr": _t(0x100000, 8, snapshot=tuple(range(8))),
        "out_ptr": _t(0x300000, 1024),
    }
    enc, reports = _t1(g, {"S": 2}, tensors)
    assert reports == []
    tensors["row_ptr"] = _t(0x100000, 8, snapshot=tuple([0] * 8))
    enc, reports = _t1(g, {"S": 2}, tensors)
    assert reports and all(max(_pids(r)) < 4 for r in reports)


LOADED_ADVANCE = _module(
    "%out_ptr: !tt.ptr<i32>, %step_ptr: !tt.ptr<i32>, %n: i32",
    "%c0 = arith.constant 0 : i32",
    "%c1 = arith.constant 1 : i32",
    "%pid = tt.get_program_id x : i32",
    "%p0 = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
    "%r = scf.for %k = %c0 to %n step %c1 iter_args(%p = %p0) -> (!tt.ptr<i32>)  : i32 {",
    "tt.store %p, %c1 : !tt.ptr<i32>",
    "%sp = tt.addptr %step_ptr, %k : !tt.ptr<i32>, i32",
    "%s = tt.load %sp : !tt.ptr<i32>",
    "%pn = tt.addptr %p, %s : !tt.ptr<i32>, i32",
    "scf.yield %pn : !tt.ptr<i32>",
    "}",
)


def test_loaded_pointer_advance_refuses_instead_of_a_false_proof():
    """``p += step[k]`` inside the loop: ``offset0 + k·delta`` stands for
    the pointer only for a loop-invariant advance. With step = (1, 5),
    pid 0 writes out[0], out[1] and pid 1 writes out[1], out[2] (a WAW
    the k·step[k] model misses), so the iter_arg refuses by name."""
    g = _mp(LOADED_ADVANCE)
    tensors = {
        "out_ptr": _t(0x300000, 64),
        "step_ptr": _t(0x100000, 2, snapshot=(1, 5)),
    }
    with pytest.raises(UnsupportedTTIR, match="pointer advance") as ei:
        encode_graph(g, {"n": 2}, tensors, multipath=True)
    assert ei.value.kind == "indirect-address"
    with pytest.raises(UnsupportedTTIR, match="pointer advance"):
        encode_graph_t0(g, multipath=True)


def test_unmodelable_other_leaves_the_masked_lanes_unspecified_and_widens():
    """``other`` is an i32 ``andi`` (unmodeled): the reader drops it, so a
    masked-off lane holds an unspecified value; the consumer's address
    goes through it, and the record is uncertain (never a definite race:
    the real lanes hold the kernel's defined ``pid & 7``)."""
    text = _module(
        "%idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>, %n: i32",
        "%c7 = arith.constant 7 : i32",
        "%pid = tt.get_program_id x : i32",
        "%m = arith.cmpi slt, %pid, %n : i32",
        "%o = arith.andi %pid, %c7 : i32",
        "%ip = tt.addptr %idx_ptr, %pid : !tt.ptr<i32>, i32",
        "%i = tt.load %ip, %m, %o : !tt.ptr<i32>",
        "%op = tt.addptr %out_ptr, %i : !tt.ptr<i32>, i32",
        "tt.store %op, %c7 : !tt.ptr<i32>",
    )
    g = _mp(text)
    (store,) = [a for a in g.accesses if a.kind == "store"]
    (leaf,) = loaded_leaves(store.offset)
    assert leaf.mask is not None and leaf.other is None
    tensors = {
        "idx_ptr": _t(0x100000, 4, snapshot=(0, 1, 2, 3)),
        "out_ptr": _t(0x300000, 64),
    }
    enc, reports = _t1(g, {"n": 2}, tensors)
    assert enc.content_qualified and 1 in enc.uncertain_event_ids
    # every witness involves a masked-off pid (its unspecified value can
    # match anything, including pid 1's real index): widened, never definite
    assert reports and all(max(_pids(r)) >= 2 for r in reports)
    enc, reports = _t1(g, {"n": 4}, tensors)  # no lane masked off: exact
    assert 1 in enc.uncertain_event_ids and reports == []


def _walk(term):
    yield term
    for attr in ("a", "b", "cond", "t", "f", "offset", "mask", "other"):
        sub = getattr(term, attr, None)
        if sub is not None:
            yield from _walk(sub)


LOADED_BOUNDS = _module(
    "%ptr_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
    "%c1 = arith.constant 1 : i32",
    "%pid = tt.get_program_id x : i32",
    "%pid1 = arith.addi %pid, %c1 : i32",
    "%lp = tt.addptr %ptr_ptr, %pid : !tt.ptr<i32>, i32",
    "%lo = tt.load %lp : !tt.ptr<i32>",
    "%hp = tt.addptr %ptr_ptr, %pid1 : !tt.ptr<i32>, i32",
    "%hi = tt.load %hp : !tt.ptr<i32>",
    "scf.for %k = %lo to %hi step %c1  : i32 {",
    "%op = tt.addptr %out_ptr, %k : !tt.ptr<i32>, i32",
    "tt.store %op, %c1 : !tt.ptr<i32>",
    "scf.yield",
    "}",
)


def test_csr_loop_bounds_from_a_loaded_row_pointer_table():
    """``for k in range(rowptr[pid], rowptr[pid+1])``: with the table
    snapshotted the bounds are exact Selects (disjoint segments prove,
    overlapping ones race); without it the range is free and the loop's
    accesses are widened."""
    g = _mp(LOADED_BOUNDS)
    assert (
        g.loop is not None
        and loaded_leaves(g.loop.lower)
        and loaded_leaves(g.loop.upper)
    )
    disjoint = {
        "ptr_ptr": _t(0x100000, 5, snapshot=(0, 3, 5, 9, 12)),
        "out_ptr": _t(0x300000, 64),
    }
    enc, reports = _t1(g, {}, disjoint)
    assert enc.content_qualified and reports == []
    # rowptr (0, 4, 1, 9, 12): pid 0 walks [0, 4), pid 1 an empty [4, 1),
    # pid 2 [1, 9), which overlaps pid 0's segment
    overlap = {
        "ptr_ptr": _t(0x100000, 5, snapshot=(0, 4, 1, 9, 12)),
        "out_ptr": _t(0x300000, 64),
    }
    enc, reports = _t1(g, {}, overlap)
    assert reports and any(set(_pids(r)) == {0, 2} for r in reports)
    free = {"ptr_ptr": _t(0x100000, 5, reason="too large"), "out_ptr": _t(0x300000, 64)}
    enc, reports = _t1(g, {}, free)
    assert not enc.content_qualified and enc.uncertain_event_ids == {2}
    # the bounds' loads carry their domain premise into the loop premise:
    # an instance beyond the 5-entry table never iterates (any-grid proof)
    enc, reports = _t1(g, {}, disjoint, pinned=False)
    assert reports == []


def test_gather_golden_binds_a_masked_index_load_with_a_dense_other():
    """The compiled gather: ``idx = load(idx_ptr + offs, mask, other=0)``
    (a dense-constant ``other``) addresses a FLOAT source. The index is a
    Loaded with the modeled mask and ``other == Const(0)``; the float load
    it steers stays unmodeled in value. At T0 the read-only groups are
    skipped, so the kernel proves without any snapshot; at T1 a permuted
    index table proves content-qualified."""
    from pathlib import Path

    from triton_viz.clients.common.ttir_reader import Const

    golden = Path(__file__).resolve().parents[1] / "golden" / "ttgir"
    g = _mp((golden / "gather_sm80.ttir").read_text())
    idx_load, src_load, store = g.accesses
    (leaf,) = loaded_leaves(src_load.offset)
    assert leaf.base_param == "idx_ptr"
    assert leaf.mask is not None and leaf.other == Const(0)
    assert loaded_leaves(store.offset) == []
    assert [name for name, _ in encode_graph_t0(g, multipath=True)] == ["out_ptr"]
    perm = tuple(reversed(range(1024)))
    tensors = {
        "idx_ptr": _t(0x100000, 1024, snapshot=perm),
        "src_ptr": _t(0x200000, 1024),
        "out_ptr": _t(0x300000, 1024),
    }
    enc, reports = _t1(g, {"n_elements": 1024}, tensors)
    assert enc.content_qualified and enc.uncertain_event_ids == set()
    assert reports == []
