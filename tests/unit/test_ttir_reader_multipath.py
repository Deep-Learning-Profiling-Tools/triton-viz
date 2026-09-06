"""Reader-level tests for multipath capture (Route 3, the ladder's L2).

``parse_ttir(text, multipath=True)`` lifts two structural boundaries of the
single-path model and is byte-identical otherwise:

  * the unstructured ``cf.*`` graph Triton emits for an ``if`` that
    contains a ``return`` (early-exit guards) gets BLOCK PATH PREDICATES:
    every access conjoins its block's predicate into ``path``, and block
    parameters (the merge values of ``visit_if_top_level``) bind to a
    Select over the incoming edges;
  * every ``scf.for`` gets its own induction variable (nested, sequential,
    under an scf.if or a block predicate).

The goldens come from ``tests/golden/ttgir/generate_golden.py multipath``
(Triton 3.6, sm80). Each test names the single-path refusal it replaces
and checks that refusal still fires without the flag: every L2 code path
starts at a single-path refusal site.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from triton_viz.clients.common.ttir_reader import (
    Arange,
    Bin,
    loaded_leaves,
    BoolBin,
    Cmp,
    Const,
    Not,
    Param,
    Pid,
    Select,
    UnsupportedTTIR,
    parse_ttir,
)

from .test_t1_rmw_static import _module

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / f"{name}_sm80.ttir").read_text()


PID_GUARD = Cmp("sge", Bin("*", Pid(0), Const(64)), Param("T"))


# ───────────────────── single-path behavior is untouched ─────────────────────


@pytest.mark.parametrize(
    "name, kind, needle",
    [
        ("early_return_pid", "control-flow", "cf.cond_br is unsupported"),
        ("early_return_loaded", "control-flow", "cf.cond_br is unsupported"),
        ("nested_guard_merge", "control-flow", "cf.cond_br is unsupported"),
        ("guard_then_loop", "control-flow", "cf.cond_br is unsupported"),
        ("nested_loops", "nested-loop", "multiple/nested loops"),
        ("sequential_loops", "nested-loop", "multiple/nested loops"),
        ("loop_under_if", "control-flow", "multiple/nested loops"),
    ],
)
def test_single_path_refusals_unchanged(name, kind, needle):
    """Without the flag every multipath shape refuses exactly as the
    pinned run recorded it (the static reason families of the paper)."""
    with pytest.raises(UnsupportedTTIR, match=needle) as ei:
        parse_ttir(_read(name))
    assert ei.value.kind == kind


def test_single_loop_kernel_parses_identically_in_both_modes():
    """A kernel that reaches no refusal site gets the same graph: the
    multipath walk only diverges at the cf.* and second-loop raises."""
    a = parse_ttir(_read("grid_stride"))
    b = parse_ttir(_read("grid_stride"), multipath=True)
    assert not a.multipath and b.multipath
    assert replace(a, multipath=True) == b
    assert a.loop is not None and a.loop.loop_ssa == "%loop"
    assert a.loops == [a.loop]
    assert all(x.loops == ("%loop",) and x.in_loop for x in a.accesses)


# ───────────────────── block path predicates ─────────────────────


def test_early_return_guard_becomes_a_path_predicate():
    """``if pid * BLOCK >= T: return``: the fall-through block's accesses
    carry the NEGATED guard as their path, exactly like an scf.if else
    region; nothing is widened."""
    g = parse_ttir(_read("early_return_pid"), multipath=True)
    assert g.cf_blocks == 2 and g.loops == [] and g.loop is None
    load, store = g.accesses
    for a in (load, store):
        assert not a.guarded
        assert a.mask is not None  # the offs < n mask is still there
        assert a.path == Not(PID_GUARD)
        assert a.loops == () and not a.in_loop


def test_loaded_value_guard_is_a_loaded_term_in_the_path():
    """``if y == -1: return`` with y loaded: under the L2 reader mode the
    guard is modeled through a Loaded term (Route 2); whether it is exact
    or free is the encoder's call (a snapshot of idx_ptr, or not)."""
    g = parse_ttir(_read("early_return_loaded"), multipath=True)
    idx_load, load, store = g.accesses
    assert not idx_load.guarded and idx_load.path is None
    for a in (load, store):
        assert not a.guarded
        assert isinstance(a.path, Not) and isinstance(a.path.a, Cmp)
        (leaf,) = loaded_leaves(a.path)
        assert leaf.access_index == 0 and leaf.base_param == "idx_ptr"
        assert leaf.mask is None and leaf.other is None


def test_nested_guards_merge_through_a_select():
    """Two guards and a merge block with a parameter: the merge block's
    predicate is the disjunction of its two incoming edge paths, and the
    parameter (``base``) binds to a Select on the edge that supplies the
    then-value, with the else-value as the fallback."""
    g = parse_ttir(_read("nested_guard_merge"), multipath=True)
    assert g.cf_blocks == 5
    load, store = g.accesses
    assert not load.guarded and not store.guarded
    path = store.path
    assert isinstance(path, BoolBin) and path.op == "or"
    # both arms reach the merge only past the outer guard `pid >= T`
    outer = Not(Cmp("sge", Pid(0), Param("T")))
    then_edge = BoolBin(
        "and",
        BoolBin("and", outer, Cmp("eq", Pid(0), Const(0))),
        Not(Cmp("slt", Param("n"), Const(0))),
    )
    else_edge = BoolBin("and", outer, Not(Cmp("eq", Pid(0), Const(0))))
    assert path == BoolBin("or", then_edge, else_edge)

    # the merged base: Select(then-edge, 0, pid*64 + n)
    def find_select(t):
        if isinstance(t, Select):
            return t
        for attr in ("a", "b", "cond", "t", "f"):
            sub = getattr(t, attr, None)
            if sub is not None:
                s = find_select(sub)
                if s is not None:
                    return s
        return None

    sel = find_select(store.offset)
    assert sel is not None
    assert sel.cond == then_edge
    assert sel.t == Const(0)
    assert sel.f == Bin("+", Bin("*", Pid(0), Const(64)), Param("n"))
    assert load.offset == store.offset


def test_shared_return_block_before_its_later_predecessor_is_tolerated():
    """Triton's canonicalized lowering routes the inner ``return`` of
    nested_guard_merge to the outer guard's return block (^bb1 has a
    predecessor, ^bb3, printed after it). A return-only block needs no
    predicate, so the parse succeeds (previous test); a block WITH an
    access in that position refuses by name."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %T: i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi sge, %pid, %T : i32",
        "cf.cond_br %g, ^bb1, ^bb2",
        "^bb1:  // 2 preds: ^bb0, ^bb2",
        "%oa = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1 : !tt.ptr<i32>",
        "tt.return",
        "^bb2:  // pred: ^bb0",
        "cf.br ^bb1",
    )
    with pytest.raises(UnsupportedTTIR, match="entered from a later block") as ei:
        parse_ttir(text, multipath=True)
    assert ei.value.kind == "control-flow"


def test_cyclic_cf_graph_refuses_by_name():
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %T: i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "cf.br ^bb1",
        "^bb1:  // 2 preds: ^bb0, ^bb1",
        "%g = arith.cmpi sge, %pid, %T : i32",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
        "cf.cond_br %g, ^bb1, ^bb2",
        "^bb2:  // pred: ^bb1",
    )
    with pytest.raises(UnsupportedTTIR, match="cyclic") as ei:
        parse_ttir(text, multipath=True)
    assert ei.value.kind == "control-flow"


def test_cf_inside_an_scf_region_refuses():
    """Triton never emits cf.* inside a loop (a return there is a compile
    error); a hand-made one must not be flat-scanned."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%g = arith.cmpi sge, %k, %n : i32",
        "cf.cond_br %g, ^bb1, ^bb2",
        "^bb1:",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
        "^bb2:",
        "scf.yield",
        "}",
    )
    with pytest.raises(UnsupportedTTIR, match="inside an scf region"):
        parse_ttir(text, multipath=True)


def test_block_arguments_of_pointer_type_merge_offsets():
    """A merged POINTER (same base, two offsets) binds to a PtrValue with a
    Select offset; a merge of two different bases stays DataDep and an
    address use fails closed."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %o2_ptr: !tt.ptr<i32>, %T: i32",
        "%c1 = arith.constant 1 : i32",
        "%c8 = arith.constant 8 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi eq, %pid, %T : i32",
        "%pa = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "%pb = tt.addptr %out_ptr, %c8 : !tt.ptr<i32>, i32",
        "cf.cond_br %g, ^bb1(%pa : !tt.ptr<i32>), ^bb2(%pb : !tt.ptr<i32>)",
        "^bb1(%p: !tt.ptr<i32>):  // pred: ^bb0",
        "tt.store %p, %c1 : !tt.ptr<i32>",
        "tt.return",
        "^bb2(%q: !tt.ptr<i32>):  // pred: ^bb0",
        "tt.store %q, %c1 : !tt.ptr<i32>",
        "tt.return",
    )
    g = parse_ttir(text, multipath=True)
    s1, s2 = g.accesses
    assert s1.base_param == "out_ptr" and s1.path == Cmp("eq", Pid(0), Param("T"))
    assert s2.base_param == "out_ptr" and s2.path == Not(Cmp("eq", Pid(0), Param("T")))
    # single-edge parameters bind to the edge's value directly
    assert s1.offset == Bin("+", Const(0), Pid(0))
    assert s2.offset == Bin("+", Const(0), Const(8))

    mixed = text.replace(
        "^bb2(%pb : !tt.ptr<i32>)", "^bb2(%pc : !tt.ptr<i32>)"
    ).replace(
        "%pb = tt.addptr %out_ptr, %c8 : !tt.ptr<i32>, i32",
        "%pb = tt.addptr %out_ptr, %c8 : !tt.ptr<i32>, i32\n"
        "    %pc = tt.addptr %o2_ptr, %c8 : !tt.ptr<i32>, i32",
    )
    # a two-edge merge of DIFFERENT bases: build one by routing both edges
    # into one block
    two_edge = _module(
        "%out_ptr: !tt.ptr<i32>, %o2_ptr: !tt.ptr<i32>, %T: i32",
        "%c1 = arith.constant 1 : i32",
        "%c8 = arith.constant 8 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi eq, %pid, %T : i32",
        "%pa = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "%pc = tt.addptr %o2_ptr, %c8 : !tt.ptr<i32>, i32",
        "cf.cond_br %g, ^bb1(%pa : !tt.ptr<i32>), ^bb1(%pc : !tt.ptr<i32>)",
        "^bb1(%p: !tt.ptr<i32>):  // 2 preds: ^bb0, ^bb0",
        "tt.store %p, %c1 : !tt.ptr<i32>",
        "tt.return",
    )
    assert parse_ttir(mixed, multipath=True).accesses[1].base_param == "o2_ptr"
    with pytest.raises(UnsupportedTTIR, match="non-pointer") as ei:
        parse_ttir(two_edge, multipath=True)
    assert ei.value.kind == "other"


# ───────────────────── several loops ─────────────────────


def test_guard_then_loop_carries_both_predicate_and_iterator():
    g = parse_ttir(_read("guard_then_loop"), multipath=True)
    assert g.cf_blocks == 2
    assert g.loop is not None and g.loop.loop_ssa == "%loop"
    assert [lp.induction_var for lp in g.loops] == ["%k"]
    for a in g.accesses:
        assert a.path == Not(PID_GUARD)
        assert a.loops == ("%loop",) and a.in_loop


def test_nested_loops_get_two_iterators_outer_first():
    g = parse_ttir(_read("nested_loops"), multipath=True)
    assert [lp.induction_var for lp in g.loops] == ["%i", "%j"]
    outer, inner = g.loops
    assert outer.loop_ssa == "%loop" and inner.loop_ssa.startswith("%loop@")
    assert g.loop is None  # the single-loop consumers must not read one
    for a in g.accesses:
        assert a.loops == (outer.loop_ssa, inner.loop_ssa)
        assert a.path is None and not a.guarded


def test_sequential_loops_are_independent():
    g = parse_ttir(_read("sequential_loops"), multipath=True)
    assert [lp.induction_var for lp in g.loops] == ["%i", "%j"]
    first, second = g.loops
    assert first.loop_ssa == "%acc_3"  # the loop with a result keeps its name
    assert g.loop is None
    load, store = g.accesses
    assert load.loops == (first.loop_ssa,)
    assert store.loops == (second.loop_ssa,)


def test_loop_under_scf_if_carries_the_condition():
    g = parse_ttir(_read("loop_under_if"), multipath=True)
    assert len(g.loops) == 1 and g.loop is g.loops[0]
    for a in g.accesses:
        assert a.path == Cmp("eq", Pid(0), Const(0))
        assert a.loops == (g.loop.loop_ssa,)
        assert not a.guarded


def test_nested_loop_iter_args_belong_to_their_loop():
    """Pointer iter_args of an inner loop advance with the INNER iterator
    from an outer-iterator-dependent start."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32, %m: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%c8 = arith.constant 8 : i32",
        "%pid = tt.get_program_id x : i32",
        "%p0 = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "%po = scf.for %i = %c0 to %n step %c1 iter_args(%pa = %p0) -> (!tt.ptr<i32>)  : i32 {",
        "%pi = scf.for %j = %c0 to %m step %c1 iter_args(%pb = %pa) -> (!tt.ptr<i32>)  : i32 {",
        "tt.store %pb, %c1 : !tt.ptr<i32>",
        "%pb2 = tt.addptr %pb, %c1 : !tt.ptr<i32>, i32",
        "scf.yield %pb2 : !tt.ptr<i32>",
        "}",
        "%pa2 = tt.addptr %pa, %c8 : !tt.ptr<i32>, i32",
        "scf.yield %pa2 : !tt.ptr<i32>",
        "}",
    )
    g = parse_ttir(text, multipath=True)
    outer, inner = g.loops
    assert outer.loop_ssa == "%po" and inner.loop_ssa.startswith("%pi@")
    outer_arg, inner_arg = g.iter_args[0], g.iter_args[1]
    assert outer_arg.loop_ssa == outer.loop_ssa and outer_arg.delta == Const(8)
    assert inner_arg.loop_ssa == inner.loop_ssa and inner_arg.delta == Const(1)
    (store,) = g.accesses
    assert store.loops == (outer.loop_ssa, inner.loop_ssa)


def test_reduce_combine_blocks_are_not_the_cf_graph():
    """``tt.reduce`` (and scan) carry an anonymous region with their own
    ``^bb0(...)`` combine block; inside a loop or not, those labels are
    ignored exactly as in single-path, never mistaken for cf.* blocks."""
    text = _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%xp = tt.addptr %xs, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "%v = tt.load %xp : tensor<64x!tt.ptr<f32>>",
        '%s = "tt.reduce"(%v) <{axis = 0 : i32}> ({',
        "^bb0(%a: f32, %b: f32):",
        "%m = arith.addf %a, %b : f32",
        "tt.reduce.return %m : f32",
        "}) : (tensor<64xf32>) -> f32",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<f32>, i32",
        "tt.store %op, %s : !tt.ptr<f32>",
        "scf.yield",
        "}",
    )
    a = parse_ttir(text)
    b = parse_ttir(text, multipath=True)
    assert replace(a, multipath=True) == b
    assert b.cf_blocks == 0 and [x.kind for x in b.accesses] == ["load", "store"]


def test_attribute_dict_region_close_pops_the_loop_frame():
    """``} {tt.num_stages = 1 : i32} loc(...)`` closes a pipelined loop.
    Before the fix the frame stayed open until the function's close, so a
    loop inside an scf.if then tripped "unexpected `else`" (two aiter
    attention rows) and an access after the loop would have been read as
    in-loop."""
    text = _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %g {",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%xp = tt.addptr %x_ptr, %k : !tt.ptr<f32>, i32",
        "%v = tt.load %xp : !tt.ptr<f32>",
        "tt.store %out_ptr, %v : !tt.ptr<f32>",
        "} {tt.num_stages = 2 : i32}",
        "} else {",
        "%c2 = arith.constant 2.0 : f32",
        "tt.store %out_ptr, %c2 : !tt.ptr<f32>",
        "}",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<f32>, i32",
        "%c3 = arith.constant 3.0 : f32",
        "tt.store %op, %c3 : !tt.ptr<f32>",
    )
    for mp in (False, True):
        if not mp:
            with pytest.raises(UnsupportedTTIR, match="multiple/nested loops"):
                parse_ttir(text)
            continue
        g = parse_ttir(text, multipath=True)
        assert [a.kind for a in g.accesses] == ["load", "store", "store", "store"]
        assert g.accesses[1].loops == (g.loops[0].loop_ssa,)
        assert g.accesses[2].loops == () and g.accesses[2].path == Not(
            Cmp("eq", Pid(0), Const(0))
        )
        assert g.accesses[3].loops == () and g.accesses[3].path is None


def test_mixed_and_mask_keeps_its_modelable_conjunct_at_l2():
    """``tl.store(p, v, mask=bounds and loaded_guard)`` where the guard
    comes from a FLOAT load (not a Loaded term): single-path drops the
    whole mask; multipath keeps ``bounds`` (a sound over-approximation of
    ``bounds ∧ guard``) and the access stays widened. An integer guard is
    a Loaded term instead (Route 2) and the mask is fully modeled."""
    text = _module(
        "%out_ptr: !tt.ptr<f32>, %tgt_ptr: !tt.ptr<f32>, %C: i32",
        "%c1 = arith.constant 1.0 : f32",
        "%cm1 = arith.constant -1.0 : f32",
        "%pid = tt.get_program_id x : i32",
        "%offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>",
        "%Cs = tt.splat %C : i32 -> tensor<256xi32>",
        "%bounds = arith.cmpi slt, %offs, %Cs : tensor<256xi32>",
        "%tp = tt.addptr %tgt_ptr, %pid : !tt.ptr<f32>, i32",
        "%tgt = tt.load %tp : !tt.ptr<f32>",
        "%g = arith.cmpf one, %tgt, %cm1 : f32",
        "%gs = tt.splat %g : i1 -> tensor<256xi1>",
        "%m = arith.andi %bounds, %gs : tensor<256xi1>",
        "%row = arith.muli %pid, %C : i32",
        "%rs = tt.splat %row : i32 -> tensor<256xi32>",
        "%o = arith.addi %rs, %offs : tensor<256xi32>",
        "%ps = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>",
        "%p = tt.addptr %ps, %o : tensor<256x!tt.ptr<f32>>, tensor<256xi32>",
        "%vs = tt.splat %c1 : f32 -> tensor<256xf32>",
        "tt.store %p, %vs, %m : tensor<256x!tt.ptr<f32>>",
    )
    single = parse_ttir(text)
    (s0,) = [a for a in single.accesses if a.kind == "store"]
    assert s0.mask_dropped and s0.mask is None
    multi = parse_ttir(text, multipath=True)
    (s2,) = [a for a in multi.accesses if a.kind == "store"]
    assert s2.mask_dropped
    assert s2.mask == Cmp("slt", Arange("%offs", 0, 256), Param("C"))
    integer = (
        text.replace("%tgt_ptr: !tt.ptr<f32>", "%tgt_ptr: !tt.ptr<i32>")
        .replace(
            "%tp = tt.addptr %tgt_ptr, %pid : !tt.ptr<f32>, i32",
            "%tp = tt.addptr %tgt_ptr, %pid : !tt.ptr<i32>, i32",
        )
        .replace(
            "%tgt = tt.load %tp : !tt.ptr<f32>", "%tgt = tt.load %tp : !tt.ptr<i32>"
        )
        .replace("%cm1 = arith.constant -1.0 : f32", "%cm1 = arith.constant -1 : i32")
        .replace(
            "%g = arith.cmpf one, %tgt, %cm1 : f32",
            "%g = arith.cmpi ne, %tgt, %cm1 : i32",
        )
    )
    (s3,) = [
        a for a in parse_ttir(integer, multipath=True).accesses if a.kind == "store"
    ]
    assert not s3.mask_dropped and len(loaded_leaves(s3.mask)) == 1


def test_attribute_dict_close_in_single_path_ends_the_loop():
    """Single-path side of the region-close fix: an access AFTER a loop
    that closes with an attribute dict is no longer read as in-loop, and
    the loop itself is recorded."""
    text = _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "scf.for %k = %c0 to %n step %c1  : i32 {",
        "%xp = tt.addptr %x_ptr, %k : !tt.ptr<f32>, i32",
        "%v = tt.load %xp : !tt.ptr<f32>",
        "tt.store %out_ptr, %v : !tt.ptr<f32>",
        "} {tt.num_stages = 2 : i32}",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<f32>, i32",
        "%c3 = arith.constant 3.0 : f32",
        "tt.store %op, %c3 : !tt.ptr<f32>",
    )
    for mp in (False, True):
        g = parse_ttir(text, multipath=mp)
        assert g.loop is not None and g.loop.induction_var == "%k"
        load, store_in, store_after = g.accesses
        assert load.in_loop and store_in.in_loop
        assert not store_after.in_loop and store_after.loops == ()


def test_kept_conjunct_follows_expand_dims():
    """A mixed ``and`` computed on 1-D lanes and then expanded to a 2-D
    tile (the guard from a FLOAT load, so it is not a Loaded term): the
    kept conjunct's Arange must be retagged with the tile dimension like
    the address's, or the partial mask would constrain a lane variable
    the address never uses."""
    text = _module(
        "%out_ptr: !tt.ptr<f32>, %tgt_ptr: !tt.ptr<f32>, %C: i32",
        "%c1 = arith.constant 1.0 : f32",
        "%cm1 = arith.constant -1.0 : f32",
        "%pid = tt.get_program_id x : i32",
        "%offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>",
        "%Cs = tt.splat %C : i32 -> tensor<8xi32>",
        "%bounds = arith.cmpi slt, %offs, %Cs : tensor<8xi32>",
        "%tp = tt.addptr %tgt_ptr, %pid : !tt.ptr<f32>, i32",
        "%tgt = tt.load %tp : !tt.ptr<f32>",
        "%g = arith.cmpf one, %tgt, %cm1 : f32",
        "%gs = tt.splat %g : i1 -> tensor<8xi1>",
        "%m1 = arith.andi %bounds, %gs : tensor<8xi1>",
        "%m2 = tt.expand_dims %m1 {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>",
        "%o1 = tt.expand_dims %offs {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>",
        "%ps = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>",
        "%p = tt.addptr %ps, %o1 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>",
        "%vs = tt.splat %c1 : f32 -> tensor<8x1xf32>",
        "tt.store %p, %vs, %m2 : tensor<8x1x!tt.ptr<f32>>",
    )
    (store,) = [
        a for a in parse_ttir(text, multipath=True).accesses if a.kind == "store"
    ]
    assert store.mask_dropped
    assert store.mask == Cmp("slt", Arange("%offs", 0, 8, 0), Param("C"))
    assert store.offset == Bin("+", Const(0), Arange("%offs", 0, 8, 0))


def _sibling_result_loops():
    """Two loops WITH results in the two arms of one scf.if. MLIR restarts
    value numbering per region, so both print as ``%acc``."""
    return _module(
        "%out_ptr: !tt.ptr<i32>, %n: i32, %m: i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%c4 = arith.constant 4 : i32",
        "%pid = tt.get_program_id x : i32",
        "%g = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %g {",
        "%acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %c0) -> (i32)  : i32 {",
        "%p = tt.addptr %out_ptr, %i : !tt.ptr<i32>, i32",
        "tt.store %p, %c1 : !tt.ptr<i32>",
        "scf.yield %a : i32",
        "}",
        "} else {",
        "%acc = scf.for %i = %c0 to %m step %c1 iter_args(%a = %c0) -> (i32)  : i32 {",
        "%o = arith.addi %i, %c4 : i32",
        "%o2 = arith.addi %o, %pid : i32",
        "%q = tt.addptr %out_ptr, %o2 : !tt.ptr<i32>, i32",
        "tt.store %q, %c1 : !tt.ptr<i32>",
        "scf.yield %a : i32",
        "}",
        "}",
    )


def test_result_loops_in_sibling_regions_stay_distinct():
    g = parse_ttir(_sibling_result_loops(), multipath=True)
    a, b = g.loops
    assert a.loop_ssa == "%acc" and b.loop_ssa.startswith("%acc@")
    assert a.upper == Param("n") and b.upper == Param("m")
    then_store, else_store = g.accesses
    assert then_store.loops == (a.loop_ssa,)
    assert else_store.loops == (b.loop_ssa,)


def _mixed_guard(pointee: str, cmp: str, const: str):
    return _module(
        f"%out_ptr: !tt.ptr<i32>, %idx_ptr: !tt.ptr<{pointee}>, %T: i32",
        "%c1 = arith.constant 1 : i32",
        f"%cm1 = arith.constant {const}",
        "%pid = tt.get_program_id x : i32",
        "%a = arith.cmpi sge, %pid, %T : i32",
        f"%ip = tt.addptr %idx_ptr, %pid : !tt.ptr<{pointee}>, i32",
        f"%y = tt.load %ip : !tt.ptr<{pointee}>",
        f"%d = {cmp}",
        "%c = arith.andi %a, %d : i1",
        "cf.cond_br %c, ^bb1, ^bb2",
        "^bb1:  // pred: ^bb0",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
        "tt.return",
        "^bb2:  // pred: ^bb0",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %op, %c1 : !tt.ptr<i32>",
        "tt.return",
    )


def test_mixed_pid_and_loaded_guard_widens_both_arms():
    """``if pid >= T and y == -1: return`` with y a FLOAT load (not a
    Loaded term): the false edge (not (a and d)) does not imply (not a),
    so the kept conjunct of the mixed ``and`` must NOT become an edge
    condition; both targets stay reachable and widened. With an integer
    load the whole condition is a modeled term (Route 2) and both arms
    carry it exactly."""
    g = parse_ttir(
        _mixed_guard("f32", "arith.cmpf oeq, %y, %cm1 : f32", "-1.0 : f32"),
        multipath=True,
    )
    _load, s1, s2 = g.accesses
    assert s1.guarded and s1.path is None
    assert s2.guarded and s2.path is None
    g = parse_ttir(
        _mixed_guard("i32", "arith.cmpi eq, %y, %cm1 : i32", "-1 : i32"),
        multipath=True,
    )
    _load, s1, s2 = g.accesses
    assert not s1.guarded and not s2.guarded
    assert len(loaded_leaves(s1.path)) == 1 and len(loaded_leaves(s2.path)) == 1
