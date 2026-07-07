"""Unit tests for the compiled sanitizer's TTIR reader."""

from pathlib import Path

import pytest

# Deliberately imports via the back-compat shim (the reader lives in
# triton_viz.clients.common.ttir_reader) so the shim surface stays covered.
from triton_viz.clients.sanitizer.compiled.ttir_reader import (
    Arange,
    Bin,
    IterArgOffset,
    UnsupportedTTIR,
    parse_ttir,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / name).read_text()


def _mini(*body_lines):
    """Wrap op lines in a minimal parseable TTIR module."""
    body = "\n    ".join(body_lines)
    return (
        "module {\n"
        "  tt.func public @k(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>)"
        " attributes {noinline = false} {\n"
        f"    {body}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


def test_add_kernel_access_graph():
    g = parse_ttir(_read("add_sm80.ttir"))
    assert g.kernel_name == "add_kernel"
    names = [a.name for a in g.func_args]
    assert names == ["x_ptr", "y_ptr", "out_ptr", "n_elements"]
    assert g.arg("x_ptr").is_ptr and g.arg("x_ptr").elem_bits == 32
    assert not g.arg("n_elements").is_ptr
    assert g.loop is None
    assert [a.kind for a in g.accesses] == ["load", "load", "store"]
    assert {a.base_param for a in g.accesses} == {"x_ptr", "y_ptr", "out_ptr"}
    assert all(a.mask is not None for a in g.accesses)  # offs < n_elements
    assert all(a.loc is not None for a in g.accesses)


def test_matmul_loop_and_iter_args():
    g = parse_ttir(_read("matmul_s3_sm80.ttir"))
    assert g.kernel_name == "matmul_kernel"
    assert g.loop is not None
    # cdiv(K, 32): upper = (K + 31) // 32
    assert isinstance(g.loop.upper, type(g.loop.upper))
    # Two pointer iter_args (a_ptrs, b_ptrs) advance monotonically.
    assert set(g.iter_args) == {0, 1}
    assert g.iter_args[0].base_param == "a_ptr"
    assert g.iter_args[1].base_param == "b_ptr"
    # A and B loads use the loop-carried pointer offset.
    loop_loads = [a for a in g.accesses if isinstance(a.offset, IterArgOffset)]
    assert len(loop_loads) == 2


def test_matmul_2d_arange_dims_separated():
    """The C store reuses one make_range for row and column; the reader must
    tag them with distinct dims so they don't collapse."""
    g = parse_ttir(_read("matmul_s3_sm80.ttir"))
    store = next(a for a in g.accesses if a.kind == "store")
    dims = set()

    def walk(t):
        if isinstance(t, Arange):
            dims.add((t.ssa, t.dim))
        for f in ("a", "b", "cond", "t", "f"):
            if hasattr(t, f):
                walk(getattr(t, f))

    walk(store.offset)
    # The same ssa appears under two different dims (row=0, col=1).
    reused = {ssa for ssa, _ in dims if sum(1 for s, _ in dims if s == ssa) > 1}
    assert reused, f"expected a reused make_range across dims, got {dims}"
    for ssa in reused:
        assert {d for s, d in dims if s == ssa} == {0, 1}


def test_2d_tile_distinct_ranges_have_dims():
    g = parse_ttir(_read("tile2d_sm80.ttir"))
    load = next(a for a in g.accesses if a.kind == "load")
    found = []

    def walk(t):
        if isinstance(t, Arange):
            found.append(t.dim)
        for f in ("a", "b", "cond", "t", "f"):
            if hasattr(t, f):
                walk(getattr(t, f))

    walk(load.offset)
    assert set(found) == {0, 1}  # one row dim, one col dim


def test_indirect_gather_is_unsupported():
    with pytest.raises(UnsupportedTTIR, match="data-dependent"):
        parse_ttir(_read("gather_sm80.ttir"))


def test_scalar_params_stay_symbolic():
    g = parse_ttir(_read("matmul_s3_sm80.ttir"))
    # The loop upper bound references the K argument as a Param (substituted
    # per launch), not a baked constant.
    text = repr(g.loop.upper)
    assert "Param(name='K')" in text


def test_block_pointer_kernel_is_unsupported():
    text = _read("add_sm80.ttir").replace(
        "%offs_0 = tt.make_range",
        "%bp = tt.make_block_ptr\n    %offs_0 = tt.make_range",
    )
    with pytest.raises(UnsupportedTTIR, match="block pointer"):
        parse_ttir(text)


def test_non_ttir_input_is_unsupported():
    with pytest.raises(UnsupportedTTIR, match="no tt.func"):
        parse_ttir("garbage\n.version 8.0\n")


def test_unrecognized_store_syntax_fails_closed():
    """A tt.store the store regex does not match (here: an attribute dict
    before the ':') must raise, not be silently dropped. A store has no SSA
    result, so without the fail-closed guard it would fall through unrecorded
    and check_graph would prove "ok" while a real write went unchecked."""
    text = _read("add_sm80.ttir").replace(
        "tt.store %1, %2, %mask_3 :",
        "tt.store %1, %2, %mask_3 {cache = 1 : i32} :",
    )
    with pytest.raises(UnsupportedTTIR, match="unsupported memory op"):
        parse_ttir(text)


def test_min_max_rem_arith_are_modeled():
    """The tutorial-matmul grouped swizzle lowers to arith.minsi / arith.remsi
    over the program id. These are launch-affine, not data-dependent: they
    must parse into Bin terms, not poison every downstream address as DataDep
    (which made fully-affine kernels like TritonBench's matmul_triton2 abstain
    as unsupported instead of catching their real OOB)."""
    text = _read("add_sm80.ttir").replace(
        "%offs = arith.muli %pid, %c1024_i32 : i32 loc(#loc20)",
        "%g = arith.remsi %pid, %c1024_i32 : i32 loc(#loc20)\n"
        "    %h = arith.minsi %g, %c1024_i32 : i32 loc(#loc20)\n"
        "    %i = arith.maxsi %h, %c1024_i32 : i32 loc(#loc20)\n"
        "    %offs = arith.muli %i, %c1024_i32 : i32 loc(#loc20)",
    )
    g = parse_ttir(text)
    ops = set()

    def walk(t):
        if isinstance(t, Bin):
            ops.add(t.op)
        for f in ("a", "b", "cond", "t", "f"):
            if hasattr(t, f):
                walk(getattr(t, f))

    walk(g.accesses[0].offset)
    assert {"%", "min", "max"} <= ops, f"swizzle ops lost in parsing: {ops}"


def test_scf_if_marks_accesses_guarded():
    """An access inside an scf.if region executes only when the branch is
    taken. The condition is not modeled, so the access must be tagged
    ``guarded`` — check_graph may then use it for a proof but never as a SAT
    witness (TritonBench's diag_ssm guards `load(y + offs - B*D)` behind
    `if t > 0`; an unguarded scan would "witness" the unreachable t == 0 at
    offset -1). Accesses outside the region keep full witness validity."""
    text = _read("add_sm80.ttir").replace(
        "%x_5 = tt.load %x_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc25)",
        "%c = arith.cmpi sgt, %offs, %c1024_i32 : i32 loc(#loc25)\n"
        "    %x_5 = scf.if %c -> (tensor<1024xf32>) { loc(#loc25)\n"
        "      %inner = tt.load %x_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc25)\n"
        "      scf.yield %inner : tensor<1024xf32> loc(#loc25)\n"
        "    } loc(#loc25)",
    )
    g = parse_ttir(text)
    by_param = {(a.base_param, a.kind): a.guarded for a in g.accesses}
    assert by_param[("x_ptr", "load")] is True  # inside the scf.if
    assert by_param[("y_ptr", "load")] is False  # after the region closed
    assert by_param[("out_ptr", "store")] is False


def test_store_of_multi_result_value_is_recorded():
    """`tt.store %ptrs, %acc#2, %mask` stores the third result of a
    multi-result scf.for (a matmul accumulator stored without a truncf in
    between). The stored VALUE plays no part in address math, so the store
    must be recorded and checked — not fail closed on the `#2` token."""
    text = _read("add_sm80.ttir").replace(
        "tt.store %1, %2, %mask_3 :",
        "tt.store %1, %acc#2, %mask_3 :",
    )
    g = parse_ttir(text)
    store = next(a for a in g.accesses if a.kind == "store")
    assert store.base_param == "out_ptr"
    assert store.mask is not None  # the mask operand still parsed


def test_malformed_atomic_syntax_fails_closed():
    """Well-formed tt.atomic_rmw/cas parse into AccessEvents (see
    test_ttir_reader_atomics.py); an atomic line the regexes do NOT match
    (here: missing sem/scope operands) must still be reported unsupported,
    not become an unchecked DataDep result that lets the rest of the kernel
    prove in-bounds."""
    text = _read("add_sm80.ttir").replace(
        "tt.store %1, %2, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc13)",
        "%atom = tt.atomic_rmw fadd, %1, %2, %mask_3 : tensor<1024xf32> loc(#loc13)",
    )
    with pytest.raises(UnsupportedTTIR, match="unsupported memory op"):
        parse_ttir(text)


def test_bitwise_andi_on_wide_ints_fails_closed_in_addresses():
    """arith.andi on non-i1 integers is BITWISE math, not boolean logic;
    modeled as And/Or it would collapse ``offs & 8`` (footprint {0, 8}) to a
    {0, 1} truth value — a false in-bounds proof. It must degrade to DataDep
    so an address use fails closed."""
    text = _mini(
        "%c8 = arith.constant dense<8> : tensor<64xi32>",
        "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "%a = arith.andi %r, %c8 : tensor<64xi32>",
        "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%q = tt.addptr %p, %a : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "%s = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%v = tt.load %s : tensor<64x!tt.ptr<f32>>",
        "tt.store %q, %v : tensor<64x!tt.ptr<f32>>",
    )
    with pytest.raises(UnsupportedTTIR):
        parse_ttir(text)


def test_descriptor_memory_ops_fail_closed():
    """tt.descriptor_* are real global accesses outside the vocabulary; they
    must not fall through as an unmodeled-op DataDep while check_graph proves
    'ok' without having checked them."""
    text = _mini("%w = tt.descriptor_load %x_ptr : !tt.whatever")
    with pytest.raises(UnsupportedTTIR, match="unsupported memory op"):
        parse_ttir(text)


def test_unknown_program_id_axis_fails_closed():
    """Printer drift in the pid axis must surface as UnsupportedTTIR, not a
    bare KeyError escaping into the client's launch teardown."""
    text = _mini("%pid = tt.get_program_id q : i32")
    with pytest.raises(UnsupportedTTIR, match="program-id axis"):
        parse_ttir(text)
