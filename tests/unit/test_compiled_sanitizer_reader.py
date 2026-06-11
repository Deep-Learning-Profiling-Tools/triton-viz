"""Unit tests for the compiled sanitizer's TTIR reader."""

from pathlib import Path

import pytest

from triton_viz.clients.sanitizer.compiled.ttir_reader import (
    Arange,
    IterArgOffset,
    UnsupportedTTIR,
    parse_ttir,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / name).read_text()


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


def test_atomic_op_fails_closed():
    """Atomics are real memory accesses the v1 model does not check. They must
    be reported unsupported, not become an unchecked DataDep result that lets
    the rest of the kernel still prove in-bounds."""
    text = _read("add_sm80.ttir").replace(
        "tt.store %1, %2, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc13)",
        "%atom = tt.atomic_rmw fadd, %1, %2, %mask_3 : tensor<1024xf32> loc(#loc13)",
    )
    with pytest.raises(UnsupportedTTIR, match="unsupported memory op"):
        parse_ttir(text)
