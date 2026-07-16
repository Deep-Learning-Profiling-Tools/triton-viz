"""Unit tests for scf.if condition modeling in the shared TTIR reader:
path conditions on accesses (negated for else regions, conjoined when
nested), the Select upgrade for single-result ifs, and the preserved
pessimism for unmodelable (data-dependent) conditions."""

from pathlib import Path

import pytest

from triton_viz.clients.common.ttir_reader import (
    BoolBin,
    Cmp,
    Const,
    Not,
    Pid,
    UnsupportedTTIR,
    parse_ttir,
)
from triton_viz.clients.sanitizer.compiled.oob import (
    LaunchContext,
    TensorMeta,
    check_graph,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / name).read_text()


def _meta(numel, ptr=1000):
    return TensorMeta(numel=numel, elem_bits=32, data_ptr=ptr, contiguous=True)


def _mini(*body_lines):
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


PID_IS_ZERO = Cmp("eq", Pid(0), Const(0))


@pytest.mark.parametrize("cap", ["sm80", "sm90"])
def test_pid_branch_store_carries_path(cap):
    g = parse_ttir(_read(f"pid_branch_{cap}.ttir"))
    load, store = g.accesses
    assert load.path is None and not load.guarded
    assert not store.guarded  # modeled condition ⇒ no pessimistic flag
    assert store.path == PID_IS_ZERO


def test_pid_branch_proof_uses_the_path():
    """The S2 headline: out_ptr sized for ONE block proves clean because the
    query knows only pid 0 stores — previously this was 'branch-guarded →
    unsupported'."""
    g = parse_ttir(_read("pid_branch_sm80.ttir"))
    ctx = LaunchContext(
        grid=(4, 1, 1),
        params={"n_elements": 1024},
        tensors={"x_ptr": _meta(1024), "out_ptr": _meta(256)},
    )
    assert check_graph(g, ctx) == []


def test_pid_branch_witness_is_reachable():
    g = parse_ttir(_read("pid_branch_sm80.ttir"))
    ctx = LaunchContext(
        grid=(4, 1, 1),
        params={"n_elements": 1024},
        tensors={"x_ptr": _meta(1024), "out_ptr": _meta(100)},
    )
    v = check_graph(g, ctx)
    assert len(v) == 1 and v[0].kind == "store"
    assert v[0].witness["pid_0"] == 0  # pinned by the path constraint


@pytest.mark.parametrize("cap", ["sm80", "sm90"])
def test_if_else_load_negates_the_else_path(cap):
    g = parse_ttir(_read(f"if_else_load_{cap}.ttir"))
    lx, ly, st = g.accesses
    assert (lx.kind, lx.base_param) == ("load", "x_ptr")
    assert (ly.kind, ly.base_param) == ("load", "y_ptr")
    assert lx.path == PID_IS_ZERO
    assert ly.path == Not(PID_IS_ZERO)
    assert st.path is None and not st.guarded


def test_if_else_load_per_branch_verdicts():
    g = parse_ttir(_read("if_else_load_sm80.ttir"))
    tensors = {"x_ptr": _meta(256), "y_ptr": _meta(1024), "out_ptr": _meta(1024)}
    ctx = LaunchContext(grid=(4, 1, 1), params={"n_elements": 1024}, tensors=tensors)
    assert check_graph(g, ctx) == []  # x is only touched by pid 0
    ctx_bad = LaunchContext(
        grid=(4, 1, 1),
        params={"n_elements": 1024},
        tensors={**tensors, "y_ptr": _meta(512)},
    )
    v = check_graph(g, ctx_bad)
    assert len(v) == 1 and v[0].base_param == "y_ptr"
    assert v[0].witness["pid_0"] >= 2  # only the else-branch blocks reach y


def test_if_else_offset_is_canonicalized_to_select():
    """A pure-scalar if/else never reaches the reader as scf.if: triton
    canonicalizes it to arith.select, which the vocabulary already covers.
    (The printed arg name follows the Python variable, hence 'base'.)"""
    g = parse_ttir(_read("if_else_offset_sm80.ttir"))
    ctx = LaunchContext(
        grid=(4, 1, 1),
        params={"base": 1000},
        tensors={"x_ptr": _meta(1256), "out_ptr": _meta(1256)},
    )
    assert check_graph(g, ctx) == []
    v = check_graph(
        g,
        LaunchContext(
            grid=(4, 1, 1),
            params={"base": 1000},
            tensors={"x_ptr": _meta(1000), "out_ptr": _meta(1256)},
        ),
    )
    assert v and v[0].base_param == "x_ptr"


def test_nested_if_paths_conjoin():
    g = parse_ttir(
        _mini(
            "%c0 = arith.constant 0 : i32",
            "%c1 = arith.constant 1 : i32",
            "%pid = tt.get_program_id x : i32",
            "%pidy = tt.get_program_id y : i32",
            "%a = arith.cmpi eq, %pid, %c0 : i32",
            "%b = arith.cmpi sgt, %pidy, %c1 : i32",
            "%val = arith.constant dense<0> : tensor<64xi32>",
            "scf.if %a {",
            "  scf.if %b {",
            "    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "    %p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "    %q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "    tt.store %q, %val : tensor<64x!tt.ptr<f32>>",
            "  }",
            "}",
        )
    )
    (store,) = g.accesses
    assert not store.guarded
    assert store.path == BoolBin("and", PID_IS_ZERO, Cmp("sgt", Pid(1), Const(1)))


def test_unmodelable_condition_stays_guarded_and_pessimistic():
    """A condition derived from loaded data cannot be modeled: the access
    keeps the pre-S2 behavior — guarded, no path, and a potential OOB on it
    is 'unsupported', never a witness."""
    g = parse_ttir(
        _mini(
            "%s = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>",
            "%l = tt.load %s : tensor<4x!tt.ptr<f32>>",
            "%z = arith.constant dense<0> : tensor<4xi32>",
            "%c = arith.cmpi eq, %l, %z : tensor<4xi32>",
            "%val = arith.constant dense<0> : tensor<64xi32>",
            "scf.if %c {",
            "  %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "  %p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "  %q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "  tt.store %q, %val : tensor<64x!tt.ptr<f32>>",
            "}",
        )
    )
    store = next(a for a in g.accesses if a.kind == "store")
    assert store.guarded and store.path is None
    with pytest.raises(UnsupportedTTIR, match="branch-guarded"):
        check_graph(
            g,
            LaunchContext(
                grid=(1, 1, 1),
                params={},
                tensors={"x_ptr": _meta(4), "out_ptr": _meta(10)},
            ),
        )


_SINGLE_RESULT_IF = [
    "%c0 = arith.constant 0 : i32",
    "%c9 = arith.constant 9 : i32",
    "%pid = tt.get_program_id x : i32",
    "%cnd = arith.cmpi eq, %pid, %c0 : i32",
]


def test_single_result_if_upgrades_to_select():
    """then/else regions legally REUSE SSA names, so yields must be resolved
    at the yield line; the single-result scf.if then becomes a Select."""
    g = parse_ttir(
        _mini(
            *_SINGLE_RESULT_IF,
            "%r = scf.if %cnd -> (i32) {",
            "  scf.yield %c0 : i32",
            "} else {",
            "  scf.yield %c9 : i32",
            "}",
            "%rng = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "%p0 = tt.addptr %out_ptr, %r : !tt.ptr<f32>, i32",
            "%p = tt.splat %p0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%q = tt.addptr %p, %rng : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "tt.store %q, %rng : tensor<64x!tt.ptr<f32>>",
        )
    )
    tensors = {"x_ptr": _meta(1), "out_ptr": _meta(73)}  # max offset 9+63
    ctx = LaunchContext(grid=(4, 1, 1), params={}, tensors=tensors)
    assert check_graph(g, ctx) == []
    v = check_graph(
        g,
        LaunchContext(
            grid=(4, 1, 1), params={}, tensors={**tensors, "out_ptr": _meta(72)}
        ),
    )
    assert len(v) == 1 and v[0].witness["pid_0"] != 0  # else-branch offset


def test_multi_result_if_use_fails_closed():
    with pytest.raises(UnsupportedTTIR):
        parse_ttir(
            _mini(
                *_SINGLE_RESULT_IF,
                "%r:2 = scf.if %cnd -> (i32, i32) {",
                "  scf.yield %c0, %c9 : i32, i32",
                "} else {",
                "  scf.yield %c9, %c0 : i32, i32",
                "}",
                "%rng = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
                "%p0 = tt.addptr %out_ptr, %r#0 : !tt.ptr<f32>, i32",
                "%p = tt.splat %p0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
                "%q = tt.addptr %p, %rng : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
                "tt.store %q, %rng : tensor<64x!tt.ptr<f32>>",
            )
        )
