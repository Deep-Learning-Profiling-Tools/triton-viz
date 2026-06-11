"""Unit tests for the compiled sanitizer's OOB query engine."""

from pathlib import Path

from triton_viz.clients.sanitizer.compiled.oob import (
    LaunchContext,
    TensorMeta,
    check_graph,
)
from triton_viz.clients.sanitizer.compiled.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    FuncArg,
    parse_ttir,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _meta(numel, elem_bits=32, ptr=1000):
    return TensorMeta(numel=numel, elem_bits=elem_bits, data_ptr=ptr, contiguous=True)


# ──────────────────────── add (1D, masked) ────────────────────────


def test_add_correct_is_in_bounds():
    g = parse_ttir((GOLDEN / "add_sm80.ttir").read_text())
    ctx = LaunchContext(
        grid=(4, 1, 1),
        params={"n_elements": 4096},
        tensors={n: _meta(4096) for n in ("x_ptr", "y_ptr", "out_ptr")},
    )
    assert check_graph(g, ctx) == []


def test_add_unmasked_tail_is_oob():
    """If the launch's mask bound (n_elements) exceeds the tensor, the last
    block's masked load/store still reaches past the end."""
    g = parse_ttir((GOLDEN / "add_sm80.ttir").read_text())
    ctx = LaunchContext(
        grid=(5, 1, 1),
        params={"n_elements": 10**9},  # mask never fires -> tail unguarded
        tensors={n: _meta(4096) for n in ("x_ptr", "y_ptr", "out_ptr")},
    )
    v = check_graph(g, ctx)
    assert len(v) == 3  # two loads + one store
    assert all(r.violation_offset >= 4096 for r in v)
    assert any(r.kind == "store" for r in v) and any(r.kind == "load" for r in v)


# ──────────────────────── matmul (loop, 2D) ────────────────────────


def test_matmul_correct_is_in_bounds():
    g = parse_ttir((GOLDEN / "matmul_s3_sm80.ttir").read_text())
    ctx = LaunchContext(
        grid=(2, 2, 1),
        params={
            "M": 128,
            "N": 128,
            "K": 128,
            "stride_am": 128,
            "stride_bk": 128,
            "stride_cm": 128,
        },  # fmt: skip
        tensors={
            n: _meta(128 * 128, elem_bits=16) for n in ("a_ptr", "b_ptr", "c_ptr")
        },
    )
    assert check_graph(g, ctx) == []


def test_matmul_oversized_grid_rows_oob():
    """The A-operand load has no row mask (only a K-dim mask), so a grid with
    too many row blocks (pid_m up to 2 with M=128, BLOCK_M=64) reads rows
    past M — a real launch-misconfiguration OOB the loop model must catch."""
    g = parse_ttir((GOLDEN / "matmul_s3_sm80.ttir").read_text())
    ctx = LaunchContext(
        grid=(3, 2, 1),  # cdiv(128,64)=2, so pid_m=2 is out of range
        params={
            "M": 128,
            "N": 128,
            "K": 128,
            "stride_am": 128,
            "stride_bk": 128,
            "stride_cm": 128,
        },  # fmt: skip
        tensors={
            n: _meta(128 * 128, elem_bits=16) for n in ("a_ptr", "b_ptr", "c_ptr")
        },
    )
    v = check_graph(g, ctx)
    assert any(r.base_param == "a_ptr" for r in v), "row-overflow A load not caught"


# ──────────────────── 2D dimension separation ────────────────────


def test_reused_arange_rows_and_cols_are_independent():
    """A single make_range reused for the row and column index (triton does
    this in 2D tiles) MUST become two independent variables. Regression
    guard: with offset = row - col, collapsing them to one variable makes
    the offset identically 0 (never OOB); keeping them independent lets
    row < col drive the offset negative -> OOB. SAT here proves separation."""
    r = "%shared_range"
    offset = Bin(
        "-",
        Arange(r, 0, 4, dim=0),  # row
        Arange(r, 0, 4, dim=1),  # col, same make_range ssa, different dim
    )
    g = AccessGraph(
        kernel_name="synthetic",
        func_args=[FuncArg("p", True, 32)],
        accesses=[
            AccessEvent("load", "p", offset, None, 32, None, 1),
        ],
        loop=None,
    )
    ctx = LaunchContext(grid=(1, 1, 1), params={}, tensors={"p": _meta(64)})
    v = check_graph(g, ctx)
    assert len(v) == 1
    assert v[0].violation_offset < 0  # row < col -> negative element offset
