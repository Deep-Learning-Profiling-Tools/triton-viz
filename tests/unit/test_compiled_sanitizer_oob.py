"""Unit tests for the compiled sanitizer's OOB query engine."""

from pathlib import Path

from triton_viz.clients.sanitizer.compiled.oob import (
    LaunchContext,
    TensorMeta,
    check_graph,
)
import pytest

from triton_viz.clients.sanitizer.compiled.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    Const,
    FuncArg,
    LoopInfo,
    LoopVar,
    Param,
    Pid,
    UnsupportedTTIR,
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


# ──────────────── loop lower bound / step (iteration model) ────────────────


def _store_loop(lower, step, upper, offset):
    return AccessGraph(
        kernel_name="synthetic",
        func_args=[FuncArg("out", True, 32)],
        accesses=[AccessEvent("store", "out", offset, None, 32, None, 1)],
        loop=LoopInfo(
            loop_ssa="%loop",
            induction_var="%k",
            lower=lower,
            step=step,
            upper=upper,
        ),  # fmt: skip
    )


def test_loop_nonzero_lower_no_false_positive():
    """for k in range(1, n): store(out + (k-1)) writes offsets 0..n-2 — in
    bounds for numel == n-1. The old model bounded the induction value as
    [0, upper), so iteration k=0 (which never runs) drove offset to -1, a
    false OOB. The iteration model excludes it."""
    g = _store_loop(
        Const(1), Const(1), Param("n"), Bin("-", LoopVar("%loop"), Const(1))
    )
    ctx = LaunchContext(grid=(1, 1, 1), params={"n": 8}, tensors={"out": _meta(7)})
    assert check_graph(g, ctx) == []


def test_loop_step_skips_unrun_iterations():
    """for k in range(0, n, 2): store(out + k) accesses only even induction
    values {0,2,4,6} for n=8 — in bounds for numel=7. The old model checked
    every v in [0, n), so the never-run v=7 false-flagged OOB."""
    g = _store_loop(Const(0), Const(2), Param("n"), LoopVar("%loop"))
    in_bounds = LaunchContext(
        grid=(1, 1, 1), params={"n": 8}, tensors={"out": _meta(7)}
    )
    assert check_graph(g, in_bounds) == []
    # A genuine OOB on a real (even) iteration is still caught.
    oob = LaunchContext(grid=(1, 1, 1), params={"n": 8}, tensors={"out": _meta(6)})
    v = check_graph(g, oob)
    assert len(v) == 1 and v[0].violation_offset == 6


def test_descending_loop_is_unsupported():
    """A non-positive step is not modeled — unsupported, never a silent
    proof."""
    g = _store_loop(Const(10), Const(-1), Const(0), LoopVar("%loop"))
    ctx = LaunchContext(grid=(1, 1, 1), params={}, tensors={"out": _meta(16)})
    with pytest.raises(UnsupportedTTIR, match="step"):
        check_graph(g, ctx)


# ──────────── min / max / signed remainder (swizzle arithmetic) ────────────


def _flat_load(offset, numel, grid_x):
    g = AccessGraph(
        kernel_name="synthetic",
        func_args=[FuncArg("p", True, 32)],
        accesses=[AccessEvent("load", "p", offset, None, 32, None, 1)],
        loop=None,
    )
    ctx = LaunchContext(grid=(grid_x, 1, 1), params={}, tensors={"p": _meta(numel)})
    return g, ctx


def test_remainder_wraps_offsets_in_bounds():
    """offset = pid % 8 stays in [0, 8) for any grid — a proof, not an OOB.
    This is the `pid % group_size_m` shape of the grouped-swizzle matmul."""
    g, ctx = _flat_load(Bin("%", Pid(0), Const(8)), numel=8, grid_x=64)
    assert check_graph(g, ctx) == []
    # The same wrap against a smaller tensor is a genuine OOB at offset 7.
    g, ctx = _flat_load(Bin("%", Pid(0), Const(8)), numel=7, grid_x=64)
    v = check_graph(g, ctx)
    assert len(v) == 1 and v[0].violation_offset == 7


def test_min_clamp_proves_in_bounds():
    """offset = min(pid, 9) never escapes [0, 10) however large the grid."""
    g, ctx = _flat_load(Bin("min", Pid(0), Const(9)), numel=10, grid_x=1000)
    assert check_graph(g, ctx) == []
    g, ctx = _flat_load(Bin("min", Pid(0), Const(9)), numel=9, grid_x=1000)
    v = check_graph(g, ctx)
    assert len(v) == 1 and v[0].violation_offset == 9


def test_max_floor_detects_oob():
    """offset = max(pid, 5) pins the floor at 5: numel=6 is a proof, numel=5
    overflows even at pid=0."""
    g, ctx = _flat_load(Bin("max", Pid(0), Const(5)), numel=6, grid_x=4)
    assert check_graph(g, ctx) == []
    g, ctx = _flat_load(Bin("max", Pid(0), Const(5)), numel=5, grid_x=4)
    v = check_graph(g, ctx)
    assert len(v) == 1 and v[0].violation_offset >= 5


def test_division_truncates_toward_zero_no_false_positive():
    """offset = (0 - pid) // 2 with pid in {0, 1}: arith.divsi truncates, so
    (-1) // 2 == 0 and both offsets are 0 — in bounds. Z3's Euclidean Int
    division would give -1, a false OOB; this pins the trunc encoding."""
    off = Bin("//", Bin("-", Const(0), Pid(0)), Const(2))
    g, ctx = _flat_load(off, numel=4, grid_x=2)
    assert check_graph(g, ctx) == []


def test_remainder_keeps_dividend_sign():
    """offset = (0 - pid) % 2 with pid in {0, 1}: arith.remsi carries the
    dividend's sign, so pid=1 gives -1 — a real negative-offset OOB. A naive
    Euclidean mod would fold it to +1 and silently prove the access safe."""
    off = Bin("%", Bin("-", Const(0), Pid(0)), Const(2))
    g, ctx = _flat_load(off, numel=2, grid_x=2)
    v = check_graph(g, ctx)
    assert len(v) == 1 and v[0].violation_offset == -1


# ──────────── branch-guarded accesses (scf.if witness policy) ────────────


def _two_access_graph(guarded_offset, unguarded_offset=None):
    accesses = [
        AccessEvent("load", "p", guarded_offset, None, 32, None, 1, guarded=True)
    ]
    if unguarded_offset is not None:
        accesses.append(AccessEvent("load", "p", unguarded_offset, None, 32, None, 2))
    return AccessGraph(
        kernel_name="synthetic",
        func_args=[FuncArg("p", True, 32)],
        accesses=accesses,
        loop=None,
    )


def test_guarded_access_unsat_is_still_a_proof():
    """Checking a guarded access as unconditional over-approximates its
    reachable states; UNSAT on the superset proves the real access too."""
    g = _two_access_graph(Pid(0))
    ctx = LaunchContext(grid=(4, 1, 1), params={}, tensors={"p": _meta(4)})
    assert check_graph(g, ctx) == []


def test_guarded_access_sat_is_unsupported_not_witnessed():
    """SAT on a guarded access may sit in a branch the launch never takes
    (diag_ssm's `if t > 0` shape) — report unsupported, never a witness."""
    g = _two_access_graph(Bin("-", Pid(0), Const(1)))  # pid=0 -> offset -1
    ctx = LaunchContext(grid=(4, 1, 1), params={}, tensors={"p": _meta(4)})
    with pytest.raises(UnsupportedTTIR, match="branch-guarded"):
        check_graph(g, ctx)


def test_unguarded_witness_beats_guarded_uncertainty():
    """A SAT unguarded access is a real, reachable violation; it must be
    reported even when a guarded access is also SAT (chunk_gla_fwd's OOB
    load sits before the scf.if in the same kernel)."""
    g = _two_access_graph(
        Bin("-", Pid(0), Const(1)),  # guarded, SAT (possible false alarm)
        Bin("+", Pid(0), Const(100)),  # unguarded, SAT (real witness)
    )
    ctx = LaunchContext(grid=(4, 1, 1), params={}, tensors={"p": _meta(4)})
    v = check_graph(g, ctx)
    assert len(v) == 1
    assert v[0].violation_offset >= 100  # the unguarded access, line 2
    assert v[0].line_no == 2


# ──────────────── completeness: no skipped access ────────────────


def test_missing_tensor_metadata_is_unsupported_not_ok():
    """A static proof requires checking EVERY access. A base pointer with no
    registered tensor metadata cannot be bounded, so the analysis must bail
    to unsupported rather than skip the access and return an empty (false)
    proof."""
    g = AccessGraph(
        kernel_name="synthetic",
        func_args=[FuncArg("p", True, 32)],
        accesses=[AccessEvent("load", "p", Const(0), None, 32, None, 1)],
        loop=None,
    )
    ctx = LaunchContext(grid=(1, 1, 1), params={}, tensors={})
    with pytest.raises(UnsupportedTTIR, match="missing tensor metadata"):
        check_graph(g, ctx)
