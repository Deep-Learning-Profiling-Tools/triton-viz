"""End-to-end tests for the compiled-mode race detector.

Two layers:
  * analyzer-level golden tests — stock TTGIR must prove race-freedom
    (UNSAT), and every mutation of the pipeline machinery must produce a
    report with a sensible witness;
  * trace-level tests — a real kernel through ``triton_viz.trace`` with the
    warmup-acquired TTGIR (requires a CUDA driver; skipped without one).
"""

from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import CompiledRaceDetector, RaceDetector, Tracer
from triton_viz.clients.race_detector.compiled import analyze_ttgir
from triton_viz.core.client import ClientManager
from triton_viz.core.config import config

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name: str) -> str:
    return (GOLDEN / name).read_text()


# ──────────────────────── analyzer-level: proofs ────────────────────────


def test_stock_pipelined_matmul_is_proven_race_free():
    """The compiler-emitted 3-stage cp.async pipeline is exactly covered by
    its async_wait counting: every (copy, load) query is UNSAT, for every
    trip count and grid — a proof for the specialization."""
    r = analyze_ttgir(_read("matmul_s3_sm80.ttgir"))
    assert r.status == "ok"
    assert r.reports == []


def test_generic_only_and_no_smem_kernels_are_ok():
    for name in ("matmul_s1_sm80.ttgir", "add_sm80.ttgir"):
        r = analyze_ttgir(_read(name))
        assert r.status == "ok"
        assert r.reports == []


def test_sm90_wgmma_is_unsupported_not_silent():
    r = analyze_ttgir(_read("matmul_s3_sm90.ttgir"))
    assert r.status == "unsupported"
    assert r.unsupported_reason is not None
    assert "ttng" in r.unsupported_reason


# ──────────────────────── analyzer-level: mutations ────────────────────────
# Each mutation hand-edits the golden TTGIR the way a pipeliner bug would
# manifest; the detector must produce a RAW report with a valid witness.


def _assert_races(text: str, expect_min: int = 1) -> list:
    r = analyze_ttgir(text)
    assert r.status == "ok", r.unsupported_reason
    assert len(r.reports) >= expect_min, "mutation not detected"
    for rep in r.reports:
        assert rep.race_type == "RAW"
        assert rep.witness["k_load"] >= 0
        assert rep.witness["slot"] >= 0
    return r.reports


def test_mutation_weakened_wait_num():
    """async_wait num too large (4 instead of 2): the wait tolerates more
    outstanding groups than the rotation distance provides."""
    stock = _read("matmul_s3_sm80.ttgir")
    _assert_races(stock.replace("{num = 2 : i32}", "{num = 4 : i32}"))


def test_mutation_wait_num_off_by_one():
    """The stock kernel is exactly tight: num=3 already races."""
    stock = _read("matmul_s3_sm80.ttgir")
    _assert_races(stock.replace("{num = 2 : i32}", "{num = 3 : i32}"))


def test_mutation_deleted_wait():
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = "\n".join(
        line for line in stock.splitlines() if "ttg.async_wait %" not in line
    )
    _assert_races(mutated)


def test_mutation_shrunk_stage_dim():
    """Single-buffering a double-buffered pipeline (stage dim 2 -> 1 with a
    consistent rotation modulus) makes the prefetch overwrite the slot the
    current iteration still reads."""
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = stock.replace("!ttg.memdesc<2x", "!ttg.memdesc<1x").replace(
        "memdesc<2x", "memdesc<1x"
    )
    # Keep the rotation consistent with the shrunk buffer: wrap at 1.
    mutated = mutated.replace(
        "arith.cmpi sge, %acc_93, %c2_i32", "arith.cmpi sge, %acc_93, %c1_i32"
    ).replace("arith.cmpi sge, %acc_104, %c2_i32", "arith.cmpi sge, %acc_104, %c1_i32")
    _assert_races(mutated)


def test_mutation_rotation_off_by_one():
    """Extract index initialized one step ahead: loads read the slot whose
    prefetch the wait does not yet cover."""
    stock = _read("matmul_s3_sm80.ttgir")
    _assert_races(stock.replace("%acc_86 = %c-1_i32", "%acc_86 = %c0_i32"))


def test_mutation_dropped_commit_group():
    """A copy whose token never reaches a commit group can never be covered
    by any wait — exactly that one pair must be reported."""
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = stock.replace(
        "%a_117 = ttg.async_commit_group tokens %a_116",
        "%a_117 = ttg.async_commit_group",
    )
    reports = _assert_races(mutated)
    assert len(reports) == 1


def test_mutation_dropped_wait_operand_token():
    """The stock loop wait awaits BOTH allocations' commit tokens
    (``async_wait %a_87, %b_89 {num = 2}``). Dropping the B token while
    keeping num=2 is a wait that no longer awaits B: the counting model alone
    would still read it as covered (the num is unchanged), so the per-token
    coverage gate must catch it and report exactly the B-buffer loads — never
    a silent ``ok``."""
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = stock.replace(
        "ttg.async_wait %a_87, %b_89 {num = 2 : i32}",
        "ttg.async_wait %a_87 {num = 2 : i32}",
    )
    reports = _assert_races(mutated)
    # A stays awaited (operand %a_87 present) — only B loads are uncovered.
    assert reports, "dropped wait token not detected"
    assert all(r.alloc == "%b" for r in reports), [r.alloc for r in reports]


def test_malformed_commit_group_is_unsupported():
    """An operand-style ``async_commit_group %tok`` (no ``tokens`` keyword)
    is outside the parsed vocabulary. It must degrade to unsupported, not be
    silently swallowed as an empty commit group (which would corrupt commit
    rank accounting)."""
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = stock.replace(
        "ttg.async_commit_group tokens %a_116",
        "ttg.async_commit_group %a_116",
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "async_commit_group" in (r.unsupported_reason or "")


def test_conditional_region_in_loop_is_unsupported():
    """A conditional region inside the pipelined loop is not modeled; the
    reader must say so rather than mis-track the loop/epilogue boundary via
    naive brace counting (a ``} else {`` nets +1, not 0)."""
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = stock.replace(
        "%a_96 = ttg.async_wait %a_87, %b_89 {num = 2 : i32} loc(#loc72)",
        "%a_96 = ttg.async_wait %a_87, %b_89 {num = 2 : i32} loc(#loc72)\n"
        "      scf.if %arg13 {\n"
        "      } else {\n"
        "      }",
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "region" in (r.unsupported_reason or "")


def test_local_alloc_after_dealloc_is_unsupported():
    """A local_alloc following a local_dealloc may reuse the freed storage —
    allocation aliasing the v1 model does not track. It must degrade to
    unsupported (the terminal epilogue deallocs in the stock IR, with no
    later alloc, stay a clean proof — see the stock test)."""
    stock = _read("matmul_s3_sm80.ttgir")
    mutated = stock.replace(
        "ttg.local_dealloc %b : !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable> loc(#loc89)",
        "ttg.local_dealloc %b : !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable> loc(#loc89)\n"
        "    %reuse = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable> loc(#loc89)",
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "dealloc" in (r.unsupported_reason or "")


def test_compiled_detector_is_standalone_only():
    """The compiled detector skips the interpreted run (pre_run=False), which
    is all()-combined — composing it with another client would suppress that
    client's capture. ClientManager must reject the composition up front."""
    ClientManager([CompiledRaceDetector()])  # standalone is fine
    with pytest.raises(RuntimeError, match="standalone"):
        ClientManager([CompiledRaceDetector(), Tracer()])
    with pytest.raises(RuntimeError, match="standalone"):
        ClientManager([Tracer(), CompiledRaceDetector()])


def test_smtlib_artifact_export():
    """SAT queries can be exported as SMT-LIB2 — the plan's SMT-IR
    interchange artifact."""
    stock = _read("matmul_s3_sm80.ttgir")
    r = analyze_ttgir(
        stock.replace("{num = 2 : i32}", "{num = 4 : i32}"), collect_smtlib=True
    )
    assert r.reports and r.smtlib
    assert "(declare-fun" in r.smtlib[0] or "(assert" in r.smtlib[0]


# ──────────────────────── trace-level (needs CUDA) ────────────────────────


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warmup compilation needs a CUDA driver"
)


@pytest.fixture
def _enable_race_detector():
    # The public RaceDetector(...) factory only returns a real backend while
    # the flag is on; otherwise it is the NullRaceDetector and trace() leaves
    # the kernel untraced.
    saved = config.enable_race_detector
    config.enable_race_detector = True
    try:
        yield
    finally:
        config.enable_race_detector = saved


@requires_cuda
def test_trace_pipelined_matmul_proves_race_free(_enable_race_detector):
    detector = RaceDetector(compile=True)
    # The factory routes compile=True to the static compiled-mode backend.
    assert isinstance(detector, CompiledRaceDetector)

    @triton_viz.trace(detector)
    @triton.jit
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_bk,
        stride_cm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - _k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - _k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

    M = N = K = 128
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    c = torch.empty(M, N, dtype=torch.float16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        K,
        N,
        N,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )

    # The sm80/sm90 split depends on the local GPU: Ampere-class targets
    # produce the cp.async pipeline (proof); Hopper+ produces ttng ops
    # (honest unsupported). Both are correct outcomes; silent wrong
    # verdicts are not.
    assert detector.last_status in ("ok", "unsupported")
    if detector.last_status == "ok":
        assert detector.last_reports == []
    else:
        assert detector.unsupported_reason is not None


@requires_cuda
def test_trace_elementwise_kernel_is_ok(_enable_race_detector):
    detector = RaceDetector(compile=True)

    @triton_viz.trace(detector)
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n)
    y = torch.randn(n)
    out = torch.empty(n)
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)

    assert detector.last_status == "ok"
    assert detector.last_reports == []
