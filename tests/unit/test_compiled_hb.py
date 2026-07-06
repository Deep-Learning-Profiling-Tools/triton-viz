"""Unit tests for the rotation closed forms and the pipeline counting model."""

from pathlib import Path

import pytest

from triton_viz.clients.race_detector.compiled.hb import (
    ConstSlot,
    RotatingSlot,
    build_pipeline_model,
    resolve_slot,
)
from triton_viz.clients.race_detector.compiled.ttgir_reader import (
    UnsupportedTTGIR,
    parse_ttgir,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _graph(name: str = "matmul_s3_sm80.ttgir"):
    return parse_ttgir((GOLDEN / name).read_text())


def test_rotation_closed_forms():
    """Extract index (init -1) and insert index (init 1) both reduce to
    k mod 2 — checked against exhaustive simulation of the parsed
    addi/cmpi/select chain inside resolve_slot."""
    g = _graph()

    extract = resolve_slot(g, "%acc_95")
    assert extract == RotatingSlot(base=0, modulus=2)

    insert = resolve_slot(g, "%acc_106")
    assert insert == RotatingSlot(base=0, modulus=2)

    assert resolve_slot(g, "%c0_i32") == ConstSlot(0)
    assert resolve_slot(g, "%c1_i32") == ConstSlot(1)


def test_rotation_simulation_rejects_non_rotations():
    """A doctored chain that does not follow (base + k) mod S must be
    rejected, not trusted: flip the select arms so the index sticks at 0
    after wrap instead of rotating."""
    text = (GOLDEN / "matmul_s3_sm80.ttgir").read_text()
    doctored = text.replace(
        "%acc_95 = arith.select %acc_94, %c0_i32, %acc_93",
        "%acc_95 = arith.select %acc_94, %acc_93, %c0_i32",
    )
    g = parse_ttgir(doctored)
    with pytest.raises(UnsupportedTTGIR, match="does not follow"):
        resolve_slot(g, "%acc_95")


def test_pipeline_counting_model():
    g = _graph()
    m = build_pipeline_model(g)

    assert not m.generic_only
    assert m.prologue_commits == 4  # two (a, b) pairs peeled
    assert m.commits_per_iter == 2  # one commit group per input per iter

    loop_copies = [c for c in m.copies if c.loop_pos is not None]
    assert len(loop_copies) == 2
    assert {c.loop_pos for c in loop_copies} == {1, 2}
    assert all(c.committed for c in m.copies)

    prologue_ranks = sorted(c.const_rank for c in m.copies if c.const_rank is not None)
    assert prologue_ranks == [1, 2, 3, 4]

    # Both loads are guarded by the leading in-loop wait: num=2, with no
    # commit groups issued before the wait inside the body.
    assert all(ld.wait_num == 2 and ld.issued_before_wait == 0 for ld in m.loads)


def test_generic_only_model():
    m = build_pipeline_model(_graph("matmul_s1_sm80.ttgir"))
    assert m.generic_only


def test_cyclic_scalar_chain_is_unsupported_not_recursion_error():
    """Adversarial use-before-def SSA cycles must degrade to unsupported."""
    from triton_viz.clients.race_detector.compiled import analyze_ttgir

    cyclic = """\
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cyclic(%p: !tt.ptr<f16>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %ub = arith.constant 100 : i32
    %ptrs = tt.splat %p : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked1>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%i = %c0_i32) -> (i32)  : i32 {
      %x = arith.addi %y, %c1_i32 : i32
      %c = arith.cmpi sge, %x, %c2_i32 : i32
      %s = arith.select %c, %c0_i32, %x : i32
      %y = arith.addi %s, %i : i32
      %v = ttg.memdesc_index %buf[%s] : !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
      %cp = ttg.async_copy_global_to_local %ptrs, %v : tensor<64x32x!tt.ptr<f16>, #blocked1> -> <64x32xf16, #shared, #smem, mutable>
      %cg = ttg.async_commit_group tokens %cp
      scf.yield %y : i32
    }
    tt.return
  }
}
"""
    r = analyze_ttgir(cyclic)
    assert r.status == "unsupported"
    assert r.unsupported_reason is not None


def test_non_ttgir_input_is_unsupported_not_a_proof():
    from triton_viz.clients.race_detector.compiled import analyze_ttgir

    for garbage in ("", "  \n\t\n", ".version 8.0\n.target sm_80\n"):
        r = analyze_ttgir(garbage)
        assert r.status == "unsupported"
        assert "tt.func" in (r.unsupported_reason or "")
