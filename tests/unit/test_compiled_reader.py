"""Unit tests for the compiled-mode TTGIR reader on golden dumps."""

from pathlib import Path

import pytest

from triton_viz.clients.race_detector.compiled.ttgir_reader import (
    UnsupportedTTGIR,
    parse_ttgir,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name: str) -> str:
    return (GOLDEN / name).read_text()


def test_matmul_s3_event_graph():
    g = parse_ttgir(_read("matmul_s3_sm80.ttgir"))

    assert g.kernel_name == "matmul_kernel"
    assert g.num_warps == 4
    assert g.threads_per_warp == 32
    assert g.target == "cuda:80"

    assert set(g.allocations) == {"%a", "%b"}
    a = g.allocations["%a"]
    assert a.memdesc.dims == (2, 64, 32)
    assert a.stages == 2
    assert a.memdesc.elem_bits == 16
    assert a.stage_bytes == 64 * 32 * 2
    assert a.loc is not None and a.loc.var_name == "a"

    # 4 prologue copies (constant slots 0,0,1,1) + 2 loop copies.
    assert len(g.copies) == 6
    assert [c.segment for c in g.copies] == ["prologue"] * 4 + ["loop"] * 2
    assert [c.index_ssa for c in g.copies[:4]] == [
        "%c0_i32",
        "%c0_i32",
        "%c1_i32",
        "%c1_i32",
    ]

    # Each copy's token feeds exactly one commit group.
    committed = {tok for c in g.commits for tok in c.copy_tokens}
    assert committed == {c.token for c in g.copies}

    # One in-loop wait (num=2) leading the body, one epilogue wait (num=0).
    assert [(w.segment, w.num) for w in g.waits] == [("loop", 2), ("epilogue", 0)]

    # Both loads rotate via the same extract index and carry the wait token.
    assert len(g.loads) == 2
    assert {ld.index_ssa for ld in g.loads} == {"%acc_95"}
    assert all(ld.token == "%a_96" for ld in g.loads)

    # Loop metadata: iter_args include the two rotation indices with the
    # constant inits the closed form depends on.
    assert g.loop is not None
    inits = dict(g.loop.iter_args)
    assert g.constants[inits["%acc_86"]] == -1  # extract index init
    assert g.constants[inits["%arg13"]] == 1  # insert index init

    # Source locations resolved through bottom-of-file aliases.
    assert g.copies[0].loc is not None
    assert g.copies[0].loc.file.endswith(".py")


def test_matmul_s1_is_generic_only():
    g = parse_ttgir(_read("matmul_s1_sm80.ttgir"))
    assert not g.copies and not g.waits
    assert len(g.loads) == 2
    assert len(g.stores) == 2  # local_alloc-with-operand init writes


def test_elementwise_has_no_smem_events():
    g = parse_ttgir(_read("add_sm80.ttgir"))
    assert not g.copies and not g.loads and not g.stores


def test_sm90_wgmma_is_unsupported():
    with pytest.raises(UnsupportedTTGIR, match="ttng"):
        parse_ttgir(_read("matmul_s3_sm90.ttgir"))


def test_explicit_barrier_is_unsupported():
    text = _read("matmul_s3_sm80.ttgir").replace(
        "%a_96 = ttg.async_wait", "gpu.barrier\n      %a_96 = ttg.async_wait"
    )
    with pytest.raises(UnsupportedTTGIR, match="barrier"):
        parse_ttgir(text)


def test_unknown_ttg_op_is_unsupported():
    text = _read("matmul_s3_sm80.ttgir").replace(
        "ttg.local_dealloc %b", "ttg.frobnicate %b"
    )
    with pytest.raises(UnsupportedTTGIR, match="frobnicate"):
        parse_ttgir(text)
