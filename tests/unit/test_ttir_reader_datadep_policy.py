"""Unit tests for the per-term DataDep policy: a data-dependent MASK is
over-approximated as free (dropped, flagged, proof-only), while a
data-dependent ADDRESS stays whole-kernel unsupported with the classified
kind that routes the kernel to the interpreter front-end."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_viz.clients.common.ttir_reader import UnsupportedTTIR, parse_ttir
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
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


# Store to out_ptr[0..63] behind a mask computed from loaded data.
DATADEP_MASK = _mini(
    "%s = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%l = tt.load %s : tensor<64x!tt.ptr<f32>>",
    "%z = arith.constant dense<0> : tensor<64xi32>",
    "%m = arith.cmpi sgt, %l, %z : tensor<64xi32>",
    "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
    "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "tt.store %q, %l, %m : tensor<64x!tt.ptr<f32>>",
)


def test_datadep_mask_is_dropped_not_fatal():
    """Previously 'data-dependent mask' failed the whole kernel at parse."""
    g = parse_ttir(DATADEP_MASK)
    store = next(a for a in g.accesses if a.kind == "store")
    assert store.mask is None and store.mask_dropped is True
    assert not store.guarded
    load = next(a for a in g.accesses if a.kind == "load")
    assert load.mask_dropped is False


def test_datadep_mask_proof_still_lands():
    """Dropping the mask only widens the footprint: if the widened access is
    in bounds, the proof is real — the coverage win of the policy."""
    g = parse_ttir(DATADEP_MASK)
    ctx = LaunchContext(
        grid=(1, 1, 1),
        params={},
        tensors={"x_ptr": _meta(64), "out_ptr": _meta(64)},
    )
    assert check_graph(g, ctx) == []


def test_datadep_mask_possible_oob_abstains():
    """A SAT under the widened mask may pick a lane the real mask disables:
    abstain (unsupported, classified), never report it as a witness."""
    g = parse_ttir(DATADEP_MASK)
    ctx = LaunchContext(
        grid=(1, 1, 1),
        params={},
        tensors={"x_ptr": _meta(64), "out_ptr": _meta(32)},
    )
    with pytest.raises(UnsupportedTTIR, match="data-dependent mask") as ei:
        check_graph(g, ctx)
    assert ei.value.kind == "data-dependent-mask"


def test_indirect_address_is_classified():
    with pytest.raises(UnsupportedTTIR, match="data-dependent") as ei:
        parse_ttir(_read("gather_sm80.ttir"))
    assert ei.value.kind == "indirect-address"


def test_nested_loop_is_classified():
    with pytest.raises(UnsupportedTTIR, match="nested loops") as ei:
        parse_ttir(
            _mini(
                "%c0 = arith.constant 0 : i32",
                "%c1 = arith.constant 1 : i32",
                "%c4 = arith.constant 4 : i32",
                "scf.for %i = %c0 to %c4 step %c1 : i32 {",
                "  scf.for %j = %c0 to %c4 step %c1 : i32 {",
                "  }",
                "}",
            )
        )
    assert ei.value.kind == "nested-loop"


def test_modeling_gap_datadep_is_not_classified_indirect():
    """DataDep is the generic unknown-value top: a loop ACCUMULATOR used as
    an offset is a loop-modeling gap, not indirection — it must not land in
    the indirect-address bucket (which routes to the interpreter and counts
    as 'permanent' in the evaluation)."""
    with pytest.raises(UnsupportedTTIR, match="loop accumulator") as ei:
        parse_ttir(
            _mini(
                "%c0 = arith.constant 0 : i32",
                "%c1 = arith.constant 1 : i32",
                "%c4 = arith.constant 4 : i32",
                "%acc = scf.for %i = %c0 to %c4 step %c1"
                " iter_args(%a = %c0) -> (i32) : i32 {",
                "  %p0 = tt.addptr %out_ptr, %a : !tt.ptr<f32>, i32",
                "  %v = arith.constant 1 : i32",
                "  scf.yield %a : i32",
                "}",
            )
        )
    assert ei.value.kind == "other"


def test_data_dependent_loop_bound_is_classified():
    """The CSR row-loop shape: for k in range(loaded_start, loaded_end)."""
    with pytest.raises(UnsupportedTTIR, match="loop upper bound") as ei:
        parse_ttir(
            _mini(
                "%c0 = arith.constant 0 : i32",
                "%c1 = arith.constant 1 : i32",
                "%n = tt.load %x_ptr : !tt.ptr<f32>",
                "scf.for %i = %c0 to %n step %c1 : i32 {",
                "}",
            )
        )
    assert ei.value.kind == "data-dependent-bound"


def test_race_client_reason_carries_the_kind():
    """The tier selector routes on the stable kind prefix."""
    det = CompiledRaceDetector()
    det.post_warmup_callback(
        None, SimpleNamespace(asm={"ttir": _read("gather_sm80.ttir")})
    )
    det.finalize()
    assert det.last_ttir_graphs == [None]
    assert det.last_ttir_unsupported[0].startswith("indirect-address: ")


def test_datadep_mask_composes_with_modeled_path():
    """A dropped mask inside a MODELED scf.if: the path still constrains the
    query precisely; the mask uncertainty alone triggers abstention on SAT."""
    g = parse_ttir(
        _mini(
            "%c0 = arith.constant 0 : i32",
            "%pid = tt.get_program_id x : i32",
            "%cnd = arith.cmpi eq, %pid, %c0 : i32",
            "%s = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%l = tt.load %s : tensor<64x!tt.ptr<f32>>",
            "%z = arith.constant dense<0> : tensor<64xi32>",
            "%m = arith.cmpi sgt, %l, %z : tensor<64xi32>",
            "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "scf.if %cnd {",
            "  tt.store %q, %l, %m : tensor<64x!tt.ptr<f32>>",
            "}",
        )
    )
    store = next(a for a in g.accesses if a.kind == "store")
    assert store.mask_dropped and store.path is not None and not store.guarded
    ok = LaunchContext(
        grid=(4, 1, 1),
        params={},
        tensors={"x_ptr": _meta(64), "out_ptr": _meta(64)},
    )
    assert check_graph(g, ok) == []
    with pytest.raises(UnsupportedTTIR, match="data-dependent mask"):
        check_graph(
            g,
            LaunchContext(
                grid=(4, 1, 1),
                params={},
                tensors={"x_ptr": _meta(64), "out_ptr": _meta(32)},
            ),
        )
