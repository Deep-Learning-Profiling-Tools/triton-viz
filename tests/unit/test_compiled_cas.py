"""Ordinary static CAS: conditional writes, synchronization and value admission."""

from dataclasses import replace

import pytest
from z3 import simplify

from triton_viz.clients.common.ttir_reader import (
    Bin,
    Const,
    Observed,
    UnsupportedTTIR,
    parse_ttir,
)
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.core.config import config as cfg


def _module(*lines):
    return (
        "module {\n"
        "  tt.func public @k(%flag_ptr: !tt.ptr<i32>, "
        "%data_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) attributes {noinline = false} {\n    "
        + "\n    ".join(lines)
        + "\n    tt.return\n  }\n}\n"
    )


_PREFIX = (
    "%c0 = arith.constant 0 : i32",
    "%c1 = arith.constant 1 : i32",
    "%c2 = arith.constant 2 : i32",
    "%true = arith.constant true",
    "%pid = tt.get_program_id x : i32",
)


def _tensors():
    return {
        "flag_ptr": GlobalTensor(0x1000, 4, 1, init_values=(0,)),
        "data_ptr": GlobalTensor(0x2000, 4, 1, init_values=(0,)),
        "out_ptr": GlobalTensor(0x3000, 4, 4),
    }


def _solve(text):
    enc = encode_graph(parse_ttir(text, multipath=True), {}, _tensors(), multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=(2, 1, 1),
        arange_dict=enc.arange_dict,
        fence_seqs=enc.fence_seqs,
        fence_order=True,
    )
    return enc, solver, solver.find_races()


@pytest.mark.parametrize("compare,has_race", [("%c1", False), ("%c0", True)])
def test_failed_ordinary_cas_does_not_write(compare, has_race):
    """The flag starts at zero. CAS(1,2) never writes and may share it
    with plain readers; CAS(0,2) writes and must conflict with those readers."""
    text = _module(
        *_PREFIX,
        f"%old = tt.atomic_cas relaxed, gpu, %flag_ptr, {compare}, %c2 : "
        "(!tt.ptr<i32>, i32, i32) -> i32",
        "%v = tt.load %flag_ptr : !tt.ptr<i32>",
    )
    _, solver, reports = _solve(text)
    assert bool(reports) is has_race
    assert solver.check_feasibility() is True


def _handoff(producer_sem="release", consumer_sem="acquire", scope="gpu"):
    return _module(
        *_PREFIX,
        "%producer = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %producer {",
        "tt.store %data_ptr, %c1 : !tt.ptr<i32>",
        "gpu.barrier",
        f"%pub = tt.atomic_cas {producer_sem}, {scope}, %flag_ptr, %c0, %c1 : "
        "(!tt.ptr<i32>, i32, i32) -> i32",
        "} else {",
        # This consumer CAS fails on the published value 1.
        f"%old = tt.atomic_cas {consumer_sem}, {scope}, %flag_ptr, %c0, %c0 : "
        "(!tt.ptr<i32>, i32, i32) -> i32",
        "gpu.barrier",
        "%ready = arith.cmpi eq, %old, %c1 : i32",
        "%v = tt.load %data_ptr, %ready : !tt.ptr<i32>",
        "%out = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %out, %v, %ready : !tt.ptr<i32>",
        "}",
    )


@pytest.mark.parametrize(
    "producer_sem,consumer_sem,scope,has_race",
    [
        ("release", "acquire", "gpu", False),
        ("relaxed", "acquire", "gpu", True),
        ("release", "relaxed", "gpu", True),
        ("release", "acquire", "cta", True),
    ],
)
def test_failed_cas_acquires_only_with_release_and_matching_scope(
    producer_sem, consumer_sem, scope, has_race
):
    enc, solver, reports = _solve(_handoff(producer_sem, consumer_sem, scope))
    assert not enc.assumes_termination
    assert bool(reports) is has_race
    assert solver.check_feasibility() is True


def _observed_operand_graph():
    return parse_ttir(
        _module(
            *_PREFIX,
            "%prior = tt.atomic_cas relaxed, gpu, %data_ptr, %c0, %c1 : "
            "(!tt.ptr<i32>, i32, i32) -> i32",
            "%old = tt.atomic_cas relaxed, gpu, %flag_ptr, %prior, %prior : "
            "(!tt.ptr<i32>, i32, i32) -> i32",
        )
    )


def test_atomic_observation_operands_preserve_value_and_copy_local_identity():
    graph = _observed_operand_graph()
    enc = encode_graph(graph, {}, _tensors())
    prior, cas = enc.records
    assert cas.cas_cmp_value.eq(prior.old_value)
    assert cas.cas_new_value.eq(prior.old_value)
    assert any(v.eq(prior.old_value) for v in cas.copy_local_vars)
    assert graph.accesses[1].atomic_cmp == Observed(0)
    # T0 groups tensors independently: the operand's observation must
    # still be renamed even though its source record is in another group.
    groups = dict(encode_graph_t0(graph))
    flag_cas = groups["flag_ptr"].records[0]
    assert any(v.eq(prior.old_value) for v in flag_cas.copy_local_vars)
    solver = TwoCopySymbolicHBSolver(groups["flag_ptr"].records, grid=(2, 1, 1))
    first, second = [e for e in solver.events if e.record.atomic_kind == "cas"]
    assert not simplify(first.written_value).eq(simplify(second.written_value))


@pytest.mark.parametrize("operand", ["compare", "new"])
@pytest.mark.parametrize("multipath", [False, True])
def test_plain_loaded_cas_operands_refuse_without_constant_or_free_substitution(
    operand, multipath
):
    compare, new = ("%value", "%c1") if operand == "compare" else ("%c0", "%value")
    graph = parse_ttir(
        _module(
            *_PREFIX,
            "%value = tt.load %data_ptr : !tt.ptr<i32>",
            f"%old = tt.atomic_cas acq_rel, gpu, %flag_ptr, {compare}, {new} : "
            "(!tt.ptr<i32>, i32, i32) -> i32",
        ),
        multipath=multipath,
    )
    for lower in (
        lambda: encode_graph(graph, {}, _tensors(), multipath=multipath),
        lambda: encode_graph_t0(graph, multipath=multipath),
    ):
        with pytest.raises(UnsupportedTTIR) as exc:
            lower()
        assert exc.value.kind == "cas-value"


@pytest.mark.parametrize("field", ["atomic_cmp", "atomic_val"])
def test_cas_operand_arithmetic_does_not_silently_ignore_i32_overflow(field):
    graph = _observed_operand_graph()
    cas = graph.accesses[1]
    # A prior observation may equal INT32_MAX. +1 would wrap to INT32_MIN,
    # so using unbounded Int here could change CAS success or publication.
    graph.accesses[1] = replace(cas, **{field: Bin("+", Observed(0), Const(1))})
    tensors = _tensors()
    tensors["data_ptr"] = GlobalTensor(0x2000, 4, 1, init_values=(2**31 - 1,))
    with pytest.raises(UnsupportedTTIR, match="operand arithmetic") as exc:
        encode_graph(graph, {}, tensors)
    assert exc.value.kind == "cas-value"


def test_cas_operand_observation_width_change_refuses():
    graph = _observed_operand_graph()
    graph.accesses[0] = replace(graph.accesses[0], elem_bits=64)
    tensors = _tensors()
    tensors["data_ptr"] = GlobalTensor(0x2000, 8, 1, init_values=(0,))
    with pytest.raises(UnsupportedTTIR, match="integer width") as exc:
        encode_graph(graph, {}, tensors)
    assert exc.value.kind == "cas-value"


@pytest.mark.parametrize("field", ["in_loop", "elem_float"])
def test_ordinary_cas_outside_integer_non_iterated_fragment_refuses(field):
    graph = _observed_operand_graph()
    graph.accesses[0] = replace(graph.accesses[0], **{field: True})
    with pytest.raises(UnsupportedTTIR):
        encode_graph(graph, {}, _tensors())


@pytest.mark.parametrize(
    "name",
    [
        "trb021_role_specific_order_no",
        "trb017_cas_unlock_no",
        "trb021_guarded_acq_rel_no",
        "trb021_release_only_yes",
        "trb021_acquire_only_yes",
        "trb017_mutex_plain_unlock_yes",
        "trb017_mutex_relaxed_cas_yes",
        "trb022_acquire_on_failure_no",
        "trb022_acquire_on_failure_relaxed_yes",
        "trb026_fenced_tile_handoff_no",
    ],
)
def test_benchmark_cas_repairs_and_racy_controls(name, monkeypatch):
    """Compile the unchanged actual benchmark and exercise the public IR
    client, including its proof-feasibility and confirmation checks."""
    from evaluation.harness import _host_compile_ttir, _static_track
    from evaluation.kernels.tritonracebench import CORPUS
    from triton_viz.clients.race_detector.ladder import LadderLevel

    monkeypatch.setattr(cfg, "race_detector_fence_order", True)
    spec = next(s for s in CORPUS.specs if s.name == name)
    result = _static_track(spec, _host_compile_ttir(spec), 0, LadderLevel.L2)
    expected_status = "ok" if spec.expected == "race-free" else "races"
    assert result["status"] == expected_status, result
    if expected_status == "ok":
        assert result["n_reports"] == 0
        assert result["provenance"].startswith("proved@")
    else:
        assert result["n_reports"] > 0


@pytest.mark.parametrize(
    "casts,operand",
    [
        (
            [
                "%small = arith.trunci %prior : i32 to i8",
                "%back = arith.extui %small : i8 to i32",
            ],
            "%back",
        ),
        (
            [
                "%bit = arith.cmpi eq, %pid, %c0 : i32",
                "%signed = arith.extsi %bit : i1 to i32",
            ],
            "%signed",
        ),
    ],
)
def test_transparent_cast_chains_cannot_change_ordinary_cas_values(casts, operand):
    graph = parse_ttir(
        _module(
            *_PREFIX,
            "%prior = tt.atomic_cas relaxed, gpu, %data_ptr, %c0, %c1 : "
            "(!tt.ptr<i32>, i32, i32) -> i32",
            *casts,
            f"%old = tt.atomic_cas relaxed, gpu, %flag_ptr, {operand}, %c1 : "
            "(!tt.ptr<i32>, i32, i32) -> i32",
        )
    )
    assert graph.has_value_changing_integer_casts
    with pytest.raises(UnsupportedTTIR, match="value-changing integer cast") as exc:
        encode_graph(graph, {}, _tensors())
    assert exc.value.kind == "cas-value"


def test_boolean_zero_extension_is_an_exact_integer_cas_operand():
    graph = parse_ttir(
        _module(
            *_PREFIX,
            "%bit = arith.cmpi eq, %pid, %c0 : i32",
            "%wide = arith.extui %bit : i1 to i32",
            "%old = tt.atomic_cas relaxed, gpu, %flag_ptr, %wide, %c1 : "
            "(!tt.ptr<i32>, i32, i32) -> i32",
        )
    )
    assert not graph.has_value_changing_integer_casts
    cas = encode_graph(graph, {}, _tensors()).records[0]
    assert cas.cas_cmp_value.sort().name() == "Int"


def test_cas_cast_gate_does_not_change_non_cas_address_encoding():
    graph = parse_ttir(
        _module(
            *_PREFIX,
            "%small = arith.trunci %c1 : i32 to i8",
            "%back = arith.extui %small : i8 to i32",
            "%value = tt.load %data_ptr : !tt.ptr<i32>",
        )
    )
    # The additional cast scan is unnecessary without any CAS.
    assert not graph.has_value_changing_integer_casts
    assert len(encode_graph(graph, {}, _tensors()).records) == 1
