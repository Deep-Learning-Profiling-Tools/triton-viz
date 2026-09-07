"""Token order survives each compiled analysis path and real alias bindings."""

from dataclasses import replace

import pytest

from triton_viz.clients.common.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Cmp,
    Const,
    DataDep,
    FuncArg,
    Loaded,
    Observed,
    Param,
    UnsupportedTTIR,
)
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)


def _access(kind, base, line, *, mask=None):
    return AccessEvent(kind, base, Const(0), mask, 32, None, line)


def _graph(*, target="x", guard=None, ordered=True):
    return AccessGraph(
        "token_encoding",
        [FuncArg("x", True, 32), FuncArg("y", True, 32), FuncArg("c", False, 0)],
        [_access("store", "x", 1), _access("load", target, 2)],
        None,
        frontend="cutile",
        token_order={(0, 1): guard} if ordered else {},
    )


def _tensors(*, alias=False):
    return {
        "x": GlobalTensor(data_ptr=4096, numel=1, elem_size=4),
        "y": GlobalTensor(data_ptr=4096 if alias else 8192, numel=1, elem_size=4),
    }


def _races(enc):
    return TwoCopySymbolicHBSolver(
        enc.records,
        grid=(1, 1, 1),
        arange_dict=enc.arange_dict,
        fence_order=True,
        token_order=enc.token_order,
    ).find_races()


@pytest.mark.parametrize("value,expected", [(0, True), (1, False)])
def test_guarded_token_order_changes_only_the_selected_path(value, expected):
    graph = _graph(guard=Cmp("eq", Param("c"), Const(1)))
    enc = encode_graph(graph, {"c": value}, _tensors())
    assert enc.fence_order_applies
    assert bool(_races(enc)) == expected


def test_t0_preserves_transitive_order_across_an_omitted_tensor_group():
    graph = _graph()
    graph.accesses.insert(1, _access("load", "y", 2, mask=Const(0)))
    graph.token_order = {(0, 1): None, (0, 2): None, (1, 2): None}
    groups = dict(encode_graph_t0(graph))
    assert set(groups) == {"x"}
    assert set(groups["x"].token_order) == {(0, 2)}
    assert not _races(groups["x"])


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("alias", [False, True])
def test_public_client_uses_actual_aliasing_without_inventing_tokens(level, alias):
    graph = _graph(target="y", ordered=False)
    detector = CompiledRaceDetector(
        confirm_races=False, differential_check=False, ladder_level=level
    )
    verdict = detector.analyze_graph(graph, {}, _tensors(alias=alias), (1, 1, 1))
    assert verdict["verdict"] == ("race" if alias else "race-free"), verdict


@pytest.mark.parametrize(
    "guard",
    [
        DataDep("unknown"),
        Observed(0),
        Arange("lane", 0, 2),
        Loaded(0, "x", Const(0), None, None),
    ],
)
def test_unmodeled_token_guard_is_never_widened_to_extra_order(guard):
    with pytest.raises(UnsupportedTTIR, match="exact scalar") as exc:
        encode_graph(_graph(guard=guard), {}, _tensors())
    assert exc.value.kind == "token-order"


def test_cutile_graph_without_token_metadata_refuses_legacy_fallback():
    with pytest.raises(UnsupportedTTIR, match="no captured token order"):
        encode_graph(replace(_graph(), token_order=None), {}, _tensors())


@pytest.mark.parametrize("frontend", ["triton", "cutile"])
def test_content_free_attempt_does_not_prove_unordered_same_instance_stores(frontend):
    # A free loaded mask can activate the second store. Omitting the
    # ordering discipline on this early proof path used to manufacture a
    # proof from full source order before the exact snapshot attempt.
    graph = _graph(ordered=False)
    graph.frontend = frontend
    graph.token_order = {} if frontend == "cutile" else None
    graph.multipath = True
    graph.accesses = [
        _access("load", "y", 1),
        _access("store", "x", 2),
        _access("store", "x", 3, mask=Loaded(0, "y", Const(0), None, None)),
    ]
    detector = CompiledRaceDetector(
        confirm_races=False, differential_check=False, ladder_level=2
    )
    assert detector._t1_content_free(graph, {}, _tensors(), (1, 1, 1)) is None
