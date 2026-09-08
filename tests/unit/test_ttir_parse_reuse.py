"""Evaluation may borrow only a unique, source-bound and unchanged TTIR parse.

Use real golden TTIR and the real parser/gate, with launch analysis mocked
where the harness clock and parse counts are under test. No kernels run.
"""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import harness
from triton_viz.clients.common import ttir_reader
from triton_viz.clients.race_detector.compiled import client, global_records
from triton_viz.clients.race_detector.ladder import LadderLevel


GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name="add_sm80.ttir"):
    return (GOLDEN / name).read_text()


def _consume(det, *texts):
    for text in texts:
        det.post_warmup_callback(None, SimpleNamespace(asm={"ttir": text}))
    det._consume_pending_ttir()


def _detector(level=LadderLevel.L2):
    return client.CompiledRaceDetector(
        ladder_level=level, retain_ttir_parse_binding=True
    )


@pytest.mark.parametrize("level", list(LadderLevel))
def test_binding_requires_exact_text_level_and_parse_mode(level):
    det = _detector(level)
    text = _read()
    _consume(det, text)
    graph = det.get_last_ttir_graph(text, ladder_level=level)
    assert graph is det.last_ttir_graphs[0]
    assert graph.multipath is (level >= 2)
    assert det.get_last_ttir_graph(text + "\n", ladder_level=level) is None
    for other in LadderLevel:
        if other != level:
            assert det.get_last_ttir_graph(text, ladder_level=other) is None

    # A cache keyed by TTIR alone must not create a new parse provenance
    # when a caller changes the client's ladder between launches.
    det.ladder_level = LadderLevel.L0 if level != 0 else LadderLevel.L2
    _consume(det, text)
    assert det.get_last_ttir_graph(text, ladder_level=det.ladder_level) is None


def test_binding_is_opt_in_and_current_launch_only():
    text = _read()
    det = client.CompiledRaceDetector()
    _consume(det, text)
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L0) is None
    assert det._ttir_parse_bindings == {}

    det = _detector()
    _consume(det, text)
    first = det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2)
    _consume(det, text)
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is first
    det.post_warmup_callback(None, SimpleNamespace(asm={"ttir": text}))
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None
    det._consume_pending_ttir()
    _consume(det, _read("atomic_sm80.ttir"))
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None
    _consume(det)
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None


@pytest.mark.parametrize("second", ["add_sm80.ttir", "atomic_sm80.ttir"])
def test_multiple_captures_are_ambiguous_even_when_text_is_identical(second):
    det = _detector()
    text = _read()
    _consume(det, text, _read(second))
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g: g.accesses.clear(),
        lambda g: g.func_args.clear(),
        lambda g: g.iter_args.update(
            {
                99: ttir_reader.IterArgInfo(
                    99, "p", ttir_reader.Const(0), ttir_reader.Const(1)
                )
            }
        ),
        lambda g: g.pid_axes.add(99),
        lambda g: g.fences.append(0.5),
        lambda g: g.loops.append(
            ttir_reader.LoopInfo(
                "x",
                "k",
                ttir_reader.Const(0),
                ttir_reader.Const(1),
                ttir_reader.Const(1),
            )
        ),
        lambda g: g.loop_token_conflicts.append(
            ttir_reader.LoopTokenConflict("x", 0, 0)
        ),
        lambda g: setattr(g, "kernel_name", "changed"),
        lambda g: setattr(g, "multipath", False),
        lambda g: setattr(
            g, "accesses", [replace(g.accesses[0], offset=ttir_reader.Const(99))]
        ),
    ],
)
def test_mutations_are_rejected_including_after_parse_cache_hit(mutate):
    det = _detector()
    text = _read()
    _consume(det, text)
    mutate(det.last_ttir_graphs[0])
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None
    _consume(det, text)
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None


def test_token_order_dictionary_does_not_alias_snapshot(monkeypatch):
    graph = ttir_reader.parse_ttir(_read(), multipath=True)
    graph.token_order = {(0, 1): None}
    monkeypatch.setattr(client, "parse_ttir", lambda *a, **k: graph)
    det = _detector()
    _consume(det, _read())
    assert det.get_last_ttir_graph(_read(), ladder_level=LadderLevel.L2) is graph
    graph.token_order.clear()
    assert det.get_last_ttir_graph(_read(), ladder_level=LadderLevel.L2) is None


def test_replaced_graph_or_failure_reason_cannot_borrow_binding():
    det = _detector()
    text = _read()
    _consume(det, text)
    graph = det.last_ttir_graphs[0]
    det.last_ttir_graphs[0] = replace(graph)
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None
    det.last_ttir_graphs[0] = graph
    det.last_ttir_unsupported[0] = "parse failed"
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None


def test_binding_failure_does_not_turn_successful_parse_into_unsupported(monkeypatch):
    def fail_copy(value):
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(client, "copy", fail_copy)
    det = _detector()
    text = _read()
    _consume(det, text)
    assert det.last_ttir_graphs[0] is not None
    assert det.last_ttir_unsupported == [None]
    assert det.get_last_ttir_graph(text, ladder_level=LadderLevel.L2) is None


@pytest.mark.parametrize("level", [LadderLevel.L0, LadderLevel.L2])
@pytest.mark.parametrize("name", ["add_sm80.ttir", "tile2d_sm80.ttir"])
def test_binding_preserves_real_static_analysis_and_graph(level, name):
    text = _read(name)
    params = {"n_elements": 4096, "M": 64, "N": 64, "stride_m": 64, "stride_n": 1}
    tensors = {
        name: global_records.GlobalTensor(data_ptr=base, elem_size=4, numel=4096)
        for name, base in [
            ("x_ptr", 0x1000),
            ("y_ptr", 0x11000),
            ("in_ptr", 0x1000),
            ("out_ptr", 0x21000),
        ]
    }
    results = []
    for retain in (False, True):
        det = client.CompiledRaceDetector(
            ladder_level=level, confirm_races=False, retain_ttir_parse_binding=retain
        )
        det._launch_params = dict(params)
        det._launch_tensors = dict(tensors)
        det._launch_grid = (4, 1, 1)
        det.post_warmup_callback(None, SimpleNamespace(asm={"ttir": text}))
        det.finalize()
        assert det.last_global_status == "ok"
        graph = det.get_last_ttir_graph(text, ladder_level=level)
        assert (graph is not None) is retain
        results.append(harness._static_result(det, 0.0, None))
    assert results[0] == results[1]


def test_shallow_snapshot_premise_holds_for_golden_ast_nodes():
    # The binding shares frozen AST nodes, copying only AccessGraph's
    # mutable fields. Check that these shared nodes have no mutable child
    # containers, so future reader extensions cannot silently break this.
    seen = set()

    def immutable(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, tuple):
            for child in value:
                immutable(child)
        elif is_dataclass(value):
            assert value.__dataclass_params__.frozen, type(value)
            for field in fields(value):
                immutable(getattr(value, field.name))
        else:
            assert value is None or isinstance(value, (str, int, float, bool)), type(
                value
            )

    for path in GOLDEN.glob("*.ttir"):
        try:
            graph = ttir_reader.parse_ttir(path.read_text(), multipath=True)
        except ttir_reader.UnsupportedTTIR:
            continue
        for field in fields(graph):
            value = getattr(graph, field.name)
            if isinstance(value, (list, set)):
                for child in value:
                    immutable(child)
            elif isinstance(value, dict):
                for key, child in value.items():
                    immutable(key)
                    immutable(child)
            else:
                immutable(value)


@pytest.fixture
def harness_probe(monkeypatch):
    events = []
    parser = ttir_reader.parse_ttir
    gate = global_records.t0_linearity_gate
    ticks = iter((10.0, 12.0))

    def clock():
        events.append("clock")
        return next(ticks)

    def client_parse(text, **kwargs):
        events.append("client_parse")
        return parser(text, **kwargs)

    def fallback_parse(text, **kwargs):
        events.append("fallback_parse")
        return parser(text, **kwargs)

    def public_gate(graph):
        events.append("gate")
        assert events.count("clock") == 2  # remains outside static.time_s
        return gate(graph)

    monkeypatch.setattr(harness, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(client, "parse_ttir", client_parse)
    monkeypatch.setattr(ttir_reader, "parse_ttir", fallback_parse)
    monkeypatch.setattr(global_records, "t0_linearity_gate", public_gate)
    monkeypatch.setattr(
        client.CompiledRaceDetector, "pre_warmup_callback", lambda *a, **k: None
    )
    monkeypatch.setattr(
        client.CompiledRaceDetector, "_analyze_global", lambda self: None
    )
    spec = SimpleNamespace(
        kernel_fn=SimpleNamespace(arg_names=[]),
        make_args=lambda seed: (),
        constexprs={},
        grid=(1,),
    )
    return spec, events


@pytest.mark.parametrize(
    ("name", "level", "expected"),
    [("add_sm80.ttir", level, True) for level in LadderLevel]
    + [("tile2d_sm80.ttir", LadderLevel.L2, False)],
)
def test_harness_parses_once_and_preserves_gate_and_clock(
    harness_probe, name, level, expected
):
    spec, events = harness_probe
    result = harness._static_track(spec, _read(name), 0, level)
    assert result["t0_gate"] is expected
    assert result["time_s"] == 2.0
    assert events == ["clock", "client_parse", "clock", "gate"]


@pytest.mark.parametrize(
    "problem", ["mutation", "different_source", "multi_graph", "wrong_mode"]
)
def test_harness_fallback_preserves_false_gate(harness_probe, monkeypatch, problem):
    spec, events = harness_probe

    def analyze(det):
        graph = det.last_ttir_graphs[0]
        if problem == "mutation":
            graph.accesses.clear()  # incorrectly trusting this gives True
        elif problem == "different_source":
            _consume(det, _read())
        elif problem == "multi_graph":
            det.last_ttir_graphs.append(graph)
        else:
            graph.multipath = False

    monkeypatch.setattr(client.CompiledRaceDetector, "_analyze_global", analyze)
    result = harness._static_track(spec, _read("tile2d_sm80.ttir"), 0, LadderLevel.L2)
    assert result["t0_gate"] is False
    assert result["time_s"] == 2.0
    assert events[-3:] == ["clock", "fallback_parse", "gate"]


@pytest.mark.parametrize("transient", [False, True])
def test_harness_parse_failure_preserves_none_or_retry_result(
    harness_probe, monkeypatch, transient
):
    spec, events = harness_probe
    if transient:

        def failed_parse(*args, **kwargs):
            events.append("client_parse")
            raise RuntimeError("first parse failed")

        monkeypatch.setattr(client, "parse_ttir", failed_parse)
    text = _read() if transient else _read("nested_loops_sm80.ttir")
    result = harness._static_track(spec, text, 0, LadderLevel.L0)
    assert result["t0_gate"] is (True if transient else None)
    assert result["parse_unsupported"]
    assert events[:4] == ["clock", "client_parse", "clock", "fallback_parse"]
    assert events[4:] == (["gate"] if transient else [])


def test_harness_gate_exception_preserves_none(harness_probe, monkeypatch):
    spec, events = harness_probe

    def failed_gate(graph):
        assert events.count("clock") == 2
        raise RuntimeError("gate failed")

    monkeypatch.setattr(global_records, "t0_linearity_gate", failed_gate)
    result = harness._static_track(spec, _read(), 0, LadderLevel.L2)
    assert result["t0_gate"] is None
    assert "fallback_parse" not in events
