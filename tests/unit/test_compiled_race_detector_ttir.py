"""Unit tests for the compiled race detector's TTIR capture (the Track 2
global-memory front-end): warmup capture, per-specialization parse cache,
and independence from the TTGIR shared-memory verdict."""

from pathlib import Path
from types import SimpleNamespace

from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / name).read_text()


def _launch(det, asm):
    # Production-faithful: the warmup-only path never calls grid_callback,
    # so finalize() must be the per-launch reset point on its own.
    det.post_warmup_callback(None, SimpleNamespace(asm=asm))
    return det.finalize()


def test_captures_and_parses_ttir_alongside_ttgir():
    det = CompiledRaceDetector()
    _launch(det, {"ttgir": _read("add_sm80.ttgir"), "ttir": _read("add_sm80.ttir")})
    assert det.last_status == "ok"  # TTGIR shared-memory path unchanged
    (g,) = det.last_ttir_graphs
    assert g is not None
    assert [a.kind for a in g.accesses] == ["load", "load", "store"]
    assert det.last_ttir_unsupported == [None]


def test_atomic_rmw_reaches_the_graph():
    det = CompiledRaceDetector()
    _launch(det, {"ttir": _read("atomic_sm80.ttir")})
    # No TTGIR captured, so the shared-memory verdict says so — but the TTIR
    # capture is independent and must still have parsed.
    assert det.last_status == "no_ttgir"
    (g,) = det.last_ttir_graphs
    assert g is not None
    assert [a.kind for a in g.accesses] == ["load", "atomic_rmw", "atomic_rmw", "store"]
    assert g.accesses[1].atomic is not None
    assert g.accesses[1].atomic.rmw_op == "fadd"


def test_unsupported_ttir_never_touches_status():
    det = CompiledRaceDetector()
    _launch(det, {"ttgir": _read("add_sm80.ttgir"), "ttir": _read("gather_sm80.ttir")})
    assert det.last_status == "ok"  # TTGIR verdict untouched by the TTIR failure
    assert det.last_ttir_graphs == [None]
    assert det.last_ttir_unsupported[0]  # the reason string is recorded


def test_parse_cache_reuses_graph_across_launches():
    det = CompiledRaceDetector()
    text = _read("atomic_sm80.ttir")
    _launch(det, {"ttir": text})
    first = det.last_ttir_graphs[0]
    assert first is not None
    _launch(det, {"ttir": text})
    assert det.last_ttir_graphs[0] is first  # per-specialization cache hit


def test_warmup_only_lifecycle_never_accumulates():
    """Without grid_callback (the production warmup-only lifecycle), each
    finalize must expose exactly the CURRENT launch's parse results."""
    det = CompiledRaceDetector()
    _launch(det, {"ttir": _read("atomic_sm80.ttir")})
    assert len(det.last_ttir_graphs) == 1
    _launch(det, {"ttir": _read("add_sm80.ttir")})
    assert len(det.last_ttir_graphs) == 1  # not [atomic, add]
    (g,) = det.last_ttir_graphs
    assert [a.kind for a in g.accesses] == ["load", "load", "store"]
    # A launch whose warmup delivers no asm exposes no stale graphs.
    det.finalize()
    assert det.last_ttir_graphs == []
    assert det.last_ttir_unsupported == []
