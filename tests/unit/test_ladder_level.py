"""Pins for the ladder-depth switch (Hao, 2026-09-04): one configuration,
three levels, stamped everywhere, consulted at exactly one gate.

* ``LadderLevel`` parsing is strict (provenance must not degrade to L0 on
  a typo);
* the clients accept the level as a constructor parameter (the
  ``ablations`` precedent) and the compiled client stamps it into the
  verdict attributes;
* the results-JSONL header carries it;
* ``_classify`` composes the L1 rung's row only when the symbolic
  composition is an abstention, and every refusal keeps the abstention.
"""

import sys
from pathlib import Path

import pytest

# the evaluation package lives at the repo root (not installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.harness import _classify, _enum_budget_s  # noqa: E402
from evaluation.runner import (  # noqa: E402
    PER_SPEC_TIMEOUT_L1_S,
    PER_SPEC_TIMEOUT_S,
    results_header,
    row_timeout_s,
)
from triton_viz.clients.race_detector.ladder import (  # noqa: E402
    DEFAULT_LADDER_LEVEL,
    LADDER_LEVEL_NAMES,
    LadderLevel,
    parse_ladder_level,
)


def test_default_is_l0_and_levels_are_ordered():
    assert DEFAULT_LADDER_LEVEL is LadderLevel.L0
    assert LadderLevel.L0 < LadderLevel.L1 < LadderLevel.L2
    assert LADDER_LEVEL_NAMES == ("L0", "L1", "L2")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("L1", LadderLevel.L1),
        ("l2", LadderLevel.L2),
        (" L0 ", LadderLevel.L0),
        (1, LadderLevel.L1),
        ("2", LadderLevel.L2),
        (LadderLevel.L1, LadderLevel.L1),
    ],
)
def test_parse_accepts_names_and_numbers(value, expected):
    assert parse_ladder_level(value) is expected


@pytest.mark.parametrize("value", ["L3", "level1", "", None, 3, True, 1.0])
def test_parse_rejects_anything_else(value):
    with pytest.raises(ValueError):
        parse_ladder_level(value)


def test_clients_take_the_level_as_a_constructor_parameter():
    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
    from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector

    assert SymbolicRaceDetector().ladder_level is LadderLevel.L0
    assert (
        SymbolicRaceDetector(ladder_level=LadderLevel.L1).ladder_level is LadderLevel.L1
    )
    assert CompiledRaceDetector().ladder_level is LadderLevel.L0
    det = CompiledRaceDetector(ladder_level="L1")
    assert det.ladder_level is LadderLevel.L1


def test_compiled_client_stamps_the_level_into_verdict_attrs():
    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    det = CompiledRaceDetector(
        confirm_races=False, differential_check=False, ladder_level=LadderLevel.L1
    )
    det.last_global_status = "unsupported"
    det.last_global_reason = "other: probe"
    det._emit_verdict_attributes([])
    assert det.last_global_verdict["ladder_level"] == "L1"
    det0 = CompiledRaceDetector(confirm_races=False, differential_check=False)
    det0.last_global_status = "unsupported"
    det0.last_global_reason = "other: probe"
    det0._emit_verdict_attributes([])
    assert det0.last_global_verdict["ladder_level"] == "L0"


def test_results_header_carries_the_level_and_the_row_budget():
    h = results_header("golden_smoke", 0, {"upstream": "abc"}, LadderLevel.L1)
    assert h["header"] is True
    assert h["ladder_level"] == "L1"
    assert h["row_timeout_s"] == 200
    assert h["upstream"] == "abc"
    h0 = results_header("golden_smoke", 0, {})
    assert h0["ladder_level"] == "L0"
    assert h0["row_timeout_s"] == 180
    assert (
        results_header("golden_smoke", 0, {}, LadderLevel.L1, 240)["row_timeout_s"]
        == 240
    )


def test_row_budget_is_level_dependent():
    # L0 keeps the paper's protocol; L1 runs a third track after the two
    # symbolic ones and gets 200 s (Hao, 2026-09-04)
    assert PER_SPEC_TIMEOUT_S == 180
    assert PER_SPEC_TIMEOUT_L1_S == 200
    assert row_timeout_s(LadderLevel.L0) == 180
    assert row_timeout_s(LadderLevel.L1) == 200
    assert row_timeout_s(LadderLevel.L2) == 200


def test_enum_budget_is_the_remaining_row_budget():
    import time

    now = time.perf_counter()
    # nothing spent yet: 200 - 10 margin
    assert abs(_enum_budget_s(now, LadderLevel.L1) - 190.0) < 1.0
    # the symbolic tracks took 100 s: 90 s remain
    assert abs(_enum_budget_s(now - 100.0, LadderLevel.L1) - 90.0) < 1.0
    # floored so a spin still ends in a named refusal
    assert _enum_budget_s(now - 1000.0, LadderLevel.L1) == 30.0


# ── the composed dispatcher with the L1 leg ────────────────────────

_GENERIC = {
    "status": "unsupported",
    "reason": "nested-loop: line 17: multiple/nested loops",
    "confirmation": None,
    "provenance": None,
}
_DEMOTED = {
    "status": "unsupported",
    "reason": (
        "race-unconfirmed: possible race under over-approximation "
        "(data-dependent mask / unmodeled branch); the interpreter "
        "replay did not reproduce it on this launch's data"
    ),
    "confirmation": None,
    "provenance": None,
}
_PROVED = {
    "status": "ok",
    "provenance": "proved@T1",
    "confirmation": None,
    "reason": None,
}


def _dyn(status="unsupported", n=0, error=None):
    return {"status": status, "n_reports": n, "error": error}


def _enum(status="ok", n=0, reason=None):
    return {"status": status, "n_reports": n, "reason": reason}


def test_without_the_enum_row_the_composition_is_unchanged():
    assert _classify(_GENERIC, _dyn()) == ("abstain", "unsupported")
    assert _classify(_GENERIC, _dyn(), None) == ("abstain", "unsupported")
    assert _classify(_DEMOTED, _dyn()) == ("abstain", "race-unconfirmed")


def test_clean_enumeration_decides_proved_at_enum():
    assert _classify(_GENERIC, _dyn(), _enum()) == ("race-free", "proved@enum")
    assert _classify(_DEMOTED, _dyn(status="timeout"), _enum()) == (
        "race-free",
        "proved@enum",
    )
    assert _classify(_GENERIC, None, _enum()) == ("race-free", "proved@enum")


def test_concrete_witnesses_decide_race_at_enum():
    assert _classify(_GENERIC, _dyn(), _enum("races", n=2)) == ("race", "race@enum")


def test_every_refusal_keeps_the_abstention():
    for reason in (
        "atomic-return: an atomic return value reaches an address",
        "value-source: the load overlaps bytes written",
        "instance-ceiling: 2031616 program instances exceed ENUM_MAX_INSTANCES=65536",
        "spin-shape: await-bearing kernel (static reader)",
        "timeout: concrete enumeration exceeded 60s",
    ):
        assert _classify(_GENERIC, _dyn(), _enum("unsupported", reason=reason)) == (
            "abstain",
            "unsupported",
        )
    assert _classify(_DEMOTED, _dyn(), _enum("unsupported", reason="x: y")) == (
        "abstain",
        "race-unconfirmed",
    )


def test_the_rung_never_preempts_a_symbolic_decision():
    # a decided row keeps its terminal even if an enum row were present
    assert _classify(_PROVED, _dyn(), _enum("races", n=1)) == ("race-free", "proved@T1")
    assert _classify(_GENERIC, _dyn(status="ok"), _enum("races", n=1)) == (
        "race-free",
        "proved@interp",
    )
    assert _classify(_GENERIC, _dyn(status="ok", n=1), _enum()) == (
        "race",
        "race@interp",
    )


def test_two_run_determinism():
    for static, dyn, enum in [
        (_GENERIC, _dyn(), _enum()),
        (_GENERIC, _dyn(), _enum("races", n=1)),
        (_DEMOTED, _dyn(), _enum("unsupported", reason="k: d")),
    ]:
        assert _classify(static, dyn, enum) == _classify(static, dyn, enum)
