"""Pins for the composed dispatcher (evaluation/harness._classify) — the
§3n content-fragile composition.

The ``race-unconfirmed`` reason marks the STRONG demotion: every widened
SAT was faithfully replayed on this launch's data and none reproduced.
§3n (decision (b)) composes it with a clean interpreter run into the
launch-scoped proof, carrying the refuted hazard as the content-fragile
attribute; every leg missing the proof stays fail-closed exactly as
before.
"""

import sys
from pathlib import Path

# the evaluation package lives at the repo root (not installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.harness import _classify  # noqa: E402

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
_GENERIC = {
    "status": "unsupported",
    "reason": (
        "possible race under over-approximation (data-dependent mask / "
        "unmodeled branch) — not a certifiable witness"
    ),
    "confirmation": None,
    "provenance": None,
}


def _dyn(status="ok", n=0, error=None):
    return {"status": status, "n_reports": n, "error": error}


def test_demoted_with_clean_interp_composes_to_proof():
    # dd_mask_dead: replay refuted the hazard AND the interpreter ran
    # this launch clean — the composition owes the launch-scoped proof
    assert _classify(_DEMOTED, _dyn()) == ("race-free", "proved@interp")


def test_demoted_with_interp_reports_is_a_race():
    # concrete interp reports subsume the widened hazard
    assert _classify(_DEMOTED, _dyn(n=2)) == ("race", "race@interp")


def test_demoted_without_dynamic_stays_fail_closed():
    assert _classify(_DEMOTED, None) == ("abstain", "race-unconfirmed")


def test_demoted_with_failed_dynamic_stays_fail_closed():
    assert _classify(_DEMOTED, _dyn(status="timeout")) == (
        "abstain",
        "race-unconfirmed",
    )
    assert _classify(_DEMOTED, _dyn(error="SIGSEGV")) == (
        "abstain",
        "race-unconfirmed",
    )


def test_generic_demotion_keeps_the_plain_composition():
    # capped / unavailable / unclassifiable demotions never carry the
    # race-unconfirmed marker: they follow the ORDINARY static-abstain
    # composition (unchanged by §3n) and never earn the attribute — the
    # attribute stamp in run_one keys on the marker, absent here
    assert _classify(_GENERIC, _dyn()) == ("race-free", "proved@interp")
    assert _classify(_GENERIC, None) == ("abstain", "unsupported")


def test_two_run_determinism():
    for static, dyn in [
        (_DEMOTED, _dyn()),
        (_DEMOTED, None),
        (_DEMOTED, _dyn(n=1)),
        (_GENERIC, _dyn()),
    ]:
        assert _classify(static, dyn) == _classify(static, dyn)
