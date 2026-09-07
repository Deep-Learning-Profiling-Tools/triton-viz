"""L2 stops at a deciding frontend while retaining fallback evidence and scope."""

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import harness  # noqa: E402
from evaluation.frontend_policy import frontend_policy  # noqa: E402
from evaluation.spec import LaunchSpec  # noqa: E402
from triton_viz.clients.race_detector.ladder import LadderLevel  # noqa: E402


def _static(status="ok", **overrides):
    return {
        "status": status,
        "provenance": "proved@T1" if status == "ok" else None,
        "confirmation": None,
        "reason": None,
        "n_reports": 0,
        "witnesses": [],
        "verdict_attrs": {},
        **overrides,
    }


def _dynamic(**overrides):
    return {
        "status": "ok",
        "error": None,
        "n_reports": 0,
        "premises": [],
        "witnesses": [],
        **overrides,
    }


@pytest.fixture
def stages(monkeypatch):
    monkeypatch.delenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", raising=False)
    calls = []
    names = {
        "compile": "_host_compile_ttir",
        "static": "_static_track",
        "cutile": "_static_track_cutile",
        "dynamic": "_dynamic_track",
        "enum": "_enum_track",
        "mutation": "_mutation_track",
    }

    def forbidden(*args, **kwargs):
        # pytest.fail escapes the harness's Exception handlers, so an
        # accidentally executed stage cannot masquerade as a caught error.
        pytest.fail("a stage that should have been skipped was executed")

    for name in names.values():
        monkeypatch.setattr(harness, name, forbidden)

    def set_track(name, result):
        def track(*args, **kwargs):
            calls.append(name)
            if isinstance(result, BaseException):
                raise result
            return copy.deepcopy(result)

        monkeypatch.setattr(harness, names[name], track)

    set_track("compile", "module {}")
    spec = LaunchSpec(
        name="frontend_policy_probe",
        kernel_fn=lambda: None,
        signature={},
        constexprs={},
        make_args=lambda seed: (),
        grid=(2,),
    )
    return SimpleNamespace(spec=spec, calls=calls, set_track=set_track)


@pytest.mark.parametrize("setting", [None, "", "0"])
def test_default_policy_depends_on_ladder_level(stages, monkeypatch, setting):
    if setting is not None:
        monkeypatch.setenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", setting)
    assert frontend_policy(LadderLevel.L0) == "all"
    assert frontend_policy(LadderLevel.L1) == "all"
    assert frontend_policy(LadderLevel.L2) == "on-demand"


@pytest.mark.parametrize("setting", ["true", "yes", "2", "on-demand"])
def test_invalid_policy_setting_is_rejected(stages, monkeypatch, setting):
    monkeypatch.setenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", setting)
    with pytest.raises(ValueError):
        frontend_policy(LadderLevel.L2)


@pytest.mark.parametrize("provenance", ["proved@T0", "proved@T1", "proved@T1-launch"])
def test_static_proof_skips_later_frontends_without_widening_scope(stages, provenance):
    static = _static(
        provenance=provenance,
        assumes_termination=True,
        grid_fragile=[{"hazard": "WRITE_WRITE", "first": 4, "second": 9}],
        verdict_attrs={
            "proved_scope": "this-params-this-grid",
            "content_fragile": True,
            "assumes_termination": True,
            "premises": ["contents-snapshot"],
        },
    )
    stages.set_track("static", static)

    row = harness.run_one(stages.spec, 7, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static"]
    assert (row["verdict"], row["terminal"]) == ("race-free", provenance)
    assert row["static"] == static
    assert row["frontend_policy"] == "on-demand"
    assert row["dynamic"]["status"] == "not-run"
    assert row["dynamic"]["reason"] == "static-decided"
    assert row["dynamic"]["time_s"] is None
    assert row["dynamic"]["n_reports"] == 0
    assert row["dynamic"]["witnesses"] == []
    assert "enum" not in row


@pytest.mark.parametrize(
    "confirmation,terminal",
    [("confirmed", "race-confirmed"), (None, "races-unclassified")],
)
def test_static_report_is_also_a_decision(stages, confirmation, terminal):
    static = _static(
        "races",
        confirmation=confirmation,
        n_reports=1,
        witnesses=[{"first": 4, "second": 9, "pids": [[0], [1]]}],
    )
    stages.set_track("static", static)

    row = harness.run_one(stages.spec, 0, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static"]
    assert (row["verdict"], row["terminal"]) == ("race", terminal)
    assert row["static"] == static
    assert row["dynamic"]["status"] == "not-run"
    assert "enum" not in row


def test_static_proof_still_runs_requested_mutation_analysis(stages):
    stages.set_track("static", _static())
    mutation = {"pid_pin": {"status": "races", "n_reports": 1}}
    stages.set_track("mutation", mutation)

    row = harness.run_one(stages.spec, 0, mutate=True, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static", "mutation"]
    assert row["terminal"] == "proved@T1"
    assert row["mutation"] == mutation
    assert row["dynamic"]["status"] == "not-run"


@pytest.mark.parametrize(
    "level,all_frontends",
    [(LadderLevel.L0, None), (LadderLevel.L1, None), (LadderLevel.L2, "1")],
)
def test_comparison_policy_runs_dynamic_even_after_static_proof(
    stages, monkeypatch, level, all_frontends
):
    if all_frontends is not None:
        monkeypatch.setenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", all_frontends)
    stages.set_track("static", _static())
    stages.set_track("dynamic", _dynamic(status="timeout", error="deadline"))

    row = harness.run_one(stages.spec, 0, ladder_level=level)

    assert stages.calls == ["compile", "static", "dynamic"]
    assert row["frontend_policy"] == "all"
    assert row["terminal"] == "proved@T1"
    assert row["dynamic"]["status"] == "timeout"
    assert "enum" not in row


@pytest.mark.parametrize("n_reports", [0, 1])
@pytest.mark.parametrize("refuted_hazard", [False, True])
def test_static_abstention_uses_dynamic_decision_and_retains_premises(
    stages, n_reports, refuted_hazard
):
    reason = "race-unconfirmed: faithfully refuted" if refuted_hazard else "nested-loop"
    static = _static("unsupported", reason=reason)
    dynamic = _dynamic(n_reports=n_reports, premises=["contents-snapshot"])
    stages.set_track("static", static)
    stages.set_track("dynamic", dynamic)

    row = harness.run_one(stages.spec, 0, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static", "dynamic"]
    assert (row["verdict"], row["terminal"]) == (
        ("race", "race@interp") if n_reports else ("race-free", "proved@interp")
    )
    assert row["dynamic"] == dynamic
    assert "enum" not in row
    attrs = row["static"]["verdict_attrs"]
    assert attrs.get("content_fragile", False) is (refuted_hazard and not n_reports)


@pytest.mark.parametrize(
    "dynamic",
    [
        _dynamic(status="unsupported"),
        _dynamic(status="timeout", error="deadline"),
        _dynamic(error="failed after producing a provisional result"),
        RuntimeError("interpreter failed"),
    ],
)
def test_failed_dynamic_falls_back_to_enum_without_claiming_interpreter_proof(
    stages, dynamic
):
    stages.set_track(
        "static", _static("unsupported", reason="race-unconfirmed: faithfully refuted")
    )
    stages.set_track("dynamic", dynamic)
    stages.set_track("enum", {"status": "ok", "n_reports": 0, "reason": None})

    row = harness.run_one(stages.spec, 0, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static", "dynamic", "enum"]
    assert (row["verdict"], row["terminal"]) == ("race-free", "proved@enum")
    assert row["static"]["verdict_attrs"]["proved_scope"] == "this-params-this-grid"
    assert row["static"]["verdict_attrs"]["content_fragile"] is True


@pytest.mark.parametrize(
    "enum,expected",
    [
        ({"status": "races", "n_reports": 1}, ("race", "race@enum")),
        (
            {"status": "unsupported", "reason": "spin-shape"},
            ("abstain", "race-unconfirmed"),
        ),
    ],
)
def test_final_fallback_preserves_race_and_refusal_outcomes(stages, enum, expected):
    stages.set_track(
        "static", _static("unsupported", reason="race-unconfirmed: faithfully refuted")
    )
    stages.set_track("dynamic", _dynamic(status="unsupported"))
    stages.set_track("enum", enum)

    row = harness.run_one(stages.spec, 0, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static", "dynamic", "enum"]
    assert (row["verdict"], row["terminal"]) == expected
    if expected[0] == "abstain":
        assert not row["static"]["verdict_attrs"].get("content_fragile")


def test_static_exception_remains_a_harness_error(stages):
    stages.set_track("static", RuntimeError("broken static frontend"))

    row = harness.run_one(stages.spec, 0, ladder_level=LadderLevel.L2)

    assert stages.calls == ["compile", "static"]
    assert (row["verdict"], row["terminal"]) == ("error", "harness-error")
    assert "broken static frontend" in row["harness_error"]
    assert row["frontend_policy"] == "on-demand"
    assert "dynamic" not in row


@pytest.mark.parametrize("all_frontends", [None, "1"])
@pytest.mark.parametrize("status", ["ok", "unsupported"])
def test_cutile_never_fabricates_an_interpreter(
    stages, monkeypatch, all_frontends, status
):
    if all_frontends is not None:
        monkeypatch.setenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", all_frontends)
    spec = replace(stages.spec, frontend="cutile", kernel_fn=None)
    stages.set_track("cutile", _static(status, reason=None if status == "ok" else "x"))

    row = harness.run_one(spec, 0, ladder_level=LadderLevel.L2)

    assert stages.calls == ["cutile"]
    assert row["frontend_policy"] == ("all" if all_frontends else "on-demand")
    assert row["dynamic"]["status"] == "unsupported"
    assert "no interpreter" in row["dynamic"]["reason"]
    assert "enum" not in row
    assert row["verdict"] == ("race-free" if status == "ok" else "abstain")
