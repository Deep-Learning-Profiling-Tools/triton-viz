"""Report explanations survive each evaluation export without reclassification."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import harness  # noqa: E402
from triton_viz.clients.race_detector import concrete_enum  # noqa: E402
from triton_viz.clients.race_detector.data import RaceType  # noqa: E402
from triton_viz.clients.race_detector.report_diagnostics import (  # noqa: E402
    append_missing_fence_diagnostic,
)


def _report():
    first = ("kernel.py", 10, "kernel")
    second = ("kernel.py", 12, "kernel")
    reason = append_missing_fence_diagnostic(
        "existing reason",
        fence_order=True,
        same_instance=True,
        distinct_operations=True,
        first_seq=0,
        second_seq=1,
        fence_between=False,
        first_site=first,
        second_site=second,
        first_mode="write",
        second_mode="read",
    )
    return SimpleNamespace(
        first_record=SimpleNamespace(source_location=first),
        second_record=SimpleNamespace(source_location=second),
        race_type=RaceType.RAW,
        witness_grid_a=(0, 0, 0),
        witness_grid_b=(0, 0, 0),
        byte_range=(100, 104),
        reason=reason,
    )


@pytest.mark.parametrize("track", ["static", "dynamic", "enum"])
def test_report_diagnostic_survives_export_and_preserves_classification(
    monkeypatch,
    track,
):
    report = _report()
    spec = SimpleNamespace(
        make_args=lambda seed: (),
        kernel_fn=SimpleNamespace(arg_names=()),
        grid=(1,),
        constexprs={},
    )
    if track == "static":
        detector = SimpleNamespace(
            last_global_status="races",
            last_global_provenance="static",
            last_global_confirmation="confirmed",
            last_global_reason=None,
            last_global_reports=[report],
            last_grid_fragile=[report],
            last_ttir_unsupported=[],
            last_differential=None,
            last_global_assumes_termination=False,
            last_global_verdict={"proof_extent": "launch", "ladder_level": "L2"},
        )
        run = lambda: harness._static_result(detector, 0.1, True)
    elif track == "dynamic":
        import triton_viz
        from triton_viz import clients

        detector = SimpleNamespace(
            last_status="ok",
            last_reports=[report],
            unsupported_reason=None,
            last_premises=("snapshot",),
        )
        monkeypatch.setattr(clients, "RaceDetector", lambda **kwargs: detector)

        class Launcher:
            def __getitem__(self, grid):
                return lambda **kwargs: None

        monkeypatch.setattr(triton_viz, "trace", lambda det: lambda fn: Launcher())
        monkeypatch.setattr(harness, "_watchdog", lambda seconds: nullcontext())
        run = lambda: harness._dynamic_track_local(spec, 0)
    else:
        outcome = concrete_enum.EnumOutcome(status="races", reports=[report])
        monkeypatch.setattr(
            concrete_enum, "enumerate_launch", lambda *args, **kwargs: outcome
        )
        run = lambda: harness._enum_track(spec, 0, {})

    result = run()
    if track == "dynamic":
        assert result["error"] is None
    witness = json.loads(json.dumps(result))["witnesses"][0]
    assert witness["reason"] == report.reason
    assert "Missing source fence:" in witness["reason"]
    assert witness["first"] == ["kernel.py", 10, "kernel"]
    assert witness["second"] == ["kernel.py", 12, "kernel"]
    assert witness["pids"] == [[0, 0, 0], [0, 0, 0]]
    assert result["reason"] is None  # launch-level refusal classification

    # Rerender the same settled verdict with its old explanation. Every
    # existing witness field, premise, status and proof-extent field survives.
    report.reason = "existing reason"
    baseline = run()
    for row in (result, baseline):
        row.pop("time_s")
        row["witnesses"][0].pop("reason")
        for hazard in row.get("grid_fragile", []):
            hazard.pop("reason")
    assert result == baseline
