import json

import pytest

from triton_viz.performance import NkiBackend, predict_latency
from triton_viz.tools import gpu_cost_model_pipeline as pipeline


def test_nki_adapter_preserves_exact_existing_prediction():
    from triton_viz.tools.nki_cost_model import CostModel, simulate

    events = [{"seq": 0, "op": "load", "engine": "dma", "bytes": 1024}]
    model = CostModel()
    expected = simulate(events, model)
    result = predict_latency(events, NkiBackend(model))
    assert result.latency_ns == expected.predicted_latency_ns
    assert result.diagnostics == expected.as_dict()


def test_fit_never_opens_holdouts_and_freezes_only_passing_controls(
    tmp_path, monkeypatch
):
    cases = [{"id": str(n)} for n in (10, 20, 40, 80)]
    pipeline._write(
        tmp_path / "manifest.json",
        {
            "fingerprint": "test",
            "splits": {"control": cases, "holdout": [{"id": "forbidden"}]},
        },
    )
    for case in cases:
        n = int(case["id"])
        features = dict.fromkeys(pipeline.FEATURES, 0)
        features.update(launch=1, global_sectors=n)
        pipeline._write(
            tmp_path / "controls" / (case["id"] + ".json"),
            {
                "role": "control",
                "contaminated": False,
                "fingerprint": "test",
                "cv_group": str(n),
                "features": features,
                "latency_us": 2 + 0.01 * n,
                "source_configuration": [4, 2, ["fp32"]],
            },
        )
    original = pipeline._read

    def guarded(path):
        assert "holdouts" not in path.parts
        return original(path)

    monkeypatch.setattr(pipeline, "_read", guarded)
    pipeline.fit(tmp_path)
    frozen = json.loads((tmp_path / "calibration" / "frozen.json").read_text())
    assert frozen["cv"]["passed"]
    assert frozen["digest"]
    pipeline._write(
        tmp_path / "calibration" / "fit_status.json",
        {"passed": False, "candidate_digest": "rejected-new-fit"},
    )
    with pytest.raises(ValueError, match="stale"):
        pipeline.evaluate(tmp_path)


def test_evaluate_rejects_tampered_frozen_calibration_before_reading_targets(tmp_path):
    pipeline._write(tmp_path / "manifest.json", {})
    pipeline._write(tmp_path / "calibration" / "frozen.json", {"digest": "invalid"})
    with pytest.raises(ValueError, match="digest mismatch"):
        pipeline.evaluate(tmp_path)
