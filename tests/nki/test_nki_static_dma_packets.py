import json

import pytest

from triton_viz.tools.nki_static_dma_packets import (
    StaticDmaPacketCalibration,
    packet_features,
    packet_fingerprint,
)


def _row(queue, engine, size):
    return {
        "queue_type": queue,
        "engine_idx": engine,
        "read_bytes": size,
        "write_bytes": 0,
        "transfer_bytes": size,
    }


def test_packet_features_are_timing_free_and_fixed_width():
    rows = [_row("input", 2, 128), _row("software_dynamic", 3, 256)]
    assert len(packet_features(rows)) == 87
    assert "software_dynamic" not in packet_fingerprint(rows)


def test_packet_calibration_prefers_stable_exact_then_knn(tmp_path):
    exact_rows = [_row("input", 2, 128)]
    fallback_rows = [_row("output", 4, 512)]
    exact_vector = packet_features(exact_rows)
    fallback_vector = packet_features(fallback_rows)
    artifact = {
        "stable_exact_ns": {packet_fingerprint(exact_rows): 12.0},
        "vectors": [exact_vector, fallback_vector],
        "targets_ns": [12.0, 34.0],
        "feature_means": [0.0] * 87,
        "feature_scales": [1.0] * 87,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))
    calibration = StaticDmaPacketCalibration.from_json(path)
    assert calibration.predict_ns(exact_rows) == (12.0, "stable_exact")
    value, match = calibration.predict_ns(fallback_rows)
    assert value == pytest.approx(34.0)
    assert match == "knn1_fallback"
