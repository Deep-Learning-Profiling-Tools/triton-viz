import json

import pytest

from triton_viz.tools.nki_tensor_instruction_mix import (
    TensorInstructionMixCalibration,
    tensor_mix_features,
)


def _row(kind, free):
    return {
        "engine": "Tensor", "opcode": "MATMUL",
        "tensor_instruction_type": kind,
        "operands": f"src=fp32@0x1234[1,0,0][{free},1,1] {free}*128",
    }


def test_tensor_mix_features_ignore_addresses_and_retain_geometry():
    rows = [_row("REGULAR", 256), _row("TRANSPOSE", 128)]
    features = tensor_mix_features(rows)
    assert features[:5] == [1.0, 1.0, 2.0, 0.0, 2.0]
    assert features[6] == 256.0


def test_tensor_mix_knn_only_applies_to_mixed_stream(tmp_path):
    mixed = [_row("REGULAR", 128), _row("TRANSPOSE", 128)]
    vector = tensor_mix_features(mixed)
    path = tmp_path / "mix.json"
    path.write_text(json.dumps({
        "vectors": [vector, [value + 1 for value in vector]],
        "targets_ns": [2000, 3000], "feature_scales": [1] * 8,
        "neighbors": 2,
    }))
    calibration = TensorInstructionMixCalibration.from_json(path)
    assert calibration.predict_ns(mixed) == (2000.0, "exact_feature")
    assert calibration.predict_ns([_row("REGULAR", 128)]) == (0.0, "not_mixed")
    many = [_row("REGULAR", 128), _row("TRANSPOSE", 128)] * 4
    assert calibration.predict_ns(many) == (0.0, "instruction_count_ood")
