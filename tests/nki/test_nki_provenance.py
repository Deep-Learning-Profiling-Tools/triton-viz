import json

from triton_viz.tools import nki_provenance
from triton_viz.tools.nki_provenance import (
    compare_fingerprints,
    make_compiler_fingerprint,
    write_experiment_manifest,
    validate_model_manifest,
    write_model_manifest,
)


def _fingerprint(**overrides):
    values = {
        "packages": {"neuronx-cc": "1.0", "triton": "3.0"},
        "tools": {"neuron-explorer": "2.0"},
        "hardware": {"machine": "aarch64", "platform": "inf2"},
        "repository_revision": "abc",
    }
    values.update(overrides)
    return make_compiler_fingerprint(**values)


def test_fingerprint_is_canonical_across_mapping_order():
    first = _fingerprint(packages={"triton": "3.0", "neuronx-cc": "1.0"})
    second = _fingerprint(packages={"neuronx-cc": "1.0", "triton": "3.0"})
    assert first == second
    assert len(first["fingerprint"]) == 20


def test_fingerprint_comparison_requires_canary_for_compiler_change():
    reference = _fingerprint()
    candidate = _fingerprint(packages={"neuronx-cc": "1.1", "triton": "3.0"})
    comparison = compare_fingerprints(reference, candidate)
    assert comparison["status"] == "compiler_stack_changed"
    assert comparison["requires_canary"] is True
    assert set(comparison["changed"]) == {"packages"}


def test_fingerprint_comparison_does_not_claim_repository_change_is_compiler_drift():
    comparison = compare_fingerprints(
        _fingerprint(), _fingerprint(repository_revision="def")
    )
    assert comparison["status"] == "repository_changed"
    assert comparison["requires_canary"] is True


def test_fingerprint_comparison_identifies_exact_and_hardware_changes():
    reference = _fingerprint()
    assert compare_fingerprints(reference, dict(reference))["status"] == "exact"
    candidate = _fingerprint(
        hardware={"machine": "aarch64", "platform": "inf2-new-host"}
    )
    assert compare_fingerprints(reference, candidate)["status"] == "hardware_changed"


def test_experiment_manifest_has_canonical_config_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nki_provenance, "collect_compiler_fingerprint", lambda root: _fingerprint()
    )
    first = write_experiment_manifest(
        tmp_path / "first",
        experiment="controls",
        config={"dims": [512, 128], "dtype": "fp32"},
    )
    second = write_experiment_manifest(
        tmp_path / "second",
        experiment="controls",
        config={"dtype": "fp32", "dims": [512, 128]},
    )
    first_data = json.loads(first.read_text())
    second_data = json.loads(second.read_text())
    assert first_data["config_hash"] == second_data["config_hash"]
    assert (
        first_data["compiler_fingerprint"]["fingerprint"]
        == _fingerprint()["fingerprint"]
    )


def test_model_manifest_checks_source_fingerprints_and_file_hashes(tmp_path):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    table = calibration / "compute.csv"
    table.write_text("engine,dtype\nvector,float32\n")
    source = tmp_path / "experiment_manifest.json"
    source.write_text(json.dumps({"compiler_fingerprint": _fingerprint()}))
    manifest = write_model_manifest(
        calibration,
        calibration_files=[table],
        source_manifests=[source],
    )
    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["calibration_source_fingerprint"] == _fingerprint()
    assert "model_builder_fingerprint" in manifest_data
    validate_model_manifest(
        manifest,
        calibration_files=[table],
        current_fingerprint=manifest_data["model_builder_fingerprint"],
    )
    table.write_text("tampered\n")
    import pytest

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_model_manifest(manifest, calibration_files=[table])


def test_model_manifest_audits_repository_only_source_changes(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    table = calibration / "compute.csv"
    table.write_text("engine,dtype\nvector,float32\n")
    clean = tmp_path / "clean.json"
    dirty = tmp_path / "dirty.json"
    clean.write_text(json.dumps({"compiler_fingerprint": _fingerprint()}))
    dirty.write_text(
        json.dumps(
            {
                "compiler_fingerprint": _fingerprint(
                    repository_dirty=True,
                    repository_diff_digest="trace-only-change",
                )
            }
        )
    )
    builder = _fingerprint(repository_revision="builder")
    monkeypatch.setattr(
        nki_provenance, "collect_compiler_fingerprint", lambda root: builder
    )

    manifest = write_model_manifest(
        calibration,
        calibration_files=[table],
        source_manifests=[clean, dirty],
    )
    data = json.loads(manifest.read_text())
    assert data["compatibility_policy"] == (
        "compiler_hardware_exact_model_builder_exact"
    )
    assert data["calibration_source_compatibility"] == [
        {
            "status": "repository_changed",
            "changed": {
                "repository_dirty": {"reference": False, "candidate": True},
                "repository_diff_digest": {
                    "reference": "",
                    "candidate": "trace-only-change",
                },
            },
            "requires_canary": True,
        }
    ]
    assert data["model_builder_fingerprint"] == builder


def test_model_manifest_rejects_compiler_source_changes(tmp_path):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    table = calibration / "compute.csv"
    table.write_text("engine,dtype\nvector,float32\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"compiler_fingerprint": _fingerprint()}))
    second.write_text(
        json.dumps(
            {
                "compiler_fingerprint": _fingerprint(
                    packages={"neuronx-cc": "2.0", "triton": "3.0"}
                )
            }
        )
    )

    import pytest

    with pytest.raises(ValueError, match="incompatible compiler fingerprints"):
        write_model_manifest(
            calibration,
            calibration_files=[table],
            source_manifests=[first, second],
        )
