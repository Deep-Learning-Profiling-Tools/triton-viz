"""READY admission never grants GO and preserves failures without a fallback."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation import dynamic_transport_admission as admission
from evaluation.spec import LaunchSpec


def _spec():
    base = torch.arange(16, dtype=torch.int32)
    return LaunchSpec(
        name="admission_views",
        kernel_fn=SimpleNamespace(arg_names=["x", "out", "N"]),
        signature={"x": "*i32", "out": "*i32", "N": "constexpr"},
        constexprs={"N": 4},
        grid=(4,),
        make_args=lambda seed: (base[2:10:2], base[3:11:2]),
    )


@pytest.fixture
def fake_child(monkeypatch):
    processes = []
    monkeypatch.setattr(admission, "_helper_identity", lambda: {"commit": "test"})
    monkeypatch.setattr(
        admission.transport, "source_identity", lambda: {"tree": "bound"}
    )
    monkeypatch.setattr(
        admission.transport, "_kernel_identity", lambda kernel: {"source": "bound"}
    )
    monkeypatch.setattr(
        admission.transport,
        "_serialize_spec",
        lambda spec: b"production-serializer-control",
    )

    def install(mode="ready"):
        class Process:
            pid = 123456

            def __init__(self, command, **kwargs):
                self.returncode = None
                self.reaped = False
                self.path = Path(command[-1])
                assert command[1:4] == [
                    "-m",
                    "evaluation.dynamic_subprocess",
                    "--child",
                ]
                assert kwargs["env"]["PYTHONPATH"]
                assert self.path.stat().st_mode & 0o077 == 0
                request = admission.transport._read(self.path / "request.json")
                inputs = torch.load(self.path / "inputs.pt", weights_only=False)
                assert (
                    inputs[0].untyped_storage()._cdata
                    == inputs[1].untyped_storage()._cdata
                )
                assert inputs[0].storage_offset() == 2
                assert inputs[1].storage_offset() == 3
                assert request["hooks"] == []
                if mode != "timeout":
                    ready = {
                        "input_sha256": admission.transport._hash(request["inputs"]),
                        "kernel_sha256": admission.transport._hash(request["kernel"]),
                        "source": request["source"],
                    }
                    if mode == "mismatch":
                        ready["input_sha256"] = "wrong"
                    if mode == "error":
                        self.returncode = 2
                        admission.transport._json(
                            self.path / "error.json", {"message": "bad callable"}
                        )
                    else:
                        admission.transport._json(self.path / "ready.json", ready)
                    if mode == "forbidden":
                        admission.transport._json(
                            self.path / "result.json", {"unexpected": True}
                        )
                processes.append(self)

            def poll(self):
                assert not (self.path / "go.json").exists()
                return self.returncode

            def kill(self):
                self.returncode = -9

            def wait(self):
                assert not (self.path / "go.json").exists()
                self.reaped = True
                return self.returncode

        monkeypatch.setattr(admission.subprocess, "Popen", Process)
        return processes

    return install


def test_ready_is_bound_then_reaped_without_analysis(fake_child):
    processes = fake_child()
    result = admission.admit(_spec(), level=2)
    assert result["status"] == "admitted", result
    assert result["ready_verified"]
    assert not result["go_written"]
    assert result["analysis_ran"] is False
    assert result["fallback_used"] is False
    assert result["analysis_artifacts"] == []
    assert result["child_reaped"] and processes[0].reaped
    assert not processes[0].path.exists()  # private transport removed
    assert result["request"]["level"] == 2
    assert result["hashes"]["request"] == admission.transport._hash(result["request"])
    assert (
        result["request"]["inputs"]["x"]["alias_group"]
        == result["request"]["inputs"]["out"]["alias_group"]
    )
    assert result["request"]["inputs"]["x"]["storage_sha256"]
    assert result["spawn_to_reap_s"] >= result["spawn_to_ready_s"]
    assert result["full_wall_s"] >= result["spawn_to_reap_s"]


@pytest.mark.parametrize("mode", ["mismatch", "error", "timeout", "forbidden"])
def test_failed_transport_is_not_admitted_or_retried(fake_child, mode):
    processes = fake_child(mode)
    result = admission.admit(
        _spec(), setup_timeout_s=0.001 if mode == "timeout" else 60
    )
    assert result["status"] == "error", result
    assert result["errors"]
    assert not result["go_written"]
    assert not result["fallback_used"]
    assert len(processes) == 1
    assert processes[0].reaped
    assert not processes[0].path.exists()
    if mode == "error":
        assert result["error"] == {"message": "bad callable"}
    if mode == "forbidden":
        assert result["analysis_ran"] is None
    else:
        assert result["analysis_ran"] is False


def test_serialization_error_is_recorded_without_a_child(fake_child, monkeypatch):
    processes = fake_child()

    def fail(spec):
        raise TypeError("cannot serialize this kernel")

    monkeypatch.setattr(admission.transport, "_serialize_spec", fail)
    result = admission.admit(_spec())
    assert result["status"] == "error"
    assert result["errors"] == [
        {"type": "TypeError", "message": "cannot serialize this kernel"}
    ]
    assert result["inputs"]["x"]["storage_offset"] == 2
    assert not processes


def test_external_builder_checks_source_before_execution(tmp_path, monkeypatch):
    module = tmp_path / "admission_builder_control.py"
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef build(case):\n    return case, {{'source': 'control'}}\n"
    module.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = tmp_path / "builder.json"
    description = {
        "module": module.stem,
        "module_sha256": "wrong",
        "function": "build",
        "kwargs": {"case": "nested-control"},
    }
    config.write_text(json.dumps(description))
    with pytest.raises(ValueError, match="source hash mismatch"):
        admission._external_builder(config)
    assert not marker.exists()
    description["module_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    config.write_text(json.dumps(description))
    result, receipt = admission._external_builder(config)
    assert marker.exists()
    assert result == "nested-control"
    assert receipt["config"] == description
    assert receipt["metadata"] == {"source": "control"}
