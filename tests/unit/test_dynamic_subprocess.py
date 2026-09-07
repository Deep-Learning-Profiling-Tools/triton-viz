"""Process containment preserves the launch and never credits a late exit."""
from dataclasses import replace
import io
import sys

import pytest
import torch
import triton
import triton.language as tl

from evaluation import dynamic_subprocess as child
from evaluation import harness
from evaluation.spec import LaunchSpec
from triton_viz.clients.race_detector.ladder import LadderLevel


@triton.jit
def _copy(x, out, N: tl.constexpr):
    i = tl.program_id(0)
    tl.store(out + i, tl.load(x + i), i < N)


@triton.jit
def _attribute_exp(x, out, N: tl.constexpr):
    i = tl.program_id(0)
    tl.store(out + i, tl.exp(tl.load(x + i)), i < N)


@triton.jit
def _helper_exp(x):
    return tl.exp(x)


@triton.jit
def _replacement_exp(x):
    return tl.exp(x) + 1


def _spec():
    return LaunchSpec(
        name="subprocess_copy",
        kernel_fn=_copy,
        signature={"x": "*fp32", "out": "*fp32", "N": "constexpr"},
        constexprs={"N": 4},
        grid=(4,),
        make_args=lambda seed: (torch.arange(4, dtype=torch.float32), torch.zeros(4)),
    )


def test_bound_input_transport_preserves_aliases_views_and_reinterpretation():
    base = torch.arange(32, dtype=torch.int32)
    inputs = (base[2:22:2], triton.reinterpret(base[2:22:2], tl.uint32))
    spec = _spec()
    before = child.input_identity(spec, inputs)
    data = io.BytesIO()
    torch.save(inputs, data)
    data.seek(0)
    after = torch.load(data, weights_only=False)
    assert child.input_identity(spec, after) == before
    assert before["x"]["alias_group"] == before["out"]["alias_group"]
    assert before["x"]["stride"] == [2]
    assert before["out"]["reinterpret_dtype"] == "uint32"
    base[3] = 1000  # backing gap is outside both logical strided views
    changed = child.input_identity(spec, inputs)
    assert changed["x"]["sha256"] == before["x"]["sha256"]
    assert changed["x"]["storage_sha256"] != before["x"]["storage_sha256"]


def test_callable_transport_discards_jit_caches_and_preserves_source():
    cloudpickle = pytest.importorskip("cloudpickle")
    spec = _spec()
    spec.kernel_fn.device_caches["unserializable-control"] = lambda: sys.stdout
    try:
        restored = cloudpickle.loads(child._serialize_spec(spec))
    finally:
        spec.kernel_fn.device_caches.pop("unserializable-control")
    assert child._kernel_identity(restored.kernel_fn) == child._kernel_identity(
        spec.kernel_fn
    )
    assert restored.make_args is None
    assert not restored.kernel_fn.device_caches


def test_attribute_name_collision_is_not_an_unused_global_dependency(monkeypatch):
    cloudpickle = pytest.importorskip("cloudpickle")
    # FLA imports a global exp helper but this kernel reads only tl.exp.
    monkeypatch.setitem(globals(), "exp", _helper_exp)
    spec = replace(_spec(), kernel_fn=_attribute_exp)
    before = child._kernel_identity(spec.kernel_fn)
    restored = cloudpickle.loads(child._serialize_spec(spec))
    assert child._kernel_identity(restored.kernel_fn) == before
    assert set(before["dependency_sources"]) == {"kernel.tl"}
    assert before["dependency_sources"]["kernel.tl"]["sha256"]


def test_real_helper_and_closure_dependencies_remain_strict(monkeypatch):
    cloudpickle = pytest.importorskip("cloudpickle")
    offset = 3

    @triton.jit
    def with_helper(x, out, N: tl.constexpr):
        i = tl.program_id(0)
        tl.store(out + i, _helper_exp(tl.load(x + i)) + offset, i < N)

    spec = replace(_spec(), kernel_fn=with_helper)
    before = child._kernel_identity(with_helper)
    dependencies = before["dependency_sources"]
    assert dependencies["kernel._helper_exp"]["source"] == _helper_exp.src
    assert dependencies["kernel._helper_exp.tl"]["sha256"]
    assert dependencies["kernel.offset"] == {"constant": 3}
    restored = cloudpickle.loads(child._serialize_spec(spec))
    assert child._kernel_identity(restored.kernel_fn) == before
    monkeypatch.setitem(
        restored.kernel_fn.fn.__globals__, "_helper_exp", _replacement_exp
    )
    assert child._kernel_identity(restored.kernel_fn) != before
    with_helper.fn.__closure__[0].cell_contents = 4
    assert child._kernel_identity(with_helper) != before


def test_fresh_process_reports_launch_identity_and_actual_wall():
    pytest.importorskip("cloudpickle")
    spec = _spec()
    result = harness._dynamic_track(spec, 0, LadderLevel.L0)
    assert result["status"] == "ok", result
    assert result["n_reports"] == 0
    execution = result["execution"]
    assert execution["protocol"] == "dynamic-spawn-v1"
    assert execution["child_exit_code"] == 0
    assert execution["child_completed"]
    assert execution["child_internal_time_s"] <= result["time_s"]
    assert (
        execution["full_wall_s"]
        >= execution["startup_s"] + execution["parent_ready_to_reap_s"]
    )
    assert execution["input_sha256"] == child._hash(
        child.input_identity(spec, spec.make_args(0))
    )


class _DiagnosticObserver:
    def __init__(self, mode):
        self.mode = mode

    def begin(self):
        if self.mode == "late-exit":
            import atexit
            import time

            atexit.register(time.sleep, 2)
        elif self.mode == "cleanup":
            import time
            import triton_viz
            import z3

            class NativeOwner:
                def __init__(self):
                    self.ast = z3.Int("cleanup_owned") + 1

                def __del__(self):
                    time.sleep(2)
                    del self.ast

            class Launcher:
                def __getitem__(self, grid):
                    def launch(**kwargs):
                        owned = NativeOwner()
                        try:
                            while True:
                                time.sleep(0.001)
                        finally:
                            del owned

                    return launch

            triton_viz.trace = lambda det: lambda fn: Launcher()

    def snapshot(self):
        if self.mode == "snapshot-error":
            raise RuntimeError("deliberate instrumentation failure")
        return {"partial": True, "mode": self.mode}

    def finish(self):
        return {"partial": False, "mode": self.mode}


def _make_observer(mode):
    return _DiagnosticObserver(mode)


@pytest.mark.parametrize("mode", ["late-exit", "cleanup"])
def test_deadline_includes_finalization_and_process_exit(monkeypatch, mode):
    pytest.importorskip("cloudpickle")
    monkeypatch.setattr(harness, "DYNAMIC_TIMEOUT_S", 0.4)
    monkeypatch.setattr(
        harness,
        "DYNAMIC_CHILD_HOOKS",
        (
            {
                "name": "diagnostic",
                "module": __name__,
                "factory": "_make_observer",
                "kwargs": {"mode": mode},
            },
        ),
    )
    result = harness._dynamic_track(_spec(), 0)
    assert result["status"] == "timeout", result
    assert result["n_reports"] == 0
    assert result["premises"] == []
    assert result["witnesses"] == []
    execution = result["execution"]
    assert execution["parent_ready_to_reap_s"] >= 0.4
    assert execution["parent_ready_to_reap_s"] < 0.9, execution
    assert execution["child_exit_code"] != 0
    assert not execution["hooks"]["diagnostic"]["complete"]
    if mode == "late-exit":
        assert execution["child_result_available"]  # proof arrived, exit was late
    else:
        assert execution["kill_sent_s"] is not None  # blocked native-owner cleanup
        assert execution["hooks"]["diagnostic"]["payload"]["partial"]


def test_transport_failure_does_not_silently_fall_through_to_enumeration(monkeypatch):
    monkeypatch.setattr(harness, "_host_compile_ttir", lambda spec: "control IR")
    monkeypatch.setattr(
        harness,
        "_static_track",
        lambda *args: {"status": "unsupported", "reason": "test"},
    )

    def fail(*args):
        raise child.DynamicSubprocessError("input transport mismatch")

    monkeypatch.setattr(harness, "_dynamic_track", fail)
    monkeypatch.setattr(
        harness,
        "_enum_track",
        lambda *a, **kw: pytest.fail("transport error silently fell through"),
    )
    result = harness.run_one(_spec(), 0, ladder_level=LadderLevel.L2)
    assert result["verdict"] == "error"
    assert result["terminal"] == "harness-error"
    assert "input transport mismatch" in result["harness_error"]


@pytest.mark.parametrize("failure", ["serialize", "save"])
def test_plain_transport_failure_is_a_harness_error(monkeypatch, failure):
    monkeypatch.setattr(harness, "_host_compile_ttir", lambda spec: "control IR")
    monkeypatch.setattr(
        harness,
        "_static_track",
        lambda *args: {"status": "ok", "provenance": "proved@T0"},
    )

    def fail(*args, **kwargs):
        raise TypeError("plain transport failure")

    if failure == "serialize":
        monkeypatch.setattr(child, "_serialize_spec", fail)
    else:
        pytest.importorskip("cloudpickle")
        monkeypatch.setattr(torch, "save", fail)
    result = harness.run_one(_spec(), 0, ladder_level=LadderLevel.L2)
    assert result["verdict"] == "error"
    assert result["terminal"] == "harness-error"
    assert "plain transport failure" in result["harness_error"]
