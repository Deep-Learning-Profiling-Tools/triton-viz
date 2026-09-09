"""Process containment preserves the launch and never credits a late exit."""
from dataclasses import replace
import hashlib
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


def _legacy_tensor_identity(value, alias_group):
    """The pre-P17 byte materialization is the compatibility oracle."""
    reinterpret_dtype = None
    if hasattr(value, "base") and isinstance(value.base, torch.Tensor):
        reinterpret_dtype = str(value.dtype)
        value = value.base
    storage = value.untyped_storage()
    backing = (
        torch.empty(0, dtype=torch.uint8)
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .numpy()
        .tobytes()
    )
    logical = (
        value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "reinterpret_dtype": reinterpret_dtype,
        "storage_offset": value.storage_offset(),
        "alias_group": alias_group,
        "storage_nbytes": storage.nbytes(),
        "sha256": hashlib.sha256(logical).hexdigest(),
        "storage_sha256": hashlib.sha256(backing).hexdigest(),
    }


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(lambda: torch.arange(12).reshape(3, 4), id="contiguous"),
        pytest.param(lambda: torch.arange(12).reshape(3, 4).T, id="transpose"),
        pytest.param(lambda: torch.arange(4).expand(3, 4), id="expanded"),
        pytest.param(lambda: torch.arange(12)[2:10:2], id="strided-offset"),
        pytest.param(lambda: torch.arange(12)[2:10], id="contiguous-offset"),
        pytest.param(lambda: torch.arange(12)[:8], id="contiguous-prefix"),
        pytest.param(lambda: torch.tensor(1.5), id="scalar"),
        pytest.param(lambda: torch.empty(0, 3), id="empty-storage"),
        pytest.param(lambda: torch.arange(8)[3:3], id="empty-view"),
        pytest.param(lambda: torch.tensor([False, True]), id="bool"),
        pytest.param(lambda: torch.arange(4, dtype=torch.bfloat16), id="bfloat16"),
        pytest.param(lambda: torch.tensor([1 + 2j, -3 + 4j]), id="complex"),
        pytest.param(
            lambda: torch.arange(24)
            .reshape(1, 2, 3, 4)
            .contiguous(memory_format=torch.channels_last),
            id="channels-last",
        ),
        pytest.param(lambda: torch.ones(4, requires_grad=True), id="requires-grad"),
    ],
)
def test_input_identity_preserves_legacy_tensor_bytes_and_layout(make_value):
    value = make_value()
    expected = _legacy_tensor_identity(value, 0)
    assert child.input_identity(_spec(), (value, value)) == {
        "x": expected,
        "out": expected,
        "N": 4,
    }


@pytest.mark.parametrize("views", ["offset", "stride", "dtype", "reinterpret"])
def test_input_identity_distinguishes_views_of_one_storage(views):
    base = torch.arange(16, dtype=torch.int32)
    inputs = {
        "offset": (base[1:9], base[2:10]),
        "stride": (base[:8], base[::2]),
        "dtype": (base, base.view(torch.int16)),
        "reinterpret": (base[2:12:2], triton.reinterpret(base[2:12:2], tl.uint32)),
    }[views]
    assert child.input_identity(_spec(), inputs) == {
        "x": _legacy_tensor_identity(inputs[0], 0),
        "out": _legacy_tensor_identity(inputs[1], 0),
        "N": 4,
    }


@pytest.mark.parametrize(
    "view,expected_hashes", [("whole", 1), ("strided", 2), ("offset", 2)]
)
def test_input_identity_hashes_repeated_physical_views_once_without_bytes_copy(
    monkeypatch, view, expected_hashes
):
    base = torch.arange(16, dtype=torch.int32)
    value = {"whole": base, "strided": base[2:12:2], "offset": base[2:12]}[view]
    expected = _legacy_tensor_identity(value, 0)
    original = hashlib.sha256
    buffers = []

    def observe(buffer):
        buffers.append(type(buffer))
        return original(buffer)

    monkeypatch.setattr(child.hashlib, "sha256", observe)
    assert child.input_identity(_spec(), (value, value)) == {
        "x": expected,
        "out": expected,
        "N": 4,
    }
    assert buffers == [memoryview] * expected_hashes


def test_input_identity_does_not_reuse_mutated_input_across_calls():
    base = torch.arange(16, dtype=torch.int32)
    value = base[::2]
    before = child.input_identity(_spec(), (value, value))
    value[0] = 99
    after = child.input_identity(_spec(), (value, value))
    assert after["x"] == _legacy_tensor_identity(value, 0)
    assert after["x"]["sha256"] != before["x"]["sha256"]
    assert after["x"]["storage_sha256"] != before["x"]["storage_sha256"]


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(lambda: torch.tensor([1 + 2j]).conj(), id="conjugate"),
        pytest.param(lambda: torch._neg_view(torch.tensor([1.0])), id="negative"),
    ],
)
def test_input_identity_keeps_lazy_view_refusal(make_value):
    value = make_value()
    with pytest.raises(RuntimeError) as legacy:
        _legacy_tensor_identity(value, 0)
    with pytest.raises(type(legacy.value)):
        child.input_identity(_spec(), (value, value))


def test_input_identity_keeps_tensor_subclass_materialization(monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass

    value = torch.arange(8).as_subclass(TensorSubclass)
    expected = _legacy_tensor_identity(value, 0)
    original = hashlib.sha256
    buffers = []

    def observe(buffer):
        buffers.append(type(buffer))
        return original(buffer)

    monkeypatch.setattr(child.hashlib, "sha256", observe)
    assert child.input_identity(_spec(), (value, value)) == {
        "x": expected,
        "out": expected,
        "N": 4,
    }
    assert buffers == [memoryview, bytes, bytes]


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


@pytest.mark.parametrize(
    "policy,static_status", [("all", "ok"), ("on-demand", "unsupported")]
)
def test_transport_failure_does_not_silently_fall_through_to_enumeration(
    monkeypatch, policy, static_status
):
    monkeypatch.setenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", "1" if policy == "all" else "0")
    monkeypatch.setattr(harness, "_host_compile_ttir", lambda spec: "control IR")
    monkeypatch.setattr(
        harness,
        "_static_track",
        lambda *args: {
            "status": static_status,
            "provenance": "proved@T0" if static_status == "ok" else None,
            "reason": "test",
        },
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
    assert result["frontend_policy"] == policy
    assert "enum" not in result
    assert "input transport mismatch" in result["harness_error"]


@pytest.mark.parametrize("failure", ["serialize", "save"])
def test_plain_transport_failure_is_a_harness_error(monkeypatch, failure):
    # This probe intentionally requests dynamic even after a static proof.
    # Default L2 on-demand correctly skips transport for that proof.
    monkeypatch.setenv("TRITON_VIZ_EVAL_ALL_FRONTENDS", "1")
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
