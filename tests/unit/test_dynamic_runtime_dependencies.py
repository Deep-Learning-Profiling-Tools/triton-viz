"""Transport must preserve runtime dependencies or refuse before analysis."""
import functools
import importlib
import types
import sys

import pytest
import torch
import triton
import triton.language as tl
import triton_viz

from evaluation import dynamic_subprocess as child
from evaluation import harness
from evaluation.spec import LaunchSpec
from triton_viz.clients.race_detector.ladder import LadderLevel


@triton.jit
def _root_next_power_of_2(out, N: tl.constexpr):
    block: tl.constexpr = triton.next_power_of_2(N)
    offsets = tl.arange(0, block)
    tl.store(out + offsets, 1.0, offsets < N)


@triton.jit
def _root_cdiv(out, N: tl.constexpr):
    tl.store(out, triton.cdiv(N, 2))


@pytest.mark.parametrize(
    "kernel,helper_name",
    [(_root_next_power_of_2, "next_power_of_2"), (_root_cdiv, "cdiv")],
)
def test_root_framework_helpers_survive_serialization(kernel, helper_name):
    cloudpickle = pytest.importorskip("cloudpickle")
    spec = LaunchSpec(
        name="root_helper",
        kernel_fn=kernel,
        signature={"out": "*fp32", "N": "constexpr"},
        constexprs={"N": 3},
        grid=(1,),
        make_args=lambda seed: (torch.zeros(3),),
    )
    before = child._kernel_identity(kernel)
    primitive = before["dependency_sources"]["kernel.triton"]["attributes"][
        helper_name
    ]["constexpr_function"]
    assert primitive["primitive"] == "triton." + helper_name
    assert primitive["code"]
    restored = cloudpickle.loads(child._serialize_spec(spec))
    assert child._kernel_identity(restored.kernel_fn) == before


def test_wrapped_root_helper_replacement_is_not_trusted(monkeypatch):
    from triton.runtime.jit import ConstexprFunction

    @functools.wraps(triton.next_power_of_2.fn)
    def replacement(n):
        return 1

    monkeypatch.setattr(triton, "next_power_of_2", ConstexprFunction(replacement))
    with pytest.raises(child.DynamicSubprocessError, match="primitive replacement"):
        child._kernel_identity(_root_next_power_of_2)


@pytest.mark.parametrize(
    "kernel,helper_name",
    [(_root_next_power_of_2, "next_power_of_2"), (_root_cdiv, "cdiv")],
)
def test_root_helper_defaults_remain_part_of_identity(monkeypatch, kernel, helper_name):
    before = child._kernel_identity(kernel)
    monkeypatch.setattr(getattr(triton, helper_name).fn, "__defaults__", (1,))
    assert child._kernel_identity(kernel) != before


def test_similar_root_module_name_is_not_trusted(monkeypatch):
    monkeypatch.setattr(triton.next_power_of_2, "__module__", "triton_extra")
    with pytest.raises(
        child.DynamicSubprocessError, match="unsupported runtime dependency"
    ):
        child._kernel_identity(_root_next_power_of_2)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    name = "dynamic_runtime_settings_control"
    (tmp_path / (name + ".py")).write_text("STRIDE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    monkeypatch.delitem(sys.modules, name, raising=False)


runtime_settings = None


@triton.jit
def _global_stride_kernel(out):
    pid = tl.program_id(0)
    tl.store(out + pid * runtime_settings.STRIDE, 1.0)


def _spec(settings):
    @triton.jit
    def kernel(out):
        pid = tl.program_id(0)
        tl.store(out + pid * settings.STRIDE, 1.0)

    return LaunchSpec(
        name="module_stride",
        kernel_fn=kernel,
        signature={"out": "*fp32"},
        constexprs={},
        grid=(2,),
        make_args=lambda seed: (torch.zeros(2, dtype=torch.float32),),
    )


def test_actual_module_attribute_is_part_of_identity(settings):
    spec = _spec(settings)
    before = child._kernel_identity(spec.kernel_fn)
    settings.STRIDE = 0
    after = child._kernel_identity(spec.kernel_fn)
    assert before != after
    assert after["dependency_sources"]["kernel.settings"]["attributes"]["STRIDE"] == {
        "constant": 0
    }


def test_changed_module_attribute_cannot_become_a_clean_child_result(
    settings, monkeypatch
):
    from dataclasses import replace

    monkeypatch.setitem(globals(), "runtime_settings", settings)
    spec = replace(_spec(settings), kernel_fn=_global_stride_kernel)
    unchanged = harness._dynamic_track(spec, 0, LadderLevel.L2)
    assert unchanged["status"] == "ok", unchanged
    assert unchanged["n_reports"] == 0
    assert unchanged["execution"]["child_completed"]
    settings.STRIDE = 0
    try:
        local = harness._dynamic_track_local(spec, 0, LadderLevel.L2)
    finally:
        triton_viz.clear()
    assert local["status"] == "ok", local
    assert local["n_reports"] > 0
    with pytest.raises(child.DynamicSubprocessError, match="kernel identity mismatch"):
        harness._dynamic_track(spec, 0, LadderLevel.L2)


def test_nested_module_and_module_jit_helper_dependencies(settings):
    nested = types.ModuleType("nested_control")
    nested.STRIDE = 1
    settings.layout = nested

    @triton.jit
    def helper():
        return settings.layout.STRIDE

    settings.helper = helper

    @triton.jit
    def kernel(out):
        tl.store(out, settings.helper())

    before = child._kernel_identity(kernel)
    nested.STRIDE = 0
    assert child._kernel_identity(kernel) != before


@pytest.mark.parametrize("indirection", ["alias", "getattr", "helper"])
def test_module_escape_is_explicitly_refused(settings, indirection):
    @triton.jit
    def alias(out):
        local = settings
        tl.store(out, local.STRIDE)

    @triton.jit
    def dynamic_attr(out):
        tl.store(out, getattr(settings, "STRIDE"))

    @triton.jit
    def take_module(module):
        return module.STRIDE

    @triton.jit
    def pass_module(out):
        tl.store(out, take_module(settings))

    kernel = {"alias": alias, "getattr": dynamic_attr, "helper": pass_module}[
        indirection
    ]
    with pytest.raises(
        child.DynamicSubprocessError, match="indirect module dependency"
    ):
        child._kernel_identity(kernel)


def test_dynamic_module_attribute_is_not_evaluated(settings):
    del settings.STRIDE
    settings.__getattr__ = lambda name: pytest.fail("executed module __getattr__")
    with pytest.raises(child.DynamicSubprocessError, match="dynamic module attribute"):
        child._kernel_identity(_spec(settings).kernel_fn)


@pytest.mark.parametrize(
    "value", [object(), lambda: 1, type("Int", (int,), {})(1), [1], {"stride": 1}]
)
def test_opaque_runtime_attributes_are_refused(settings, value):
    settings.STRIDE = value
    with pytest.raises(
        child.DynamicSubprocessError, match="unsupported runtime dependency"
    ):
        child._kernel_identity(_spec(settings).kernel_fn)


def test_callable_defaults_are_checked_and_module_defaults_refused(settings):
    @triton.jit
    def kernel(out, stride: tl.constexpr = 1):
        tl.store(out + tl.program_id(0) * stride, 1.0)

    before = child._kernel_identity(kernel)
    kernel.fn.__defaults__ = (0,)
    assert child._kernel_identity(kernel) != before
    kernel.fn.__defaults__ = (settings,)
    with pytest.raises(
        child.DynamicSubprocessError, match="unsupported runtime dependency"
    ):
        child._kernel_identity(kernel)


def test_wrapped_primitive_replacement_is_not_trusted(settings, monkeypatch):
    @functools.wraps(tl.program_id)
    def replacement(*args, **kwargs):
        return 0

    spec = _spec(settings)
    child._kernel_identity(spec.kernel_fn)
    monkeypatch.setattr(tl, "program_id", replacement)
    with pytest.raises(child.DynamicSubprocessError, match="primitive replacement"):
        child._kernel_identity(spec.kernel_fn)


def test_module_helper_replacement_and_closure_are_checked(settings):
    offset = 1

    @triton.jit
    def helper():
        return offset

    @triton.jit
    def replacement():
        return 0

    @triton.jit
    def kernel(out):
        tl.store(out, settings.helper())

    settings.helper = helper
    before = child._kernel_identity(kernel)
    offset = 0
    assert child._kernel_identity(kernel) != before
    settings.helper = replacement
    assert child._kernel_identity(kernel) != before


def test_identity_sensitive_dependency_is_refused(settings):
    settings.A = (1, 2)
    settings.B = settings.A

    @triton.jit
    def kernel(out):
        tl.store(out, settings.A is settings.B)

    with pytest.raises(child.DynamicSubprocessError, match="identity comparison"):
        child._kernel_identity(kernel)


def test_runtime_introspection_alias_is_refused(settings):
    settings.identity = id

    @triton.jit
    def kernel(out):
        tl.store(out, settings.identity(out))

    with pytest.raises(child.DynamicSubprocessError, match="runtime introspection"):
        child._kernel_identity(kernel)


def test_builtin_jit_helpers_and_defaults_survive_serialization():
    import subprocess

    # Test the normal runtime without pytest's assertion rewriting of helper
    # functions. Rewritten application callables are correctly not trusted.
    # These 16 real kernels cover dtype/constexpr/native-enum defaults and
    # wrapper closure metadata, including tl.exp's captured dtype-name list.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import cloudpickle
from evaluation import dynamic_subprocess as child
from evaluation.kernels import golden_smoke, tutorials
for corpus in (golden_smoke.CORPUS, tutorials.CORPUS):
    for spec in corpus.specs:
        before = child._kernel_identity(spec.kernel_fn)
        restored = cloudpickle.loads(child._serialize_spec(spec))
        assert child._kernel_identity(restored.kernel_fn) == before, spec.name
""",
        ],
        check=True,
    )
