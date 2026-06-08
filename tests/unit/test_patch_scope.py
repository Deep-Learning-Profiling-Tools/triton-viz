import pytest
import warnings
from types import SimpleNamespace

import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from triton_viz.core import patch as patch_mod
from triton_viz.core.config import config as cfg
from triton_viz.core.frontend import triton as triton_frontend


def _dummy_kernel():
    """Provides tl/tl.core globals for language patch scope capture."""
    return tl.arange(0, 1)


_UNUSED_TL_CORE_ALIAS = tl.core
_TL_CORE_ALIAS = tl.core


def _uses_no_triton_language_module():
    return 1


def _uses_tl_core_alias():
    return _TL_CORE_ALIAS


def test_jit_function_lang_patch_check_uses_referenced_globals_only():
    needs_patch = triton_frontend.frontend._jit_function_needs_lang_patch

    assert not needs_patch(SimpleNamespace(fn=_uses_no_triton_language_module))
    assert needs_patch(SimpleNamespace(fn=_uses_tl_core_alias))


def test_scope_restores_tensor_magic_methods():
    """Scope must restore tensor magic methods patched by interpreter mode."""
    tensor = tl.tensor
    attrs = [
        attr
        for attr in ("__index__", "__bool__", "__repr__", "__str__", "T")
        if hasattr(tensor, attr)
    ]
    scope = triton_frontend.frontend._triton_snapshot_scope(_dummy_kernel)
    originals = {attr: getattr(tensor, attr) for attr in attrs}

    try:
        for attr in attrs:
            replacement = (
                property(lambda self: self)
                if attr == "T"
                else (lambda self, _attr=attr: _attr)
            )
            setattr(tensor, attr, replacement)
            assert getattr(tensor, attr) is replacement
    finally:
        scope.restore()

    for attr, original in originals.items():
        assert getattr(tensor, attr) is original


def test_scope_restores_tensor_descriptor_base_builtins(monkeypatch):
    """Scope must restore descriptor builtins on tensor_descriptor_base."""
    if not hasattr(tl.core, "tensor_descriptor_base"):
        pytest.skip("tensor_descriptor_base is not available")

    descriptor = tl.core.tensor_descriptor_base
    attr = "load"
    if not hasattr(descriptor, attr):
        pytest.skip("descriptor load is not available")

    original = getattr(descriptor, attr)
    sentinel = object()

    def _getmembers(obj):
        """Returns a controlled descriptor member list for deterministic capture."""
        if obj is descriptor:
            return [(attr, sentinel)]
        return []

    monkeypatch.setattr(triton_frontend.inspect, "getmembers", _getmembers)
    monkeypatch.setattr(
        triton_frontend.tl.core, "is_builtin", lambda member: member is sentinel
    )

    scope = triton_frontend.frontend._triton_snapshot_scope(_dummy_kernel)
    replacement = lambda *_args, **_kwargs: None

    try:
        setattr(descriptor, attr, replacement)
        assert getattr(descriptor, attr) is replacement
    finally:
        scope.restore()

    assert getattr(descriptor, attr) is original


def test_patch_lang_restores_core_reduce_scan_from_tl_kernel():
    """Interpreter scan patching mutates tl.core even for tl-only kernels."""
    original_reduce = tl.reduce
    original_scan = tl.associative_scan
    original_core_reduce = tl.core.reduce
    original_core_scan = tl.core.associative_scan

    patch_mod.patch_lang(_dummy_kernel, "triton")
    try:
        assert tl.associative_scan is not original_scan
        assert tl.core.associative_scan is not original_core_scan
    finally:
        patch_mod.unpatch_lang("triton")

    assert tl.reduce is original_reduce
    assert tl.associative_scan is original_scan
    assert tl.core.reduce is original_core_reduce
    assert tl.core.associative_scan is original_core_scan


def test_inline_asm_patch_returns_inputs_and_warns_once():
    class CastableArg:
        def __init__(self, name):
            self.name = name
            self.cast_dtypes = []

        def to(self, dtype):
            self.cast_dtypes.append(dtype)
            return (self.name, dtype)

    a = CastableArg("a")
    b = CastableArg("b")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        patch_mod.patch_lang(_dummy_kernel, "triton")
        try:
            assert tl.inline_asm_elementwise("", "", [a, b], tl.int32, True, 1) == (
                "a",
                tl.int32,
            )
            tuple_result = tl.inline_asm_elementwise(
                "", "", [a, b], (tl.int32, tl.float32, tl.int1), True, 1
            )
            assert tuple_result == (
                ("a", tl.int32),
                ("a", tl.float32),
                ("a", tl.int1),
            )
        finally:
            patch_mod.unpatch_lang("triton")

    assert a.cast_dtypes == [tl.int32, tl.int32, tl.float32, tl.int1]
    assert b.cast_dtypes == []
    assert len(caught) == 1
    assert "inline assembly is approximated" in str(caught[0].message)


def test_patch_lang_accepts_constexpr_wrapped_symbolic_tensors():
    handle = TensorHandle(np.array([1, 2], dtype=np.int32), tl.int32)
    tensor = tl.core.tensor(handle, tl.block_type(tl.int32, [2]))

    patch_mod.patch_lang(_dummy_kernel, "triton")
    try:
        assert (
            triton_frontend.interpreter_semantic.to_tensor(tl.constexpr(tensor))
            is tensor
        )
    finally:
        patch_mod.unpatch_lang("triton")


def test_patch_lang_does_not_inject_loop_global():
    legacy_key = "_triton_viz_" + "loop_patcher"
    wrapper_key = "_triton_viz_loop_iter_wrapper"
    globals_dict = _dummy_kernel.__globals__

    class ClientManagerStub:
        pass

    try:
        patch_mod.patch_lang(_dummy_kernel, "triton", ClientManagerStub())
        assert legacy_key not in globals_dict
        assert wrapper_key not in globals_dict
    finally:
        patch_mod.unpatch_lang("triton")

    assert legacy_key not in globals_dict
    assert wrapper_key not in globals_dict


def test_grid_executor_temporarily_disables_overflow_checks(monkeypatch):
    frontend = triton_frontend.frontend
    old_virtual_memory = cfg.virtual_memory
    cfg.virtual_memory = True

    class ClientManagerStub:
        def get_client(self, name):
            return None

    class GridExecutorStub:
        fn = staticmethod(_dummy_kernel)

    def prepare_grid_launch(_grid_executor, _args_dev, kwargs):
        return (
            ClientManagerStub(),
            kwargs,
            (),
            {},
            {},
            (1, 1, 1),
            1,
        )

    observed = []

    def run_grid(_grid_executor, _call_args, _client_manager, _grid, _max_workers):
        observed.append(frontend.builder.options.sanitize_overflow)

    monkeypatch.setattr(frontend, "_prepare_grid_launch", prepare_grid_launch)
    monkeypatch.setattr(frontend, "_run_grid", run_grid)

    had_attr = hasattr(frontend.builder.options, "sanitize_overflow")
    old_value = getattr(frontend.builder.options, "sanitize_overflow", None)
    object.__setattr__(frontend.builder.options, "sanitize_overflow", True)
    try:
        frontend._grid_executor_call(GridExecutorStub())
    finally:
        cfg.virtual_memory = old_virtual_memory
        if had_attr:
            object.__setattr__(
                frontend.builder.options,
                "sanitize_overflow",
                old_value,
            )
        else:
            object.__delattr__(frontend.builder.options, "sanitize_overflow")

    assert observed == [False]
    assert frontend.builder.options.sanitize_overflow is True
