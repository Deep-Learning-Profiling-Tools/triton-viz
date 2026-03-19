import pytest

import triton.language as tl

from triton_viz.core import patch as patch_mod
from triton_viz.core.patch import _LangPatchScope, _triton_snapshot_scope


def _dummy_kernel():
    """Provides tl/tl.core globals for language patch scope capture."""
    return tl.arange(0, 1)


def test_scope_restores_tensor_magic_methods():
    """Scope must restore tensor magic methods patched by interpreter mode."""
    tensor = tl.tensor
    attrs = [
        attr
        for attr in ("__index__", "__bool__", "__repr__", "__str__", "T")
        if hasattr(tensor, attr)
    ]
    scope = _triton_snapshot_scope(_dummy_kernel)
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

    monkeypatch.setattr(patch_mod.inspect, "getmembers", _getmembers)
    monkeypatch.setattr(
        patch_mod.tl.core, "is_builtin", lambda member: member is sentinel
    )

    scope = _triton_snapshot_scope(_dummy_kernel)
    replacement = lambda *_args, **_kwargs: None

    try:
        setattr(descriptor, attr, replacement)
        assert getattr(descriptor, attr) is replacement
    finally:
        scope.restore()

    assert getattr(descriptor, attr) is original


# ======== set_item Tests ===========


def test_scope_set_item_restores_original():
    """set_item on existing key restores original value."""
    scope = _LangPatchScope()
    d = {"key": "original"}
    scope.set_item(d, "key", "patched")
    assert d["key"] == "patched"
    scope.restore()
    assert d["key"] == "original"


def test_scope_set_item_removes_new_key():
    """set_item on non-existent key removes it on restore."""
    scope = _LangPatchScope()
    d = {}
    scope.set_item(d, "new_key", "value")
    assert d["new_key"] == "value"
    scope.restore()
    assert "new_key" not in d


def test_scope_interleaved_restore_order():
    """Interleaved set_attr and set_item restore in LIFO order."""
    scope = _LangPatchScope()

    class Obj:
        x = "orig_x"

    d = {"k": "orig_k"}

    scope.set_attr(Obj, "x", "patched_x")  # first
    scope.set_item(d, "k", "patched_k")  # second
    scope.set_attr(Obj, "x", "patched_x2")  # third

    assert Obj.x == "patched_x2"
    assert d["k"] == "patched_k"

    scope.restore()

    assert Obj.x == "orig_x"
    assert d["k"] == "orig_k"


def test_scope_restore_not_monkey_patched():
    """After _patch_libdevice, scope.restore is the original method, not a closure."""
    from triton_viz.core.patch import _patch_libdevice

    scope = _LangPatchScope()
    _patch_libdevice(_dummy_kernel, scope)
    assert scope.restore.__func__ is _LangPatchScope.restore
    scope.restore()  # clean up


def test_libdevice_patch_and_restore():
    """_patch_libdevice patches module attrs + aliases; restore undoes both."""
    import triton.language.extra.libdevice as _ld
    from triton_viz.core.patch import _patch_libdevice

    original_tanh = _ld.tanh
    scope = _LangPatchScope()
    _patch_libdevice(_dummy_kernel, scope)

    # Module attr is now patched
    assert _ld.tanh is not original_tanh
    scope.restore()
    # Module attr is restored
    assert _ld.tanh is original_tanh


def test_patch_libdevice_alias_restore():
    """_patch_libdevice patches fn.__globals__ aliases; restore undoes them."""
    import triton.language.extra.libdevice as _ld
    from triton_viz.core.patch import _patch_libdevice

    original_tanh = _ld.tanh

    def _kernel_with_alias():
        pass

    _kernel_with_alias.__globals__["my_tanh"] = original_tanh

    scope = _LangPatchScope()
    _patch_libdevice(_kernel_with_alias, scope)
    assert _kernel_with_alias.__globals__["my_tanh"] is not original_tanh

    scope.restore()
    assert _kernel_with_alias.__globals__["my_tanh"] is original_tanh
    del _kernel_with_alias.__globals__["my_tanh"]
