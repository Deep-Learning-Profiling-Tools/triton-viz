import pytest

import triton.language as tl
import triton.language.extra.libdevice as _ld_ref  # noqa: F401

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


def test_patch_libdevice_multiple_aliases():
    """Multiple globals aliasing different libdevice fns are all patched and restored."""
    import triton.language.extra.libdevice as _ld
    from triton_viz.core.patch import _patch_libdevice

    original_tanh = _ld.tanh
    original_asin = _ld.asin

    def _kernel():
        pass

    _kernel.__globals__["my_tanh"] = original_tanh
    _kernel.__globals__["my_asin"] = original_asin

    scope = _LangPatchScope()
    _patch_libdevice(_kernel, scope)
    assert _kernel.__globals__["my_tanh"] is not original_tanh
    assert _kernel.__globals__["my_asin"] is not original_asin

    scope.restore()
    assert _kernel.__globals__["my_tanh"] is original_tanh
    assert _kernel.__globals__["my_asin"] is original_asin
    del _kernel.__globals__["my_tanh"]
    del _kernel.__globals__["my_asin"]


def test_patch_libdevice_unregistered_alias_unchanged():
    """Unregistered libdevice aliases are left as-is (not blanket-replaced)."""
    import triton.language.extra.libdevice as _ld
    from triton_viz.core.patch import _patch_libdevice
    from triton_viz.core.libdevice_registry import _REGISTERED_SPECS

    # Find an unregistered public function on the libdevice module
    import inspect

    unregistered_name = None
    for name in dir(_ld):
        if name.startswith("_") or name in _REGISTERED_SPECS:
            continue
        member = getattr(_ld, name, None)
        if (
            inspect.isfunction(member)
            and getattr(member, "__module__", None) == _ld.__name__
        ):
            unregistered_name = name
            break
    if unregistered_name is None:
        pytest.skip("No unregistered libdevice functions found")

    original = getattr(_ld, unregistered_name)

    def _kernel():
        pass

    _kernel.__globals__["_unreg_alias"] = original

    scope = _LangPatchScope()
    _patch_libdevice(_kernel, scope)
    # Unregistered alias must NOT be modified
    assert _kernel.__globals__["_unreg_alias"] is original

    scope.restore()
    del _kernel.__globals__["_unreg_alias"]


def test_patch_libdevice_helper_same_module_alias():
    """Direct-import alias in a same-module helper is patched via shared globals."""
    import triton.language.extra.libdevice as _ld
    from triton_viz.core.patch import _patch_libdevice

    original_tanh = _ld.tanh

    # Simulate: from triton.language.extra.libdevice import tanh as _tanh
    # Both kernel and helper share __globals__ when in the same module.
    def _kernel():
        pass

    _kernel.__globals__["_tanh"] = original_tanh

    scope = _LangPatchScope()
    _patch_libdevice(_kernel, scope)
    # Alias must be patched (same-module helpers share the kernel's globals)
    assert _kernel.__globals__["_tanh"] is not original_tanh

    scope.restore()
    assert _kernel.__globals__["_tanh"] is original_tanh
    del _kernel.__globals__["_tanh"]


def test_patch_libdevice_default_arg_not_patched():
    """Default args capturing a libdevice fn are NOT patched (known limitation)."""
    import triton.language.extra.libdevice as _ld
    from triton_viz.core.patch import _patch_libdevice

    original_tanh = _ld.tanh

    def helper(x, op=original_tanh):
        return op(x)

    def _kernel():
        pass

    _kernel.__globals__["helper"] = helper

    scope = _LangPatchScope()
    _patch_libdevice(_kernel, scope)
    # The default arg is captured at definition time and NOT rewritten.
    assert helper.__defaults__[0] is original_tanh

    scope.restore()
    del _kernel.__globals__["helper"]


def test_patch_lang_rollback_on_libdevice_failure(monkeypatch):
    """If _patch_libdevice throws, all triton lang state must be restored."""
    from triton_viz.core import patch as patch_mod

    original_arange = tl.arange

    def _exploding_patch(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(patch_mod, "_patch_libdevice", _exploding_patch)

    with pytest.raises(RuntimeError, match="boom"):
        patch_mod.patch_lang(_dummy_kernel, "triton")

    # tl.arange must be restored to its original (not leaked as interpreter version)
    assert tl.arange is original_arange
    # scope must not have been pushed
    assert len(patch_mod._LANG_PATCH_SCOPES["triton"]) == 0
