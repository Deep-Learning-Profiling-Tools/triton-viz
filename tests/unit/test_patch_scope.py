import pytest

import triton.language as tl

from triton_viz.core import patch as patch_mod
from triton_viz.core.patch import _triton_snapshot_scope, patch_lang, unpatch_lang


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


def test_constexpr_patch_lifecycle():
    """patch_lang must add .to/__getattr__ to constexpr; unpatch_lang must restore."""
    # Snapshot before-state
    had_to = hasattr(tl.constexpr, "to")
    had_getattr = hasattr(tl.constexpr, "__getattr__")
    orig_to = getattr(tl.constexpr, "to", None)
    orig_getattr = getattr(tl.constexpr, "__getattr__", None)

    patch_lang(_dummy_kernel, "triton")
    try:
        assert hasattr(tl.constexpr, "to")
        assert callable(tl.constexpr.to)
        assert hasattr(tl.constexpr, "__getattr__")
        assert callable(tl.constexpr.__getattr__)
    finally:
        unpatch_lang("triton")

    # Verify exact before-state is restored
    assert hasattr(tl.constexpr, "to") == had_to
    assert hasattr(tl.constexpr, "__getattr__") == had_getattr
    if had_to:
        assert getattr(tl.constexpr, "to") is orig_to
    if had_getattr:
        assert getattr(tl.constexpr, "__getattr__") is orig_getattr
