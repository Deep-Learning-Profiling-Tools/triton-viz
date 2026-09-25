"""Compatibility imports for the former triton_viz namespace."""

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader

# Returned while another thread is still importing the canonical parent.
_DEFERRED = object()


def _is_legacy_finder(finder):
    return getattr(finder, "_tilelens_legacy_aliases", False)


def _find_spec_without_import(name, path):
    """Ask the other meta path finders for ``name``, in import order."""
    for finder in list(sys.meta_path):
        if _is_legacy_finder(finder):
            continue
        finder_find_spec = getattr(finder, "find_spec", None)
        if finder_find_spec is None:
            continue
        spec = finder_find_spec(name, path, None)
        if spec is not None:
            return spec
    return None


def _find_canonical_spec(name):
    """Locate the canonical ``tilelens`` module without importing anything.

    Python calls ``MetaPathFinder.find_spec`` while holding the global import
    lock. Importing here (for example through ``importlib.util.find_spec``,
    which imports parent packages) waits on a per-module lock that another
    thread can hold while it waits for the global lock, deadlocking both.
    Like importlib itself, only ask the finders once the parent package is in
    ``sys.modules``; namespace packages read their path from it. If another
    thread is still importing the parent, resolution is left to
    ``exec_module``.
    """
    spec = getattr(sys.modules.get(name), "__spec__", None)
    if spec is not None:
        return spec
    parent = sys.modules.get(name.rpartition(".")[0])
    if parent is None:
        return _DEFERRED
    parent_path = getattr(parent, "__path__", None)
    if parent_path is None:
        return None
    return _find_spec_without_import(name, parent_path)


class _LegacyLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Import the canonical ancestors top-down so this thread never holds a
        # child module lock while waiting for a parent that another thread is
        # still importing.
        parts = self.target.split(".")
        for depth in range(1, len(parts)):
            import_module(".".join(parts[:depth]))
        canonical = import_module(self.target)
        # A thread that raced this import may already hold this placeholder:
        # ``from triton_viz.x import name`` keeps the module it found in
        # sys.modules before waiting for the import lock. Forward its
        # attribute lookups to the canonical module.
        module.__getattr__ = lambda name: getattr(canonical, name)
        # Reuse the canonical module, preserving class identity and trace state.
        sys.modules[module.__name__] = canonical


class _LegacyFinder(MetaPathFinder):
    _tilelens_legacy_aliases = True

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("triton_viz."):
            return None
        canonical_name = "tilelens." + fullname.removeprefix("triton_viz.")
        canonical_spec = _find_canonical_spec(canonical_name)
        if canonical_spec is None:
            return None
        if canonical_spec is _DEFERRED:
            # exec_module replaces this placeholder with the canonical module.
            is_package = True
        else:
            is_package = canonical_spec.submodule_search_locations is not None
        return spec_from_loader(
            fullname,
            _LegacyLoader(canonical_name),
            is_package=is_package,
        )


# Precede filesystem lookup so stale files from old installs cannot be loaded.
# Python publishes this module in sys.modules before running it, so a
# concurrent triton_viz submodule import can still miss the finder; installing
# it before the slow tilelens import narrows that window but cannot close it.
if not any(_is_legacy_finder(finder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyFinder())

from tilelens import (  # noqa: E402
    __all__ as __all__,
    __version__ as __version__,
    clear as clear,
    config as config,
    git_version as git_version,
    launch as launch,
    load as load,
    save as save,
    trace as trace,
)
