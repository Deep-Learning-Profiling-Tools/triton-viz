"""Compatibility imports for the former triton_viz namespace."""

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader

from tilelens import (
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


class _LegacyLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Reuse the canonical module, preserving class identity and trace state.
        sys.modules[module.__name__] = import_module(self.target)


class _LegacyFinder(MetaPathFinder):
    _tilelens_legacy_aliases = True

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("triton_viz."):
            return None
        canonical_name = "tilelens." + fullname.removeprefix("triton_viz.")
        canonical_spec = find_spec(canonical_name)
        if canonical_spec is None:
            return None
        return spec_from_loader(
            fullname,
            _LegacyLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


# Precede filesystem lookup so stale files from old installs cannot be loaded.
if not any(
    getattr(finder, "_tilelens_legacy_aliases", False) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _LegacyFinder())
