from __future__ import annotations

from typing import Any

from .core.config import config
from .version import __version__, git_version

__all__ = [
    "trace",
    "clear",
    "save",
    "load",
    "config",
    "launch",
    "__version__",
    "git_version",
]


def trace(*args: Any, **kwargs: Any) -> Any:
    from .core.trace import trace as _trace

    return _trace(*args, **kwargs)


def clear(*args: Any, **kwargs: Any) -> Any:
    from .core.trace import clear as _clear

    return _clear(*args, **kwargs)


def save(*args: Any, **kwargs: Any) -> Any:
    from .core.trace_io import save as _save

    return _save(*args, **kwargs)


def load(*args: Any, **kwargs: Any) -> Any:
    from .core.trace_io import load as _load

    return _load(*args, **kwargs)


def launch(*args: Any, **kwargs: Any) -> Any:
    from .visualizer import launch as _launch

    return _launch(*args, **kwargs)
