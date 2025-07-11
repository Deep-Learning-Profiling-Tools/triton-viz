from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class OpCallbacks:
    before_callback: Callable | None = None
    after_callback: Callable | None = None
    op_overrider: Callable | None = None