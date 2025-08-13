from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpCallbacks:
    before_callback: Optional[Callable] = None
    after_callback: Optional[Callable] = None
    op_overrider: Optional[Callable] = None
