from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpCallbacks:
    before_callback: Optional[Callable] = None
    after_callback: Optional[Callable] = None
    op_overrider: Optional[Callable] = None


@dataclass
class ForLoopCallbacks:
    range_wrapper_factory: Optional[Callable] = None  # optional: (iterable|None, lineno, range_type, iter_args, iter_kwargs, iter_callable) -> iterable|None
    range_type_callback: Optional[Callable] = None
    before_loop_callback: Optional[Callable] = None
    loop_iter_overrider: Optional[Callable] = None
    loop_iter_listener: Optional[Callable] = None
    after_loop_callback: Optional[Callable] = None
