from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class OpCallbacks:
    before_callback: Callable | None = None
    after_callback: Callable | None = None
    op_overrider: Callable | None = None


@dataclass
class ForLoopCallbacks:
    range_wrapper_factory: (
        Callable | None
    ) = None  # optional: (iterable|None, lineno, range_type, iter_args, iter_kwargs, iter_callable) -> iterable|None
    range_type_callback: Callable | None = None
    before_loop_callback: Callable | None = None
    loop_iter_overrider: Callable | None = None
    loop_iter_listener: Callable | None = None
    after_loop_callback: Callable | None = None
