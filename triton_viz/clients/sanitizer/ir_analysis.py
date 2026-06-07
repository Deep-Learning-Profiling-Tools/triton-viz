from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoopRangeSummary:
    lineno: int
    target: str
    range_type: str
    start: int | None
    stop: int | None
    step: int | None

    @property
    def is_constant_range(self) -> bool:
        return (
            self.start is not None and self.stop is not None and self.step is not None
        )

    @property
    def trip_count(self) -> int | None:
        if not self.is_constant_range:
            return None
        assert self.start is not None
        assert self.stop is not None
        assert self.step is not None
        return len(range(self.start, self.stop, self.step))


def summarize_loop_structure(
    fn_or_source: Callable[..., Any] | str,
) -> list[LoopRangeSummary]:
    source = _source_text(fn_or_source)
    tree = ast.parse(source)
    return [
        _summarize_for(node) for node in ast.walk(tree) if isinstance(node, ast.For)
    ]


def _source_text(fn_or_source: Callable[..., Any] | str) -> str:
    source_target = _source_target(fn_or_source)
    if isinstance(source_target, str):
        return textwrap.dedent(source_target)
    return textwrap.dedent(inspect.getsource(source_target))


def _source_target(
    fn_or_source: Callable[..., Any] | str,
    seen: set[int] | None = None,
) -> Callable[..., Any] | str:
    if isinstance(fn_or_source, str):
        return fn_or_source

    seen = set() if seen is None else seen
    obj_id = id(fn_or_source)
    if obj_id in seen:
        return fn_or_source
    seen.add(obj_id)

    if inspect.isfunction(fn_or_source) or inspect.ismethod(fn_or_source):
        return fn_or_source

    for attr in ("base_fn", "fn"):
        wrapped = getattr(fn_or_source, attr, None)
        if wrapped is not None and wrapped is not fn_or_source:
            return _source_target(wrapped, seen)

    raw_src = getattr(fn_or_source, "raw_src", None)
    if isinstance(raw_src, str):
        return raw_src

    return fn_or_source


def _summarize_for(node: ast.For) -> LoopRangeSummary:
    range_type, start, stop, step = _range_call_bounds(node.iter)
    return LoopRangeSummary(
        lineno=node.lineno,
        target=ast.unparse(node.target),
        range_type=range_type,
        start=start,
        stop=stop,
        step=step,
    )


def _range_call_bounds(
    node: ast.expr,
) -> tuple[str, int | None, int | None, int | None]:
    if not isinstance(node, ast.Call):
        return "unknown", None, None, None

    range_type = _range_type(node.func)
    if range_type == "unknown":
        return range_type, None, None, None

    start, stop, step = _parse_range_args(node)
    return range_type, start, stop, step


def _range_type(func: ast.expr) -> str:
    if isinstance(func, ast.Name) and func.id == "range":
        return "python_range"
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
    ):
        if func.attr == "range":
            return "tl_range"
        if func.attr == "static_range":
            return "tl_static_range"
    return "unknown"


def _parse_range_args(call: ast.Call) -> tuple[int | None, int | None, int | None]:
    positional = [_literal_int(arg) for arg in call.args]
    kwargs = {
        kw.arg: _literal_int(kw.value) for kw in call.keywords if kw.arg is not None
    }

    start: int | None = 0
    stop: int | None = None
    step: int | None = 1

    if len(positional) == 1:
        stop = positional[0]
    elif len(positional) == 2:
        start, stop = positional
    elif len(positional) >= 3:
        start, stop, step = positional[:3]

    if "start" in kwargs:
        start = kwargs["start"]
    if "stop" in kwargs:
        stop = kwargs["stop"]
    if "end" in kwargs:
        stop = kwargs["end"]
    if "step" in kwargs:
        step = kwargs["step"]

    return start, stop, step


def _literal_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    return None
