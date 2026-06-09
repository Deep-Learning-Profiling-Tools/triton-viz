from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.symbolic_metadata import SymbolicTensorValue, element_bytewidth
from ..symbolic_engine import LoopContext, SymbolicExpr


_UNARY_INTERVAL_OPS = frozenset({"abs", "fabs", "negative"})
_BINARY_INTERVAL_OPS = frozenset({"add", "sub", "mul", "maximum", "minimum"})
_TERNARY_INTERVAL_OPS = frozenset({"where", "fma"})

# These ops only change tensor shape/layout in the symbolic engine's Z3 path.
# Value-changing ops, especially casts and bitcasts, must not be added here.
_VALUE_PRESERVING_RESHAPE_OPS = frozenset(
    {"broadcast", "expand_dims", "reshape", "splat", "trans", "unsplat"}
)


@dataclass(frozen=True)
class IntRange:
    lower: int
    upper: int

    @staticmethod
    def constant(value: int) -> "IntRange":
        return IntRange(value, value)

    def contains(self, other: "IntRange") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper


def access_interval_summary(
    symbolic_expr: SymbolicExpr,
    grid: tuple[int, ...] | None,
) -> IntRange | None:
    """Return a conservative integer interval for one memory access address.

    The summary is intentionally small and sound: unsupported expressions return
    ``None`` so sanitizer falls back to the existing Z3 check.
    """
    ptr = getattr(symbolic_expr, "ptr", symbolic_expr)
    if not isinstance(ptr, SymbolicExpr):
        return None
    return _expr_interval(ptr, grid)


def access_range_proves_in_bounds(
    symbolic_expr: SymbolicExpr,
    ranges: Sequence[tuple[int, int, Any]],
    grid: tuple[int, ...] | None,
) -> bool:
    interval = access_interval_summary(symbolic_expr, grid)
    if interval is None:
        return False

    for start, end, _tensor in ranges:
        valid = IntRange(int(start), int(end))
        if valid.contains(interval):
            return True
    return False


def _expr_interval(expr: SymbolicExpr, grid: tuple[int, ...] | None) -> IntRange | None:
    if expr.loop_ctx is not None:
        return _loop_interval(expr.loop_ctx)

    if expr.op == "const":
        return _value_interval(getattr(expr, "value", None))

    if expr.op == "pid":
        return _pid_interval(expr, grid)

    if expr.op == "arange":
        start = _literal_int(getattr(expr, "start", None))
        end = _literal_int(getattr(expr, "end", None))
        if start is None or end is None:
            return None
        return _range_interval(start, end, 1)

    if expr.op == "addptr":
        return _addptr_interval(expr, grid)

    if expr.op in _UNARY_INTERVAL_OPS:
        return _unary_interval(expr, grid)

    if expr.op in _BINARY_INTERVAL_OPS:
        return _binary_interval(expr, grid)

    if expr.op in _TERNARY_INTERVAL_OPS:
        return _ternary_interval(expr, grid)

    if expr.op in _VALUE_PRESERVING_RESHAPE_OPS:
        passthrough = _passthrough_child(expr)
        if passthrough is not None:
            return _expr_interval(passthrough, grid)

    # Every other op keeps the existing Z3 path. This avoids false negatives for
    # ops with bitvector wrapping, division/mod semantics, or value-changing casts.
    return None


def _unary_interval(
    expr: SymbolicExpr, grid: tuple[int, ...] | None
) -> IntRange | None:
    child = getattr(expr, "arg", None)
    if not isinstance(child, SymbolicExpr):
        return None
    arg = _expr_interval(child, grid)
    if arg is None:
        return None

    if expr.op in {"abs", "fabs"}:
        return _abs_interval(arg)
    if expr.op == "negative":
        return IntRange(-arg.upper, -arg.lower)
    return None


def _binary_interval(
    expr: SymbolicExpr, grid: tuple[int, ...] | None
) -> IntRange | None:
    lhs_child = getattr(expr, "lhs", None)
    rhs_child = getattr(expr, "rhs", None)
    if not isinstance(lhs_child, SymbolicExpr) or not isinstance(
        rhs_child, SymbolicExpr
    ):
        return None
    lhs = _expr_interval(lhs_child, grid)
    rhs = _expr_interval(rhs_child, grid)
    if lhs is None or rhs is None:
        return None

    if expr.op == "add":
        return IntRange(lhs.lower + rhs.lower, lhs.upper + rhs.upper)
    if expr.op == "sub":
        return IntRange(lhs.lower - rhs.upper, lhs.upper - rhs.lower)
    if expr.op == "mul":
        return _mul_interval(lhs, rhs)
    if expr.op == "maximum":
        return IntRange(max(lhs.lower, rhs.lower), max(lhs.upper, rhs.upper))
    if expr.op == "minimum":
        return IntRange(min(lhs.lower, rhs.lower), min(lhs.upper, rhs.upper))
    return None


def _ternary_interval(
    expr: SymbolicExpr, grid: tuple[int, ...] | None
) -> IntRange | None:
    if expr.op == "where":
        lhs = _child_interval(expr, "lhs", grid)
        rhs = _child_interval(expr, "rhs", grid)
        return None if lhs is None or rhs is None else _merge(lhs, rhs)

    if expr.op == "fma":
        x = _child_interval(expr, "x", grid)
        y = _child_interval(expr, "y", grid)
        z = _child_interval(expr, "z", grid)
        if x is None or y is None or z is None:
            return None
        product = _mul_interval(x, y)
        return IntRange(product.lower + z.lower, product.upper + z.upper)

    return None


def _addptr_interval(
    expr: SymbolicExpr, grid: tuple[int, ...] | None
) -> IntRange | None:
    ptr = _child_interval(expr, "ptr", grid)
    offset = _child_interval(expr, "offset", grid)
    if ptr is None or offset is None:
        return None

    ptr_child = getattr(expr, "ptr", None)
    if not isinstance(ptr_child, SymbolicExpr):
        return None
    try:
        element_size = element_bytewidth(ptr_child.dtype)
    except (TypeError, ValueError):
        return None

    scaled = _mul_interval(offset, IntRange.constant(element_size))
    return IntRange(ptr.lower + scaled.lower, ptr.upper + scaled.upper)


def _child_interval(
    expr: SymbolicExpr, attr: str, grid: tuple[int, ...] | None
) -> IntRange | None:
    child = getattr(expr, attr, None)
    if not isinstance(child, SymbolicExpr):
        return None
    return _expr_interval(child, grid)


def _passthrough_child(expr: SymbolicExpr) -> SymbolicExpr | None:
    for attr in ("arg", "src"):
        child = getattr(expr, attr, None)
        if isinstance(child, SymbolicExpr):
            return child
    return None


def _abs_interval(arg: IntRange) -> IntRange:
    if arg.lower <= 0 <= arg.upper:
        return IntRange(0, max(-arg.lower, arg.upper))
    if arg.upper < 0:
        return IntRange(-arg.upper, -arg.lower)
    return arg


def _pid_interval(expr: SymbolicExpr, grid: tuple[int, ...] | None) -> IntRange | None:
    if grid is None:
        return None
    axis = _literal_int(getattr(expr, "axis", None))
    if axis is None or axis < 0 or axis >= len(grid):
        return None
    return IntRange(0, max(0, int(grid[axis]) - 1))


def _loop_interval(ctx: LoopContext) -> IntRange | None:
    return _range_interval(ctx.start, ctx.stop, ctx.step)


def _range_interval(start: int, stop: int, step: int) -> IntRange | None:
    if step == 0:
        return None
    values = range(int(start), int(stop), int(step))
    if not values:
        return None
    first = values[0]
    last = values[-1]
    return IntRange(min(first, last), max(first, last))


def _value_interval(value: Any) -> IntRange | None:
    if isinstance(value, SymbolicTensorValue):
        return _array_interval(value.data)
    if isinstance(value, np.ndarray):
        return _array_interval(value)
    if isinstance(value, (list, tuple)):
        return _sequence_interval(value)
    if isinstance(value, (int, np.integer, bool)):
        return IntRange.constant(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return IntRange.constant(int(value))
    return None


def _array_interval(array: np.ndarray) -> IntRange | None:
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    return IntRange(int(array.min()), int(array.max()))


def _sequence_interval(values: Sequence[Any]) -> IntRange | None:
    if not values:
        return None
    intervals = [_value_interval(value) for value in values]
    if any(interval is None for interval in intervals):
        return None
    concrete = [interval for interval in intervals if interval is not None]
    return IntRange(
        min(interval.lower for interval in concrete),
        max(interval.upper for interval in concrete),
    )


def _literal_int(value: Any) -> int | None:
    if isinstance(value, SymbolicExpr) and value.op == "const":
        interval = _value_interval(getattr(value, "value", None))
        if interval is not None and interval.lower == interval.upper:
            return interval.lower
    if isinstance(value, (int, np.integer, bool)):
        return int(value)
    return None


def _merge(lhs: IntRange, rhs: IntRange) -> IntRange:
    return IntRange(min(lhs.lower, rhs.lower), max(lhs.upper, rhs.upper))


def _mul_interval(lhs: IntRange, rhs: IntRange) -> IntRange:
    return _from_candidates(
        lhs.lower * rhs.lower,
        lhs.lower * rhs.upper,
        lhs.upper * rhs.lower,
        lhs.upper * rhs.upper,
    )


def _from_candidates(*values: int) -> IntRange:
    return IntRange(min(values), max(values))
