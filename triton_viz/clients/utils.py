import itertools
import sys
import threading
import time
import traceback
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
import torch


P = ParamSpec("P")
R = TypeVar("R")
_TIMER_STATE = threading.local()


def _get_timer_stack() -> list[dict[str, float]]:
    stack = getattr(_TIMER_STATE, "stack", None)
    if stack is None:
        stack = []
        _TIMER_STATE.stack = stack
    return stack


@overload
def frame_timer(fn: Callable[P, R], *, label: str = "") -> Callable[P, R]:
    ...


@overload
def frame_timer(*, label: str = "") -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


def frame_timer(
    fn: Callable[P, R] | None = None, *, label: str = ""
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            stack = _get_timer_stack()
            frame = {"child_time": 0.0}
            stack.append(frame)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1]["child_time"] += elapsed
                exclusive_ms = (elapsed - frame["child_time"]) * 1000.0
                if label == "z3":
                    self = args[0] if args else None
                    op_name = getattr(self, "op", type(self).__name__)
                    class_name = type(self).__name__
                    name = f"{label} {class_name}({op_name})"
                elif label:
                    name = f"{label} {func.__name__}"
                else:
                    name = func.__name__
                print(
                    f"{name} {exclusive_ms:.3f}ms",
                    file=sys.stderr,
                    flush=True,
                )

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)


def check_out_of_bounds_access(
    ptrs: npt.NDArray, masks: npt.NDArray[np.bool_], tensor: torch.Tensor
):
    offsets = ptrs - tensor.data_ptr()
    max_valid_offset = np.prod(tensor.shape) * tensor.element_size()
    valid_access_masks = (offsets >= 0) & (offsets < max_valid_offset)
    invalid_access_masks = (~valid_access_masks) & masks
    corrected_offsets = np.where(valid_access_masks, offsets, 0)
    return {
        "tensor": tensor,
        "offsets": offsets,
        "masks": masks,
        "valid_access_masks": valid_access_masks & masks,
        "invalid_access_masks": invalid_access_masks,
        "corrected_offsets": corrected_offsets,
    }


def check_storage_contiguous(tensor: torch.Tensor):
    # Note that this is different from if a tensor is accessed contiguously, so we cannot use tensor.is_contiguous()
    # 1. Sort strides from smallest to largest
    # 2. If the tensor is contiguous, the stride product should be the same of the shape product of all previous dimensions
    from triton.runtime.jit import TensorWrapper

    if isinstance(tensor, TensorWrapper):
        tensor = tensor.base
    assert (
        type(tensor) == torch.Tensor
    ), f"Only torch.Tensor is supported, but found {type(tensor)}"
    shape_prod = 1
    indices = sorted(range(len(tensor.stride())), key=tensor.stride().__getitem__)
    for i, index in enumerate(indices):
        stride = tensor.stride(index)
        shape = tensor.shape[index]
        if i == 0 and stride != 1:
            return False
        if i != 0 and stride != shape_prod:
            return False
        shape_prod *= shape
    return True


def check_inner_stride_equal_to_one(tensor: torch.Tensor):
    return sorted(tensor.stride())[0] == 1


def get_physical_addr_from_tensor_slice(tensor: torch.Tensor) -> list[tuple[int, int]]:
    if sorted(tensor.stride())[0] != 1:
        raise ValueError("inner dim must be contiguous!")
    dims = tensor.dim()
    inner_dim = min(range(dims), key=lambda d: tensor.stride(d))
    outer_dims = [d for d in range(dims) if d != inner_dim]

    segments = []
    for idxs in itertools.product(*(range(tensor.size(d)) for d in outer_dims)):
        offset = int(tensor.storage_offset()) + sum(
            idx * int(tensor.stride(d)) for idx, d in zip(idxs, outer_dims)
        )
        inner_dim_size = int(tensor.size(inner_dim))
        segments.append(
            (
                tensor.data_ptr() + offset * tensor.element_size(),
                tensor.data_ptr()
                + (offset + inner_dim_size - 1) * tensor.element_size(),
            )
        )
    return segments


def extract_complete_statement_from_line(filename: str, lineno: int | None) -> str:
    """
    Extract the complete Python statement containing the specified line from a source file.

    This function handles multi-line Python statements (e.g., multi-line function calls).
    It performs the following steps:

    1. Determine if the target line is the start of a new statement:
       - Check if the previous line is a complete, non-continuing statement
         (balanced brackets and doesn't end with continuation characters)
       - If the previous line is complete, the target line starts a new statement
       - Otherwise, search backwards to find the actual start of the statement

    2. Collect lines forward from the start until the statement is complete:
       - Use bracket/parenthesis/brace balance to determine statement completion
       - Once we reach the target line and all brackets are balanced, we're done

    3. Remove excess indentation while preserving relative indentation

    Args:
        filename: Path to the source code file
        lineno: Target line number (1-based, i.e., first line is 1)

    Returns:
        The complete statement as a string with excess indentation removed.
        Returns empty string if extraction fails.

    Example:
        For code like this (where line 35 contains the tl.load call):
        -----------------------
        31: src_data = tl.load(
        32:     K + offset,
        33:     mask=mask,
        34:     other=0.0,
        35: )
        -----------------------
        Calling extract_complete_statement_from_line(filename, 35) returns
        the complete 5-line statement.
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        if lineno is None or lineno < 1 or lineno > len(lines):
            return ""

        # Target line index (0-based)
        target_idx = lineno - 1

        # Step 1: Determine the statement's starting line
        # By default, assume target line is the start
        start_idx = target_idx

        if target_idx > 0:
            # Check if the previous line is a complete, non-continuing statement
            prev_line = lines[target_idx - 1].rstrip()
            prev_stripped = prev_line.strip()

            # Count bracket balance in the previous line
            paren_count = prev_line.count("(") - prev_line.count(")")
            bracket_count = prev_line.count("[") - prev_line.count("]")
            brace_count = prev_line.count("{") - prev_line.count("}")

            # If previous line has balanced brackets and doesn't end with continuation
            if (
                paren_count == 0
                and bracket_count == 0
                and brace_count == 0
                and prev_stripped
                and not prev_stripped.endswith(("\\", ",", "(", "[", "{"))
            ):
                # Previous line is complete, so target line starts a new statement
                start_idx = target_idx
            else:
                # Previous line is incomplete or continuing, search backwards for true start
                # Track bracket depth by scanning in reverse
                paren_depth = 0
                bracket_depth = 0
                brace_depth = 0

                for i in range(target_idx - 1, -1, -1):
                    line = lines[i]
                    # Traverse characters in reverse order to track depth
                    # Reverse because we're going backwards: ')' means we need '('
                    for char in reversed(line):
                        if char == ")":
                            paren_depth += 1
                        elif char == "(":
                            paren_depth -= 1
                        elif char == "]":
                            bracket_depth += 1
                        elif char == "[":
                            bracket_depth -= 1
                        elif char == "}":
                            brace_depth += 1
                        elif char == "{":
                            brace_depth -= 1

                    # Negative depth means unmatched opening bracket, part of statement
                    if paren_depth < 0 or bracket_depth < 0 or brace_depth < 0:
                        start_idx = i
                    # If we return to balanced state
                    elif paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                        stripped = line.rstrip()
                        # Check if this line doesn't continue
                        if not stripped.endswith(("\\", ",", "(", "[", "{")):
                            # This is a complete previous statement, next line is start
                            start_idx = i + 1
                            break

                    if i == 0:
                        start_idx = 0

        # Step 2: Collect lines forward from start until statement is complete
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        statement_lines = []

        i = start_idx
        while i < len(lines):
            line = lines[i].rstrip()
            statement_lines.append(line)

            # Count brackets in current line
            for char in line:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1

            # If we've reached or passed target line and all brackets are balanced
            if (
                i >= target_idx
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                stripped = line.strip()
                # Check if line doesn't continue
                if not stripped.endswith((",", "\\")):
                    # Statement is complete
                    break

            i += 1
            # Safety limit: don't collect more than 20 lines
            if i - start_idx > 20:
                break

        # Step 3: Clean up indentation
        full_statement = "\n".join(statement_lines)
        if statement_lines:
            # Find minimum indentation (excluding empty lines)
            non_empty_lines = [line for line in statement_lines if line.strip()]
            if non_empty_lines:
                min_indent = min(
                    len(line) - len(line.lstrip()) for line in non_empty_lines
                )
                # Remove common indentation
                full_statement = "\n".join(
                    line[min_indent:] if len(line) > min_indent else line
                    for line in statement_lines
                )

        return full_statement.strip()
    except Exception:
        # Return empty string on any error
        return ""


def get_source_location_from_stack(
    exclude_patterns: list[str] | None = None,
) -> Tuple[int, str, str]:
    """
    Extract user code location information from the current call stack.

    This function examines the current call stack, filters out internal framework
    frames (e.g., triton, triton_viz), finds the first user code location, and
    extracts the complete statement at that location.

    Args:
        exclude_patterns: List of file path patterns to filter out. Defaults to
                         filtering triton and triton_viz internal code.
                         Each pattern is a substring of the file path, e.g., "triton/runtime".

    Returns:
        A tuple (lineno, filename, code):
        - lineno: Line number (1-based)
        - filename: Full path to the source file
        - code: Complete Python statement at that location (may span multiple lines)

        Returns (0, "", "") if no user code location is found.

    Example:
        When called inside a triton kernel:
        ---------------------------------------------
        @triton.jit
        def my_kernel(...):
            data = tl.load(ptr, mask=mask)  # Line 42
        ---------------------------------------------
        If get_source_location_from_stack() is called inside tl.load,
        it returns (42, "/path/to/my_kernel.py", "data = tl.load(ptr, mask=mask)")
    """
    # Default filter patterns: exclude triton and triton_viz internals
    if exclude_patterns is None:
        exclude_patterns = [
            "triton/runtime",
            "triton/language",
            "triton_viz/core",
            "triton_viz/clients",
        ]

    # Get current call stack
    stack = traceback.extract_stack()

    # Search from top of stack downward (reversed) for first non-filtered frame
    for frame in reversed(stack):
        if not frame.lineno:
            continue
        # Check if this frame's file path matches any filter pattern
        if not any(pattern in frame.filename for pattern in exclude_patterns):
            # Found user code, extract complete statement
            full_statement = extract_complete_statement_from_line(
                frame.filename, frame.lineno
            )
            return frame.lineno, frame.filename, full_statement

    # If no user code location found, return empty values
    return 0, "", ""
