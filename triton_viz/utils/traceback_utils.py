import linecache
import os
import sys
from dataclasses import dataclass
from types import FrameType
from typing import Callable

# IDs all triton_viz-traced functions to display its code within clients
CODE_KEYS: set[tuple[str, str]] = set()


@dataclass
class TracebackInfo:
    filename: str
    lineno: int
    func_name: str
    line_of_code: str


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def get_code_key(obj: FrameType | Callable) -> tuple[str, str]:
    """
    Given an object that points to a function (a call frame/function itself),
    inspect the function's code and create a key that tells triton-viz to display
    the code for displaying in clients (i.e. Code View in visualizer).
    """
    code = obj.f_code if isinstance(obj, FrameType) else obj.__code__
    return (os.path.realpath(code.co_filename), code.co_qualname)


def extract_user_frames(num_frames: int = 0) -> list[TracebackInfo]:
    """
    Return traceback info from frames from the current call stack that are running
    inside functions marked with @triton_viz.trace(...) or @triton_viz.trace_source.
    Args:
        num_frames: max number of frames to return (i.e. num_frames = 1 means that only last frame
        inside a traced function is returned). If zero (default), return all frames.
    """
    matched: list[TracebackInfo] = []
    enough_frames = lambda: num_frames != 0 and len(matched) >= num_frames
    frame: FrameType | None = sys._getframe(1)
    while frame and not enough_frames():
        code_key = get_code_key(frame)
        if code_key in CODE_KEYS:
            filename, func_name = code_key
            lineno = frame.f_lineno
            line_of_code = linecache.getline(filename, lineno).rstrip()
            matched.insert(0, TracebackInfo(filename, lineno, func_name, line_of_code))
        frame = frame.f_back
    return matched


# ---------------------------------------------------------------------------
# Sanitizer-oriented helpers (originally in report.py)
# ---------------------------------------------------------------------------


def location_to_traceback_info(
    source_location: tuple[str, int, str],
) -> TracebackInfo:
    """Convert a ``(filename, lineno, func_name)`` tuple to ``TracebackInfo``."""
    filename, lineno, func_name = source_location
    line_of_code = linecache.getline(filename, lineno).rstrip()
    return TracebackInfo(
        filename=filename,
        lineno=lineno,
        func_name=func_name,
        line_of_code=line_of_code,
    )


# ---------------------------------------------------------------------------
# Source code reading utilities (originally in interface.py)
# ---------------------------------------------------------------------------


def read_source_segment(
    filename: str,
    lineno: int,
    context: int = 8,
) -> dict | None:
    """Read a segment of a source file around *lineno*.

    Returns a dict with ``filename``, ``lineno``, ``start``, ``end``,
    ``highlight``, and ``lines`` keys, or ``None`` on failure.
    """
    try:
        path = os.path.realpath(filename)
        start = max(1, lineno - context)
        end = lineno + context
        lines: list[dict] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if i > end:
                    break
                lines.append({"no": i, "text": line.rstrip("\n")})
        return {
            "filename": path,
            "lineno": lineno,
            "start": start,
            "end": end,
            "highlight": lineno,
            "lines": lines,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statement extraction (originally in clients/utils.py)
# ---------------------------------------------------------------------------


def extract_complete_statement_from_line(filename: str, lineno: int | None) -> str:
    """Extract the complete Python statement containing the specified line.

    Handles multi-line statements (e.g. multi-line function calls) by tracking
    bracket balance.  Returns the statement with excess indentation removed, or
    an empty string on failure.
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        if lineno is None or lineno < 1 or lineno > len(lines):
            return ""

        target_idx = lineno - 1
        start_idx = target_idx

        if target_idx > 0:
            prev_line = lines[target_idx - 1].rstrip()
            prev_stripped = prev_line.strip()

            paren_count = prev_line.count("(") - prev_line.count(")")
            bracket_count = prev_line.count("[") - prev_line.count("]")
            brace_count = prev_line.count("{") - prev_line.count("}")

            if (
                paren_count == 0
                and bracket_count == 0
                and brace_count == 0
                and prev_stripped
                and not prev_stripped.endswith(("\\", ",", "(", "[", "{"))
            ):
                start_idx = target_idx
            else:
                paren_depth = 0
                bracket_depth = 0
                brace_depth = 0

                for i in range(target_idx - 1, -1, -1):
                    line = lines[i]
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

                    if paren_depth < 0 or bracket_depth < 0 or brace_depth < 0:
                        start_idx = i
                    elif paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                        stripped = line.rstrip()
                        if not stripped.endswith(("\\", ",", "(", "[", "{")):
                            start_idx = i + 1
                            break

                    if i == 0:
                        start_idx = 0

        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        statement_lines: list[str] = []

        i = start_idx
        while i < len(lines):
            line = lines[i].rstrip()
            statement_lines.append(line)

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

            if (
                i >= target_idx
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                stripped = line.strip()
                if not stripped.endswith((",", "\\")):
                    break

            i += 1
            if i - start_idx > 20:
                break

        full_statement = "\n".join(statement_lines)
        if statement_lines:
            non_empty = [sl for sl in statement_lines if sl.strip()]
            if non_empty:
                min_indent = min(len(sl) - len(sl.lstrip()) for sl in non_empty)
                full_statement = "\n".join(
                    sl[min_indent:] if len(sl) > min_indent else sl
                    for sl in statement_lines
                )

        return full_statement.strip()
    except Exception:
        return ""
