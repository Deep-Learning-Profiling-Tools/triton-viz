import linecache
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


# Unified superset of framework path fragments used to filter out non-user frames.
FRAMEWORK_PATHS: list[str] = [
    "triton_viz/core/",
    "triton_viz/clients/",
    "triton_viz/utils/",
    "triton_viz/wrapper",
    "triton/runtime/",
    "triton/language/",
    "site-packages/triton/",
    "runpy",
    "IPython",
]

_BIN_DIR = os.path.normcase(os.path.dirname(sys.executable))


@dataclass
class TracebackInfo:
    filename: str
    lineno: int
    func_name: str
    line_of_code: str


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_user_frames(skip_tail: int = 0) -> list[traceback.FrameSummary]:
    """Return user-code frames from the current call stack.

    Framework frames (matching any entry in ``FRAMEWORK_PATHS``) are removed.
    ``skip_tail`` additional frames are stripped from the *end* of the raw stack
    before filtering (useful to exclude the caller itself).
    """
    stack: list[traceback.FrameSummary] = list(traceback.extract_stack())
    if skip_tail > 0:
        stack = stack[:-skip_tail]

    cleaned: list[traceback.FrameSummary] = []
    for f in stack:
        fn = f.filename.replace("\\", "/")
        if any(s in fn for s in FRAMEWORK_PATHS):
            continue
        # Skip console_scripts entry points (e.g. triton-sanitizer, triton-profiler)
        if os.path.normcase(os.path.dirname(os.path.abspath(f.filename))) == _BIN_DIR:
            continue
        cleaned.append(f)

    if cleaned:
        return cleaned

    # fallback: last non-"<...>" frame
    for f in reversed(stack):
        if not f.filename.startswith("<"):
            return [f]
    return stack[-1:] if stack else []


def frame_to_traceback_info(frame: traceback.FrameSummary) -> "TracebackInfo":
    """Convert a ``FrameSummary`` to a ``TracebackInfo``."""
    return TracebackInfo(
        filename=frame.filename,
        lineno=frame.lineno or 0,
        func_name=frame.name,
        line_of_code=frame.line or "",
    )


# ---------------------------------------------------------------------------
# Sanitizer-oriented helpers (originally in report.py)
# ---------------------------------------------------------------------------


def locate_user_frame() -> tuple[str, int, str] | None:
    """Walk the live call stack and return the first user-code location.

    Returns ``(filename, lineno, func_name)`` or ``None``.
    """
    from types import FrameType

    frame: FrameType | None = sys._getframe()

    while frame is not None:
        filename = Path(frame.f_code.co_filename).as_posix()

        # Skip Python internals
        if filename.startswith("<"):
            frame = frame.f_back
            continue

        is_framework = any(path in filename for path in FRAMEWORK_PATHS)

        # User code: not in framework paths, OR in examples directory
        if not is_framework or "examples/" in filename:
            return (
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_name,
            )

        frame = frame.f_back

    return None


def location_to_traceback_info(
    source_location: tuple[str, int, str],
) -> "TracebackInfo":
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
