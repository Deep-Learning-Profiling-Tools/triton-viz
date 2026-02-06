import linecache
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# Unified superset of framework path fragments used to filter out non-user frames.
FRAMEWORK_PATHS: list[str] = [
    "triton_viz/core/",
    "triton_viz/clients/",
    "triton_viz/utils/",
    "triton/runtime/",
    "triton/language/",
    "site-packages/triton/",
    "runpy.py",
    "IPython",
]


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


def get_user_code_location() -> tuple[str, int, str] | None:
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


def get_sanitizer_traceback_info() -> list["TracebackInfo"]:
    """Extract user-code frames for sanitizer OOB reports.

    This is the logic previously in ``report._get_traceback_info``.
    """
    stack_summary = traceback.extract_stack()
    user_code_tracebacks: list[TracebackInfo] = []

    # First pass: collect user code frames that follow a patch.py call
    for i, frame in enumerate(stack_summary):
        if any(path in frame.filename.replace("\\", "/") for path in FRAMEWORK_PATHS):
            if "examples/" not in frame.filename.replace("\\", "/"):
                continue

        if frame.filename.startswith("<"):
            continue

        if i > 0:
            prev_frame = stack_summary[i - 1]
            if "triton_viz/core/patch.py" in prev_frame.filename:
                user_code_tracebacks.append(
                    TracebackInfo(
                        filename=frame.filename,
                        lineno=frame.lineno or 0,
                        func_name=frame.name,
                        line_of_code=frame.line or "",
                    )
                )

    # Fallback: look for frames after _jit_function_call / _grid_executor_call
    if not user_code_tracebacks:
        for i, frame in enumerate(stack_summary):
            if (
                "_jit_function_call" in frame.name
                or "_grid_executor_call" in frame.name
            ) and "triton_viz/core/patch.py" in frame.filename:
                if i + 1 < len(stack_summary):
                    next_frame = stack_summary[i + 1]
                    if not any(path in next_frame.filename for path in FRAMEWORK_PATHS):
                        user_code_tracebacks.append(
                            TracebackInfo(
                                filename=next_frame.filename,
                                lineno=next_frame.lineno or 0,
                                func_name=next_frame.name,
                                line_of_code=next_frame.line or "",
                            )
                        )

    # Reverse so the most immediate error location comes first
    user_code_tracebacks.reverse()

    # Remove duplicates while preserving order
    seen: set[tuple[str, int]] = set()
    unique_tracebacks: list[TracebackInfo] = []
    for tb in user_code_tracebacks:
        key = (tb.filename, tb.lineno)
        if key not in seen:
            seen.add(key)
            unique_tracebacks.append(tb)

    return unique_tracebacks


# ---------------------------------------------------------------------------
# Source code reading utilities (originally in interface.py)
# ---------------------------------------------------------------------------


def safe_read_file_segment(
    filename: str,
    lineno: int,
    context: int = 8,
    cwd: str | None = None,
) -> dict | None:
    """Read a segment of a source file around *lineno*.

    Only files under *cwd* (defaults to ``os.getcwd()``) are allowed.
    """
    try:
        if cwd is None:
            cwd = os.path.realpath(os.getcwd())
        path = os.path.realpath(filename)
        if not path.startswith(cwd):
            return None
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


def score_traceback_frame(tb: dict, cwd: str | None = None) -> int:
    """Score a traceback frame dict for relevance to the user's kernel code.

    Higher scores indicate the frame is more likely the interesting user code.
    *cwd* defaults to ``os.path.realpath(os.getcwd())``.
    """
    if cwd is None:
        cwd = os.path.realpath(os.getcwd())

    fn = tb.get("filename") or ""
    line = (tb.get("line") or "").strip()
    name = tb.get("name") or ""
    p = os.path.realpath(fn)
    score = 0

    if "tl.load" in line or "tl.store" in line:
        score += 100
    elif "tl." in line:
        score += 50

    if p.startswith(cwd):
        score += 5
    if any(
        s in p
        for s in ["site-packages", "triton_viz/", "triton/", "runpy.py", "IPython"]
    ):
        score -= 10

    if name.endswith("_kernel") or "kernel" in name:
        score += 3

    if "examples" in p:
        score += 1

    return score


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


def get_source_location_from_stack(
    exclude_patterns: list[str] | None = None,
) -> Tuple[int, str, str]:
    """Extract user code location from the current call stack.

    Filters out internal framework frames and returns
    ``(lineno, filename, complete_statement)``, or ``(0, "", "")`` if not found.
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "triton/runtime",
            "triton/language",
            "triton_viz/core",
            "triton_viz/clients",
        ]

    stack = traceback.extract_stack()

    for frame in reversed(stack):
        if not frame.lineno:
            continue
        if not any(pattern in frame.filename for pattern in exclude_patterns):
            full_statement = extract_complete_statement_from_line(
                frame.filename, frame.lineno
            )
            return frame.lineno, frame.filename, full_statement

    return 0, "", ""
