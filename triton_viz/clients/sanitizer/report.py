import sys
import traceback
from typing import Optional, TYPE_CHECKING

import numpy as np

from ...core.data import Load, Store
from .data import (
    TracebackInfo,
    OutOfBoundsRecord,
    OutOfBoundsRecordBruteForce,
    OutOfBoundsRecordZ3,
)

if TYPE_CHECKING:
    from .sanitizer import SymbolicExpr


# Paths that identify framework code (not user code)
_FRAMEWORK_PATHS = [
    "triton_viz/core/",
    "triton_viz/clients/",
    "triton/runtime/",
    "triton/language/",
    "site-packages/triton/",
]


def _get_user_code_location() -> Optional[tuple[str, int, str]]:
    """
    Lightweight function to capture the current user code location.

    This is much faster than traceback.extract_stack() because it:
    1. Directly traverses frame objects instead of creating FrameSummary objects
    2. Only extracts essential info (filename, lineno, func_name)
    3. Does not read source files

    Returns:
        A tuple of (filename, lineno, func_name) for the user code frame,
        or None if no user code frame is found.
    """
    from types import FrameType

    frame: Optional[FrameType] = sys._getframe()

    while frame is not None:
        filename = frame.f_code.co_filename.replace("\\", "/")

        # Skip Python internals
        if filename.startswith("<"):
            frame = frame.f_back
            continue

        # Check if this is user code (not in framework paths)
        is_framework = any(path in filename for path in _FRAMEWORK_PATHS)

        # User code: not in framework paths, OR in examples directory
        if not is_framework or "examples/" in filename:
            return (
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_name,
            )

        frame = frame.f_back

    return None


def _read_source_line(filename: str, lineno: int) -> str:
    """
    Read a single line of source code from a file.

    This is called only when an error is detected, to minimize I/O overhead
    during normal execution.

    Args:
        filename: Path to the source file
        lineno: Line number to read (1-indexed)

    Returns:
        The source line content, or empty string if reading fails.
    """
    try:
        with open(filename, "r") as f:
            for i, line in enumerate(f, 1):
                if i == lineno:
                    return line.rstrip()
    except (FileNotFoundError, IOError):
        pass
    return ""


def _location_to_traceback_info(
    source_location: tuple[str, int, str],
) -> TracebackInfo:
    """
    Convert a lightweight source location tuple to a full TracebackInfo object.

    This reads the source line from file, so it should only be called when
    an error is being reported.

    Args:
        source_location: A tuple of (filename, lineno, func_name)

    Returns:
        A TracebackInfo object with full information including source line.
    """
    filename, lineno, func_name = source_location
    line_of_code = _read_source_line(filename, lineno)
    return TracebackInfo(
        filename=filename,
        lineno=lineno,
        func_name=func_name,
        line_of_code=line_of_code,
    )


def print_oob_record(oob_record: OutOfBoundsRecord, max_display=10):
    """
    Print detailed logs for a given OOB record.

    Parameters
    ----------
    oob_record : OutOfBoundsRecord
        The record containing information about out-of-bounds accesses.
    max_display : int
        Maximum number of invalid accesses to display in detail.
    """
    if issubclass(oob_record.op_type, Store):
        op_type = "Store"
    elif issubclass(oob_record.op_type, Load):
        op_type = "Load"
    else:
        assert False, "Not supported op type: " + str(oob_record.op_type)

    # Read the tensor from the record
    tensor = oob_record.tensor

    # Basic info about the OOB event
    print("============================================================")
    print("                 Out-Of-Bounds Access Detected              ")
    print("============================================================")
    print(f"Operation: {op_type}")
    tensor_name = getattr(oob_record, "tensor_name", None)
    if tensor_name:
        print(f"Tensor Arg: {tensor_name}")
    print(
        f"Tensor Info: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}"
    )
    print(f"Tensor base memory address: {tensor.data_ptr()}")
    print(
        "Valid Access Range: [0, %d)" % (np.prod(tensor.shape) * tensor.element_size())
    )
    for traceback_info in oob_record.user_code_tracebacks:
        print(
            f"File: {traceback_info.filename}, Line: {traceback_info.lineno}, in {traceback_info.func_name}"
        )
        print(f"  Code: {traceback_info.line_of_code}")
    print("------------------------------------------------------------")

    if isinstance(oob_record, OutOfBoundsRecordBruteForce):
        # Convert memoryviews to NumPy arrays
        offsets_arr = np.array(oob_record.offsets)
        invalid_access_masks_arr = np.array(oob_record.invalid_access_masks)

        # Determine all invalid indices
        invalid_indices = np.where(invalid_access_masks_arr.flatten())[0]
        assert len(invalid_indices) != 0, "No invalid accesses found in this record."

        # Print OOB details
        print(f"Total invalid accesses: {len(invalid_indices)}")
        invalid_offsets = offsets_arr.flatten()[invalid_indices]
        print("Invalid offsets:")
        print(invalid_offsets)

    elif isinstance(oob_record, OutOfBoundsRecordZ3):
        # Read the violation index and constraints
        violation_address = oob_record.violation_address
        constraints = oob_record.constraints

        # Print OOB details
        print(f"Invalid access detected at address: {violation_address}")
        if constraints is not None:
            print("Constraints:")
            print(constraints)

    else:
        raise NotImplementedError(
            "Invalid OutOfBoundsRecord type: " + str(type(oob_record))
        )

    print("============================================================")
    print("            End of Out-Of-Bounds Record Details             ")
    print("============================================================")


def print_oob_record_pdb_style(
    oob_record: OutOfBoundsRecord, symbolic_expr: Optional["SymbolicExpr"] = None
):
    """
    Print a comprehensive diagnostic report for OOB errors in PDB-style format.

    Parameters
    ----------
    oob_record : OutOfBoundsRecord
        The record containing information about out-of-bounds accesses.
    symbolic_expr : Optional[SymbolicExpr]
        The symbolic expression tree that led to the OOB access.
    """
    from pathlib import Path

    # Determine operation type
    if issubclass(oob_record.op_type, Store):
        op_color = "\033[91m"  # Red for store
    elif issubclass(oob_record.op_type, Load):
        op_color = "\033[93m"  # Yellow for load
    else:
        op_color = "\033[0m"

    reset_color = "\033[0m"
    bold = "\033[1m"
    cyan = "\033[96m"
    green = "\033[92m"
    magenta = "\033[95m"

    # ===================== Header =====================
    print(f"\n{bold}{op_color}🚨 ILLEGAL MEMORY ACCESS DETECTED 🚨{reset_color}")

    # ===================== PDB-Style Code Context =====================
    if oob_record.user_code_tracebacks:
        print(f"{bold}{cyan}━━━ Code Context ━━━{reset_color}")

        for tb_info in oob_record.user_code_tracebacks:
            # Try to read the source file to show context
            try:
                with open(tb_info.filename, "r") as f:
                    lines = f.readlines()

                # Show 3 lines before and after for context
                start_line = max(0, tb_info.lineno - 4)
                end_line = min(len(lines), tb_info.lineno + 3)

                print(f"  {magenta}File:{reset_color} {tb_info.filename}")
                print(f"  {magenta}Function:{reset_color} {tb_info.func_name}")
                print(f"  {magenta}Line {tb_info.lineno}:{reset_color}")

                for i in range(start_line, end_line):
                    line_num = i + 1
                    line_content = lines[i].rstrip()

                    if line_num == tb_info.lineno:
                        # Highlight the problematic line
                        print(
                            f"{op_color}→ {line_num:4d} │ {line_content}{reset_color}"
                        )
                    else:
                        print(f"  {line_num:4d} │ {line_content}")

            except (FileNotFoundError, IOError):
                # Fallback if we can't read the file
                print(
                    f"  {magenta}File:{reset_color} {tb_info.filename}:{tb_info.lineno}"
                )
                print(f"  {magenta}Function:{reset_color} {tb_info.func_name}")
                print(f"  {magenta}Code:{reset_color} {tb_info.line_of_code}")

            break  # Only show the first traceback for brevity

    # ===================== Tensor Information =====================
    print(f"{bold}{cyan}━━━ Tensor Information ━━━{reset_color}")
    tensor = oob_record.tensor
    tensor_name = getattr(oob_record, "tensor_name", None)

    # Display two items per line to save vertical space
    # Calculate column widths for alignment
    col1_width = 12  # Width for first label
    col2_width = 25  # Width for first value
    col3_width = 12  # Width for second label

    if tensor_name:
        print(f"  {green}{'arg:':<{col1_width}}{reset_color} {tensor_name}")
    print(
        f"  {green}{'dtype:':<{col1_width}}{reset_color} {str(tensor.dtype):<{col2_width}} {green}{'shape:':<{col3_width}}{reset_color} {tensor.shape}"
    )
    print(
        f"  {green}{'strides:':<{col1_width}}{reset_color} {str(tensor.stride()):<{col2_width}} {green}{'device:':<{col3_width}}{reset_color} {tensor.device}"
    )
    print(
        f"  {green}{'contiguous:':<{col1_width}}{reset_color} {str(tensor.is_contiguous()):<{col2_width}} {green}{'base_ptr:':<{col3_width}}{reset_color} 0x{tensor.data_ptr():016x}"
    )

    total_bytes = np.prod(tensor.shape) * tensor.element_size()
    size_str = f"{total_bytes} bytes"
    range_str = (
        f"[0x{tensor.data_ptr():016x}, 0x{tensor.data_ptr() + total_bytes:016x})"
    )
    print(
        f"  {green}{'size:':<{col1_width}}{reset_color} {size_str:<{col2_width}} {green}{'valid_range:':<{col3_width}}{reset_color} {range_str}"
    )

    # ===================== Call Stack =====================
    print(f"{bold}{cyan}━━━ Call Stack ━━━{reset_color}")

    if oob_record.user_code_tracebacks:
        for i, tb_info in enumerate(oob_record.user_code_tracebacks):
            frame_num = len(oob_record.user_code_tracebacks) - i
            file_name = Path(tb_info.filename).name
            print(f"  #{frame_num} {tb_info.func_name} at {file_name}:{tb_info.lineno}")
            if tb_info.line_of_code:
                print(f"     └─ {tb_info.line_of_code.strip()}")
    else:
        print("  (No traceback information available)")

    # ===================== Violation Details =====================
    print(f"{bold}{cyan}━━━ Violation Details ━━━{reset_color}")

    if isinstance(oob_record, OutOfBoundsRecordBruteForce):
        offsets_arr = np.array(oob_record.offsets)
        invalid_access_masks_arr = np.array(oob_record.invalid_access_masks)
        invalid_indices = np.where(invalid_access_masks_arr.flatten())[0]

        if len(invalid_indices) > 0:
            print(f"  {green}Total violations:{reset_color} {len(invalid_indices)}")
            invalid_offsets = offsets_arr.flatten()[invalid_indices]

            # Show first few invalid offsets
            display_count = min(10, len(invalid_offsets))
            print(f"  {green}Invalid offsets (first {display_count}):{reset_color}")
            for offset in invalid_offsets[:display_count]:
                print(f"    • 0x{offset:016x} (offset: {offset})")

            if len(invalid_offsets) > display_count:
                print(f"    ... and {len(invalid_offsets) - display_count} more")

    elif isinstance(oob_record, OutOfBoundsRecordZ3):
        if hasattr(oob_record, "violation_address"):
            print(
                f"  {green}Violation address:{reset_color} 0x{oob_record.violation_address:016x}"
            )

        constraints = getattr(oob_record, "constraints", None)
        if constraints is not None:
            print(f"  {green}SMT constraints:{reset_color}")
            print(f"    {constraints}")

    # ===================== Symbolic Expression Tree =====================
    if symbolic_expr is not None:
        print(f"{bold}{cyan}━━━ Symbolic Expression Tree ━━━{reset_color}")

        if hasattr(symbolic_expr, "to_tree_str"):
            tree_str = symbolic_expr.to_tree_str()
            # Indent the tree for better display
            for line in tree_str.split("\n"):
                if line:
                    print(f"  {line}")
        else:
            print(f"  {symbolic_expr}")

    print(f"{bold}End of IMA Diagnostic Report{reset_color}\n")


def _get_traceback_info():
    """
    Extract user code frames from the call stack, focusing on actual user code
    that contains the memory access operations.

    Why do both _grid_executor_call and _jit_function_call appear in the call stacks?
    1) Main kernel dispatch (kernel[grid](...)) triggers _grid_executor_call.
    2) Inlined @triton.jit functions trigger _jit_function_call.
    3) Some code sees only _grid_executor_call if no separate JIT function is present or patched.
    4) Complex kernels (e.g., fused_attention) may show both: outer dispatch and inner JIT calls.
    """
    stack_summary = traceback.extract_stack()
    user_code_tracebacks = []

    # Filter out framework code to find user code frames
    # We want to find frames that:
    # 1. Are not in triton_viz internal files (except examples)
    # 2. Are not in triton runtime/language files
    # 3. Are user-defined functions

    framework_paths = [
        "triton_viz/core/",
        "triton_viz/clients/",
        "triton/runtime/",
        "triton/language/",
        "site-packages/triton/",
    ]

    # First pass: collect all potential user code frames
    for i, frame in enumerate(stack_summary):
        # Skip framework code
        if any(path in frame.filename.replace("\\", "/") for path in framework_paths):
            # Exception: include examples directory
            if "examples/" not in frame.filename.replace("\\", "/"):
                continue

        # Skip Python internals
        if frame.filename.startswith("<"):
            continue

        # Check if this frame is just after a patch.py call
        # This usually indicates a transition from framework to user code
        if i > 0:
            prev_frame = stack_summary[i - 1]
            if "triton_viz/core/patch.py" in prev_frame.filename:
                # This is likely user code called by the framework
                user_code_tracebacks.append(
                    TracebackInfo(
                        filename=frame.filename,
                        lineno=frame.lineno,
                        func_name=frame.name,
                        line_of_code=frame.line,
                    )
                )

    # If we didn't find any user code using the above method,
    # fall back to the original approach but collect ALL relevant frames
    if not user_code_tracebacks:
        for i, frame in enumerate(stack_summary):
            if (
                "_jit_function_call" in frame.name
                or "_grid_executor_call" in frame.name
            ) and "triton_viz/core/patch.py" in frame.filename:
                # Look at the next frame which should be user code
                if i + 1 < len(stack_summary):
                    next_frame = stack_summary[i + 1]
                    # Only add if it's not already in our list and is user code
                    if not any(path in next_frame.filename for path in framework_paths):
                        user_code_tracebacks.append(
                            TracebackInfo(
                                filename=next_frame.filename,
                                lineno=next_frame.lineno,
                                func_name=next_frame.name,
                                line_of_code=next_frame.line,
                            )
                        )

    # Reverse the list so the most immediate error location comes first
    # (closest to the actual tl.load/tl.store operation)
    user_code_tracebacks.reverse()

    # Remove duplicates while preserving order
    seen = set()
    unique_tracebacks = []
    for tb in user_code_tracebacks:
        key = (tb.filename, tb.lineno)
        if key not in seen:
            seen.add(key)
            unique_tracebacks.append(tb)

    return unique_tracebacks
