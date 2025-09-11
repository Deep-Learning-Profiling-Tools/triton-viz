import traceback
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, Union
import re


import numpy as np
from torch import Tensor
from anytree import Node, RenderTree
from z3 import (
    Solver,
    Int,
    IntVal,
    If,
    Sum,
    And,
    Or,
    Not,
    sat,
    simplify,
)
from z3.z3 import BoolRef, ArithRef, IntNumRef

import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from ...core.client import Client
from ...core.callbacks import OpCallbacks
from ...core.data import (
    Op,
    RawLoad,
    Load,
    RawStore,
    Store,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    ProgramId,
    Dot,
    MakeRange,
    AddPtr,
    ExpandDims,
    Broadcast,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    Splat,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
    Idiv,
    Rsqrt,
    CastImpl,
    Reshape,
    Fabs,
    Ashr,
    Advance,
    FpToFp,
    Umulhi,
    Trans,
    CumSum,
)
from ..utils import (
    check_out_of_bounds_access,
    check_storage_contiguous,
    get_physical_addr_from_tensor_slice,
    check_inner_stride_equal_to_one,
)
from .data import (
    TracebackInfo,
    OutOfBoundsRecord,
    OutOfBoundsRecordBruteForce,
    OutOfBoundsRecordZ3,
)
from ...core import config as cfg


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
        print("Constraints:")
        for constraint in constraints:
            print(constraint)

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

    # Display two items per line to save vertical space
    # Calculate column widths for alignment
    col1_width = 12  # Width for first label
    col2_width = 25  # Width for first value
    col3_width = 12  # Width for second label

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

        if hasattr(oob_record, "constraints") and oob_record.constraints:
            print(
                f"  {green}SMT constraints ({len(oob_record.constraints)}):{reset_color}"
            )
            for i, constraint in enumerate(oob_record.constraints[:5]):  # Show first 5
                print(f"    [{i}] {constraint}")
            if len(oob_record.constraints) > 5:
                print(f"    ... and {len(oob_record.constraints) - 5} more constraints")

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


def _get_tensor(tensor_list, data_ptr):
    # From a give ptr, get where the original tensor is stored
    # Tensors have been sorted by ptr
    ret_idx = 0
    for i in range(len(tensor_list)):
        if data_ptr < tensor_list[i].data_ptr():
            break
        ret_idx = i
    return tensor_list[ret_idx]


@dataclass
class LoopContext:
    lineno: int
    length: int
    idx_z3: ArithRef
    values: list[int] = field(default_factory=list)
    signature_cache: set[int] = field(default_factory=set)
    pending_checks: list[
        tuple[Union[ArithRef, list[ArithRef]], list[BoolRef], Optional["SymbolicExpr"]]
    ] = field(default_factory=list)
    # Clean up variable names by removing suffixes like _81, _144
    # Cache compiled regex pattern for better performance
    re_pattern = re.compile(r"(loop_i|arange)_\d+")


class Sanitizer(Client):
    """
    Factory class that returns the concrete sanitizer implementation
    based on the value of ``cfg.sanitizer_backend``.
    """

    NAME = "sanitizer"

    def __new__(cls, abort_on_error: bool = False, *args, **kwargs):
        if cls is not Sanitizer:
            return super().__new__(cls)

        cfg.sanitizer_activated = True

        backend = cfg.sanitizer_backend

        if backend == "brute_force":
            return object.__new__(SanitizerBruteForce)

        if backend == "symexec":
            return object.__new__(SanitizerSymbolicExecution)

        if backend == "off":
            return object.__new__(NullSanitizer)

        raise ValueError(f"Invalid TRITON_SANITIZER_BACKEND: {backend!r} ")

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def arg_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def finalize(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def grid_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def grid_idx_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def register_op_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def register_for_loop_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError


def _get_last_grid(grid: tuple[int, ...]) -> tuple[int, int, int]:
    return (grid[0] - 1, grid[1] - 1, grid[2] - 1)


class SanitizerBruteForce(Sanitizer):
    def __init__(
        self,
        abort_on_error: bool = False,
        callpath: bool = True,
    ):
        self.callpath = callpath
        self.abort_on_error = abort_on_error
        self.tensors: list[Tensor] = []
        self.records: list = []
        self.grid_idx: Optional[tuple[int, ...]] = None
        self.last_grid: Optional[tuple[int, int, int]] = None

    def _report(self, op_type, record):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordBruteForce(
            op_type=op_type, user_code_tracebacks=traceback_info, **record
        )
        if self.abort_on_error:
            if np.any(oob_record.invalid_access_masks):
                print_oob_record(oob_record)
                assert (
                    False
                ), "Out-of-bounds access detected. See detailed report above."
        else:
            self.records.append(oob_record)

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        if self.grid_idx == self.last_grid:
            self.tensors.clear()
        return True

    def arg_callback(self, name, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            assert check_storage_contiguous(
                arg
            ), "The address sanitizer only supports contiguouly stored tensors for now"
            self.tensors.append(arg)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self.grid_idx = grid_idx

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self.last_grid = _get_last_grid(grid)
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, oob)
            ptr.data = tensor.data_ptr() + oob["corrected_offsets"]

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(
                op_type, check_out_of_bounds_access(ptr.data, mask.data, tensor)
            )
            ptr.data = tensor.data_ptr() + oob["corrected_offsets"]

        if op_type is Load:
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)

        return OpCallbacks()

    def register_for_loop_callback(self):
        return None, None, None, None

    def finalize(self) -> list:
        return self.records


class SymbolicExprDataWrapper:
    """
    This wrapper is used as a workaround of triton interpreter legacy code.
    In def _get_bool(self) of class tensor,
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Since we replaced TensorHandle with SymbolicExpr,
    we need to wrap SymbolicExpr with a class that has size attribute, and data.size != 1.
    """

    def __init__(self, value, symbolic_expr):
        self.value = value
        self.symbolic_expr = symbolic_expr

    @property
    def size(self):
        return 2

    def __int__(self):
        int_val, _ = self.symbolic_expr.eval()
        if isinstance(int_val, IntNumRef):
            return int_val.as_long()
        if isinstance(int_val, int):
            return int_val
        raise ValueError(
            f"SymbolicExprDataWrapper is type: {type(int_val)}, value: {int_val} and cannot be converted to int"
        )

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


Spec = namedtuple(
    "Spec",
    [
        "req",  # Required Arguments (tuple)
        "opt",  # Optional Arguments (tuple)
        "post",  # post-hook: def(self) -> None
    ],
    defaults=((), (), None),
)


def _load_post(self):
    self.dtype_tt = self.ptr.dtype_tt.element_ty


def _broadcast_dtype(self):
    self.dtype_tt = self.children["arg"].dtype_tt


def _binary_dtype(expr):
    expr.dtype_tt = expr.lhs.dtype_tt


def _binary_shape(expr):
    lhs, rhs = expr.children["lhs"], expr.children["rhs"]
    if not lhs.shape:
        expr.shape = rhs.shape
    elif not rhs.shape:
        expr.shape = lhs.shape
    else:
        assert lhs.shape == rhs.shape, f"lhs shape {lhs.shape} != rhs shape {rhs.shape}"
        expr.shape = lhs.shape


def _binary_post(expr):
    _binary_shape(expr)
    _binary_dtype(expr)


def _pid_post(expr):
    expr.dtype_tt = tl.int32  # Program ID is always int32


def _arange_post(expr):
    expr.dtype_tt = tl.int32  # tl.arange is always int32


def _cast_impl_post(expr):
    expr.dtype_tt = expr.dst_type


def _addptr_post(expr):
    expr.dtype_tt = expr.ptr.dtype_tt


class SymbolicExpr:
    BASIC_OPS = ("const", "pid", "arange")
    INDIRECT_OPS = ("load", "store")
    UNARY_OPS = (
        "cos",
        "exp",
        "exp2",
        "abs",
        "fabs",
        "floor",
        "ceil",
        "log",
        "log2",
        "sqrt",
        "sin",
        "rsqrt",
    )
    BINARY_OP_SYMBOL_TABLE = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "idiv": "//",
        "mod": "%",
        "less": "<",
        "less_equal": "<=",
        "greater": ">",
        "greater_equal": ">=",
        "not_equal": "!=",
        "equal": "==",
        "maximum": "max",
        "minimum": "min",
        "bitwise_and": "&",
        "right_shift": ">>",
        "left_shift": "<<",
        "ashr": ">>>",
    }
    BINARY_OPS = tuple(BINARY_OP_SYMBOL_TABLE.keys())
    TERNARY_OPS = ("where",)
    REDUCE_OPS = ("sum", "max", "min", "dot")
    SCAN_OPS = ("cumsum",)
    POINTER_OPS = ("make_block_ptr", "addptr", "advance")
    BROADCAST_OPS = ("splat", "expand_dims", "broadcast", "reshape")
    CAST_OPS = ("cast_impl",)
    SUPPORTED_OPS = (
        BASIC_OPS
        + INDIRECT_OPS
        + UNARY_OPS
        + BINARY_OPS
        + TERNARY_OPS
        + REDUCE_OPS
        + POINTER_OPS
        + BROADCAST_OPS
        + CAST_OPS
        + SCAN_OPS
    )

    OP_SPEC = {
        # Core / primitive ops
        "arange": Spec(req=("ret_ty", "start", "end"), post=_arange_post),
        "pid": Spec(req=("grid", "axis"), post=_pid_post),
        # Memory access ops
        "load": Spec(req=("ptr",), opt=("mask", "other"), post=_load_post),
        "store": Spec(req=("ptr", "value"), opt=("mask", "other")),
        # Unary ops
        **{
            op: Spec(req=("arg",))
            for op in (
                "cos",
                "exp",
                "exp2",
                "abs",
                "fabs",
                "floor",
                "ceil",
                "log",
                "log2",
                "sqrt",
                "sin",
                "rsqrt",
            )
        },
        #  Binary ops
        **{
            op: Spec(req=("lhs", "rhs"), post=_binary_post)
            for op in (
                "add",
                "sub",
                "mul",
                "div",
                "idiv",
                "mod",
                "less",
                "less_equal",
                "greater",
                "greater_equal",
                "not_equal",
                "equal",
                "maximum",
                "minimum",
                "bitwise_and",
                "right_shift",
                "left_shift",
                "ashr",
            )
        },
        # Ternary ops
        "where": Spec(req=("cond", "lhs", "rhs")),
        # Reduction ops
        "sum": Spec(req=("input", "axis", "keepdims")),
        "max": Spec(req=("input", "axis", "keepdims")),
        "min": Spec(req=("input", "axis", "keepdims")),
        # Scan Ops
        "cumsum": Spec(req=("input", "axis", "reverse", "dtype")),
        "dot": Spec(req=("a", "b"), opt=("d",)),
        # Pointer utilities
        "make_block_ptr": Spec(
            req=("base", "shape", "strides", "offsets", "block_shape", "order")
        ),
        "addptr": Spec(req=("ptr", "offset"), post=_addptr_post),
        # Broadcasting / shape manipulation
        "splat": Spec(req=("shape", "arg"), post=_broadcast_dtype),
        "expand_dims": Spec(req=("arg", "axis"), post=_broadcast_dtype),
        "broadcast": Spec(req=("arg", "shape"), post=_broadcast_dtype),
        "reshape": Spec(req=("arg", "shape"), post=_broadcast_dtype),
        # Casting
        "cast_impl": Spec(req=("src", "dst_type"), post=_cast_impl_post),
        # Misc
        "advance": Spec(req=("ptr", "offsets")),
        "umulhi": Spec(req=("lhs", "rhs")),
    }

    ARANGE_COUNTER = 0  # Used to name arange variables
    PID0 = Int("pid_0")
    PID1 = Int("pid_1")
    PID2 = Int("pid_2")

    def __init__(self, op, *args):
        """
        :param op: Operation type, e.g. "const", "add", "sub", "mul", "div", "pid", "arange"
        :param args: Sub-expressions (for compound operations)
        :param value: For "const" op, the constant value
        :param grid, axis: For "pid" op, the grid and axis
        """
        assert op in self.SUPPORTED_OPS, f"Unsupported op: {op}"
        self.op = op
        self.attrs = {}
        self.dtype_tt = None
        self.shape = []

        # Functions and arguments for concretization
        self._binary_numpy_op = None
        self._concrete_fn = None
        self._concrete_args = ()
        self._concrete_kwargs = {}

        # deal with args
        self.children = {}  # Used for storing child expressions
        if self.op == "const":  # leaf nodes
            self.value = args[0]
            if len(args) >= 2:
                self.dtype_tt = args[1]
        else:
            self._init_from_spec(*args)

        # for-loop iterator association
        self._loop_ctx: Optional[LoopContext] = None

        # z3
        self._z3 = None

        self._constraints: list[BoolRef] = []

    def _init_from_spec(self, *args: Any) -> None:
        if self.op not in self.OP_SPEC:
            raise NotImplementedError(f"Unsupported op: {self.op}")
        spec = self.OP_SPEC[self.op]

        # check if args number match the spec
        min_n = len(spec.req)
        max_n = min_n + len(spec.opt)
        if not (min_n <= len(args) <= max_n):
            raise ValueError(
                f"{self.op} expects {min_n} - {max_n} args, got {len(args)}"
            )

        # store in self.children
        names = list(spec.req) + list(spec.opt)
        for name, val in zip(names, args):
            val = SymbolicExpr.from_value(val)
            self.children[name] = val
        for name in names[len(args) :]:
            self.children[name] = None

        # post-hook
        if spec.post:
            spec.post(self)

    def __getattr__(self, name):
        if name in self.children:
            return self.children[name]
        # Special handling for dtype to provide compatibility
        if name == "dtype":
            return self.dtype_tt
        raise AttributeError(name)

    def set_attr(self, name, values):
        self.attrs[name] = values

    def __add__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("add", self, other)

    def __sub__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("sub", self, other)

    def __mul__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("mul", self, other)

    def __truediv__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("div", self, other)

    def __floordiv__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("idiv", self, other)

    def __mod__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("mod", self, other)

    def __lt__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("less", self, other)

    def __le__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("less_equal", self, other)

    def __ne__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("not_equal", self, other)

    def __eq__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("equal", self, other)

    def to_anytree(self):
        """Convert this SymbolicExpr into an anytree Node."""
        # Generate the node label
        label = self._node_label()
        root = Node(label)

        # Recursively add child nodes
        for child_key, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                Node(f"{child_key}: None", parent=root)
                continue
            child_node = child_symbolic_expr.to_anytree()
            child_node.name = f"{child_key}: {child_node.name}"
            child_node.parent = root

        return root

    def _node_label(self):
        """Generate a short label for this node."""
        if self.op == "const":
            label = f"const={self.value}"
        elif self.op == "pid":
            axis_val = self.axis.to_py()
            grid_val = self.grid.to_py()
            label = f"pid_{axis_val}={grid_val[axis_val]}"
        else:
            label = self.op

        # Add the dtype to the label if available
        label = f"{label} [dtype={self.dtype_tt}]"

        return label

    def to_tree_str(self) -> str:
        """
        Render the AST as an ASCII tree using anytree.RenderTree.
        """
        root = self.to_anytree()
        lines = []
        for prefix, _, node in RenderTree(root):
            lines.append(f"{prefix}{node.name}")
        return "\n" + "\n".join(lines)

    def __str__(self):
        return self.to_tree_str()

    def __repr__(self):
        return self.__str__()

    @property
    def data(self):
        return SymbolicExprDataWrapper(self.__str__(), self)

    triton_scala_dtypes = (
        tl.int8,
        tl.int16,
        tl.int32,
        tl.int64,
        tl.uint8,
        tl.uint16,
        tl.uint32,
        tl.uint64,
        tl.float16,
        tl.float32,
        tl.float64,
    )
    builtin_scala_types = (int, float)
    tuple_types = (tl.core.tuple, tuple, list)

    @staticmethod
    def _infer_literal_dtype(var):
        if isinstance(var, tl.core.dtype):
            return var
        if isinstance(var, tl.core.tensor):
            return var.dtype
        if isinstance(var, SymbolicExpr.tuple_types):
            first_dtype = SymbolicExpr._infer_literal_dtype(var[0])
            for v in var[1:]:  # assume only one consistent dtype in the tuple
                if SymbolicExpr._infer_literal_dtype(v) != first_dtype:
                    raise ValueError(
                        f"All elements in the tuple must have the same dtype, but found {first_dtype} and {SymbolicExpr.from_value(v).dtype_tt}"
                    )
            return first_dtype
        if isinstance(var, TensorHandle):
            if len(var.data) != 1:
                raise ValueError(
                    f"Unsupported var.data: {var.data} with length more than one!"
                )
            if var.dtype in SymbolicExpr.triton_scala_dtypes:  # if an immediate
                return var.dtype
            if isinstance(var.dtype, tl.pointer_type):  # if a pointer
                return var.dtype
        if isinstance(var, SymbolicExpr.builtin_scala_types):
            return tl.int32 if isinstance(var, int) else tl.float32
        if var is None:
            return tl.int32  # Default dtype for None
        raise ValueError(f"Unsupported type: {type(var)}")

    _loop_ctx_provider: Optional[Callable[[], Optional[LoopContext]]] = None

    @classmethod
    def set_loop_ctx_provider(cls, fn: Callable[[], Optional[LoopContext]]):
        """Register a function that, on each call,
        returns the current active `LoopContext` (or `None` if none exists)."""
        cls._loop_ctx_provider = fn

    @classmethod
    def from_value(cls, var):
        if isinstance(var, cls):  # if already SymbolicExpr
            return var

        dtype_tt = SymbolicExpr._infer_literal_dtype(var)  # get the triton dtype

        if isinstance(var, SymbolicExpr.tuple_types):  # if a tuple
            return cls("const", tuple(var), dtype_tt)
        if isinstance(var, TensorHandle):  # if a TensorHandle
            return cls("const", var.data.item(), dtype_tt)
        if isinstance(
            var, SymbolicExpr.builtin_scala_types
        ):  # if a python builtin type
            return cls("const", var, dtype_tt)
        if isinstance(var, tl.core.dtype):
            # If it's a triton dtype, we can create a const node with it
            return cls("const", var, dtype_tt)
        if var is None:
            # If it's None, we can create a const node with default dtype
            return cls("const", None, dtype_tt)

        raise ValueError("Unknown type:", type(var))

    def eval(self) -> tuple[Union[ArithRef, list[ArithRef]], list[BoolRef]]:
        """
        Returns a tuple (expr, constraints):
        - expr: Z3 expression corresponding to the root node
        - constraints: list of Z3 BoolExpr objects, recording all range constraints created by program_id and arange
        """
        expr, constraints = self._to_z3()

        if isinstance(expr, list):
            expr = [simplify(e).as_long() for e in expr]
        else:
            expr = simplify(expr)

        return expr, constraints

    def _to_z3(self) -> tuple[ArithRef, list]:
        if self._z3 is not None:
            return self._z3, self._constraints

        # Recursively convert the current node to a Z3 expression
        if self.op == "const":
            if self._loop_ctx:  # if the self is a loop iterator
                self._z3 = self._loop_ctx.idx_z3
            elif isinstance(self.value, np.ndarray):
                self._z3 = [IntVal(int(v)) for v in self.value.flat]
            else:
                self._z3 = IntVal(self.value)

        if self.op == "pid":
            axis_val = self.axis.to_py()
            if axis_val == 0:
                self._z3 = SymbolicExpr.PID0
            elif axis_val == 1:
                self._z3 = SymbolicExpr.PID1
            else:
                self._z3 = SymbolicExpr.PID2

        if self.op == "arange":
            # Get file name, line number, and column number

            idx = SymbolicExpr.ARANGE_COUNTER
            SymbolicExpr.ARANGE_COUNTER += 1
            name = f"arange_{idx}"
            v = Int(name)
            start = self.start.value
            end = self.end.value
            self._constraints.append(v >= start)
            self._constraints.append(v < end)
            self._z3 = v

        # Unary operations (only abs is demonstrated here; others can be added using z3.Function as needed)
        if self.op in self.UNARY_OPS:
            val, self._constraints = self.arg._to_z3()
            if self.op == "abs" or self.op == "fabs":
                self._z3 = If(val >= 0, val, -val)
            else:
                raise NotImplementedError(f"Unary op {self.op} is not implemented")

        # Binary arithmetic, comparison, etc.
        if self.op in self.BINARY_OPS:
            lhs, constraints_lhs = self.lhs._to_z3()
            rhs, constraints_rhs = self.rhs._to_z3()
            self._constraints.extend(constraints_lhs)
            self._constraints.extend(constraints_rhs)
            if self.op == "add":
                self._z3 = lhs + rhs
            if self.op == "sub":
                self._z3 = lhs - rhs
            if self.op == "mul":
                self._z3 = lhs * rhs
            if self.op in ("idiv"):
                self._z3 = lhs / rhs
            if self.op == "mod":
                self._z3 = lhs % rhs
            if self.op == "less":
                self._z3 = lhs < rhs
            if self.op == "less_equal":
                self._z3 = lhs <= rhs
            if self.op == "greater":
                self._z3 = lhs > rhs
            if self.op == "greater_equal":
                self._z3 = lhs >= rhs
            if self.op == "equal":
                self._z3 = lhs == rhs
            if self.op == "not_equal":
                self._z3 = lhs != rhs
            if self.op == "maximum":
                self._z3 = If(lhs >= rhs, lhs, rhs)
            if self.op == "minimum":
                self._z3 = If(lhs <= rhs, lhs, rhs)
            if self.op == "bitwise_and":
                self._z3 = And(lhs, rhs)
            if self.op == "ashr":
                raise NotImplementedError(
                    "Arithmetic shift right is not implemented in Z3"
                )

        # where(cond, lhs, rhs)
        if self.op == "where":
            cond, constraints_cond = self.cond._to_z3()
            lhs, constraints_lhs = self.lhs._to_z3()
            rhs, constraints_rhs = self.rhs._to_z3()
            self._z3 = If(cond, lhs, rhs)
            self._constraints.extend(
                constraints_cond + constraints_lhs + constraints_rhs
            )

        # sum(input, axis, keepdims)
        if self.op == "sum":
            arr, self._constraints = self.input._to_z3()
            self._z3 = Sum(arr)

        if self.op == "max":
            raise NotImplementedError("_to_z3 of max is not implemented yet")

        if self.op == "min":
            raise NotImplementedError("_to_z3 of min is not implemented yet")

        if self.op == "load" or self.op == "store":
            # Load and store operations
            ptr, constraints_ptr = self.ptr._to_z3()
            self._constraints.extend(constraints_ptr)
            if self.mask is not None:
                mask, _ = self.mask._to_z3()
                self._constraints.append(mask)
            self._z3 = ptr

        if self.op in ("splat", "expand_dims", "broadcast"):
            self._z3, self._constraints = self.arg._to_z3()

        if self.op == "addptr":
            # Add pointer operation
            ptr_z3, constraints_ptr = self.ptr._to_z3()
            offset_z3, constraints_offset = self.offset._to_z3()
            self._constraints = constraints_ptr + constraints_offset
            element_bytewidth = max(
                1, self.ptr.dtype_tt.element_ty.primitive_bitwidth // 8
            )
            if isinstance(ptr_z3, list) and isinstance(
                offset_z3, list
            ):  # both ptr and offset are lists
                if len(ptr_z3) != len(offset_z3):  # check if they have the same length
                    raise ValueError(
                        f"ptr {ptr_z3} and offset {offset_z3} don't have the same length!"
                    )
                self._z3 = [
                    p + o * element_bytewidth for p, o in zip(ptr_z3, offset_z3)
                ]
            if isinstance(ptr_z3, list) and isinstance(offset_z3, list):
                if len(ptr_z3) != len(offset_z3):
                    raise ValueError(
                        f"ptr {ptr_z3} and offset {offset_z3} don't have the same length!"
                    )
                self._z3 = [
                    p + o * element_bytewidth for p, o in zip(ptr_z3, offset_z3)
                ]
            elif isinstance(ptr_z3, list):
                self._z3 = [p + offset_z3 * element_bytewidth for p in ptr_z3]
            elif isinstance(offset_z3, list):
                self._z3 = [ptr_z3 + o * element_bytewidth for o in offset_z3]
            else:
                self._z3 = ptr_z3 + offset_z3 * element_bytewidth

        if self.op == "cast_impl":
            # Cast operation - pass through the source value
            self._z3, self._constraints = self.src._to_z3()

        if self.op == "advance":
            raise NotImplementedError("Advance operation is not implemented yet")

        if self._z3 is None:
            # Other operations can be implemented as needed
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")

        return self._z3, self._constraints

    def has_op(self, op_name: str) -> bool:
        if self.op == op_name:
            return True
        for _, child_symbolic_expr in self.children.items():
            if child_symbolic_expr.has_op(op_name):
                return True
        return False

    def to_py(self):
        """
        Valid only for nodes with op == 'const':
        - If `value` is a TensorHandle:
            • Scalar  -> return int/float
            • Multi-element -> return a Python list
        - Otherwise, return the original Python object
          (e.g., int, float, tuple, list, etc.).
        """
        if self.op != "const":
            raise TypeError("SymbolicExpr.to_py() can be used only on 'const' nodes")

        v = self.value
        if isinstance(v, TensorHandle):
            if len(v.data) == 1:
                return v.data.item()  # scalar case
            else:
                raise NotImplementedError(
                    "SymbolicExpr.to_py() for multi-element tensors is not implemented yet!"
                )
        return v

    _concrete_fn_cache: dict[str, Callable] = {}
    _binary_numpy_op_cache: dict[str, Callable] = {}

    @property
    def concrete_fn(self):
        """Return the concrete evaluation function bound to this node."""
        if self._concrete_fn is None and self.op in SymbolicExpr._concrete_fn_cache:
            self._concrete_fn = SymbolicExpr._concrete_fn_cache[self.op]
        return self._concrete_fn

    @concrete_fn.setter
    def concrete_fn(self, fn):
        """Bind / override the concrete evaluation function."""
        self._concrete_fn = fn
        SymbolicExpr._concrete_fn_cache[self.op] = fn

    @property
    def binary_numpy_op(self):
        """Return the numpy operation corresponding to this binary op."""
        if (
            self._binary_numpy_op is None
            and self.op in SymbolicExpr._binary_numpy_op_cache
        ):
            self._binary_numpy_op = SymbolicExpr._binary_numpy_op_cache[self.op]
        return self._binary_numpy_op

    @binary_numpy_op.setter
    def binary_numpy_op(self, op):
        """Bind / override the numpy operation for this binary op."""
        self._binary_numpy_op = op
        SymbolicExpr._binary_numpy_op_cache[self.op] = op

    def concretize(self_or_cls, obj=None):
        """
        Usage:
        1. expr.concretize()               — Evaluate this SymbolicExpr instance to its concrete value.
        2. SymbolicExpr.concretize(x)      — Recursively concretize any object x (which may contain nested SymbolicExprs).
        """
        if obj is None:  # expr.concretize()
            obj = self_or_cls
        elif isinstance(
            self_or_cls, SymbolicExpr
        ):  # expr.concretize(x), which is erroneous
            raise TypeError(
                "Either use expr.concretize() or SymbolicExpr.concretize(x)!"
            )

        # Non-SymbolicExpr -> return as-is
        if not isinstance(obj, SymbolicExpr):
            return obj

        # concretize logic
        if obj.concrete_fn is None:
            if obj.op == "const":
                if obj.dtype_tt == tl.pointer_type:
                    result = TensorHandle(
                        np.array([obj.value], dtype=np.uint64), obj.dtype_tt
                    )
                else:
                    result = TensorHandle(obj.value, obj.dtype_tt)
            else:
                raise RuntimeError(f"{obj.op}'s concrete function is not set!")
        elif obj.op == "pid":
            result = obj.concrete_fn(obj.axis.to_py())
        elif obj.op == "arange":
            result = obj.concrete_fn(
                obj.ret_ty.to_py(), obj.start.to_py(), obj.end.to_py()
            )
        elif obj.op == "splat":
            result = obj.concrete_fn(
                obj.children["shape"].to_py(),
                obj.arg.concretize(),
            )
        elif obj.op in SymbolicExpr.BINARY_OPS:
            # Special handling for idiv which doesn't take a third parameter
            if obj.op == "idiv":
                result = obj.concrete_fn(obj.lhs.concretize(), obj.rhs.concretize())
            else:
                result = obj.concrete_fn(
                    obj.lhs.concretize(), obj.rhs.concretize(), obj.binary_numpy_op
                )
        elif obj.op == "load":
            from ...core.patch import original_ops

            ptr_concrete = obj.ptr.concretize()
            # concretize mask
            # create an all-True mask if mask is None
            if obj.mask is None:
                mask_concrete = TensorHandle(
                    np.ones_like(ptr_concrete.data, dtype=bool), tl.int1
                )
            else:
                mask_concrete = obj.mask.concretize()

            # concretize the 'other' argument if it exists
            if obj.other is None:
                other_concrete = None
            else:
                other_concrete = obj.other.concretize()

            result = original_ops[Load](
                ptr_concrete,
                mask_concrete,
                other_concrete,
                None,  # cache_modifier
                None,  # eviction_policy
                None,  # is_volatile
            )
        else:
            concrete_args = [
                SymbolicExpr.concretize(v) for k, v in obj.children.items()
            ]
            result = obj.concrete_fn(*concrete_args)

        return result


def _replace_load_subtree(expr: SymbolicExpr) -> SymbolicExpr:
    """
    Post-order traversal that replaces *all* minimal `load` sub-trees
    with constant nodes produced by `concretize`.
    """
    if not isinstance(expr, SymbolicExpr):
        raise TypeError("replace_min_load_subtree expects a SymbolicExpr instance!")

    # check subtrees
    for name, child in list(expr.children.items()):
        if child is None:
            continue
        # replace subtree if load found
        expr.children[name] = _replace_load_subtree(child)
        expr.children[name]._z3 = None
        expr.children[name]._constraints.clear()

    # check self
    if expr.op == "load" and all(
        (child is None) or not child.has_op("load") for child in expr.children.values()
    ):
        concrete = expr.concretize()

        if not isinstance(concrete, TensorHandle):
            raise TypeError(f"Unexpected dtype: {type(concrete)}!")

        # inplace replace to "const" node
        expr.op = "const"
        expr.value = concrete.data
        expr.dtype_tt = concrete.dtype
        expr.children.clear()
        expr.concrete_fn = None

    return expr


def _make_signature(addr_expr, constraints, re_pattern) -> int:
    """
    Convert (addr, constraints) into a stable string signature.
    • addr_expr can be a single z3 expr or list[expr]
    • constraints is list[expr]
    """
    if isinstance(addr_expr, list):
        addr_repr = "|".join(sorted(e.sexpr() for e in addr_expr))
    else:
        addr_repr = addr_expr.sexpr()

    constr_repr = "|".join(sorted(c.sexpr() for c in constraints))

    addr_repr = re_pattern.sub(r"\1", addr_repr)
    constr_repr = re_pattern.sub(r"\1", constr_repr)
    return hash(addr_repr + "##" + constr_repr)


@dataclass(frozen=True)
class _FnSymbolicCache:
    fn: Callable
    grid: tuple[int, ...]
    args: tuple

    @cached_property
    def _hash(self):
        return hash((self.fn, self.grid, self.args))

    def __hash__(self):
        return self._hash


_fn_symbolic_cache_set: set[_FnSymbolicCache] = set()


class SanitizerSymbolicExecution(Sanitizer):
    def __init__(self, abort_on_error: bool = False):
        self.abort_on_error: bool = abort_on_error
        self.records: list[OutOfBoundsRecordZ3] = []
        self.grid: Optional[tuple[int, ...]] = None
        self.grid_idx: Optional[tuple[int, ...]] = None
        self.tensors: list[Tensor] = []
        self.tensor_addrs: list[tuple[Int, Int]] = []
        self.unique_load_store_id: int = 0
        self.need_full_grid: bool = False
        self.loop_stack: list[LoopContext] = []
        self.last_grid: Optional[tuple[int, int, int]] = None
        self.cache_args: list = []
        self.cache_grid: Optional[tuple[int, ...]] = None
        SymbolicExpr.set_loop_ctx_provider(
            lambda: self.loop_stack[-1] if self.loop_stack else None
        )

    def _check_range_satisfiable(
        self,
        access_addr: Union[int, list[int], ArithRef, list[ArithRef]],
        expr_constraints: list,
    ):
        if isinstance(access_addr, list):
            for addr in access_addr:
                self._check_range_satisfiable(addr, expr_constraints)
            return
        self._solver.push()
        self._solver.add(self._addr_sym == access_addr)
        self._solver.add(And(*expr_constraints))
        if self._solver.check() == sat:
            print("out of bound access detected!")
            if self.abort_on_error:
                raise ValueError("Out-of-bounds access detected!")
        self._solver.pop()

    def _check_range_satisfiable_with_expr(
        self,
        access_addr: Union[int, list[int], ArithRef, list[ArithRef]],
        expr_constraints: list,
        symbolic_expr: Optional[SymbolicExpr] = None,
    ):
        """Check if access is satisfiable and report with symbolic expression if OOB."""
        if isinstance(access_addr, list):
            for addr in access_addr:
                self._check_range_satisfiable_with_expr(
                    addr, expr_constraints, symbolic_expr
                )
            return
        self._solver.push()
        self._solver.add(self._addr_sym == access_addr)
        self._solver.add(And(*expr_constraints))
        if self._solver.check() == sat:
            # Get the model to find the violation address
            model = self._solver.model()
            violation_addr = model[self._addr_sym].as_long() if model else 0

            # Find the tensor that this address belongs to
            tensor = None
            if self.tensors:
                # Simple heuristic: use the first tensor for now
                # In practice, you'd want to match the address to the correct tensor
                tensor = self.tensors[0]

            # Determine operation type from symbolic expression
            op_type: type[Load] | type[Store] = Load  # Default
            if symbolic_expr and symbolic_expr.op == "store":
                op_type = Store

            # Report with symbolic expression
            self._report(op_type, tensor, violation_addr, symbolic_expr)
        self._solver.pop()

    def _handle_access_check(self, expr: SymbolicExpr):
        """
        Evaluate a memory access expression and either defer it (inside a loop)
        or check it immediately (outside a loop).

        Returns True if this access was identified as a duplicate within the
        current loop and should be skipped by the caller (early-return case).
        Otherwise returns False.
        """
        # check memory access using z3
        z3_addr, z3_constraints = expr.eval()
        if self.loop_stack:  # for-loop iterator association
            ctx = self.loop_stack[-1]
            # check if addr already appeared before in the for-loop
            signature = _make_signature(z3_addr, z3_constraints, ctx.re_pattern)
            if signature in ctx.signature_cache:  # if appeared before
                if cfg.verbose:
                    print("[Sanitizer]  ↪ skip duplicated addr in loop")
            else:  # new addr expr
                if cfg.verbose:
                    print(
                        "[Sanitizer]  ↪ new addr in for-loop, will check later",
                        z3_addr,
                        z3_constraints,
                    )
                ctx.signature_cache.add(signature)
                # Store the expression along with the z3 data for later checking
                ctx.pending_checks.append((z3_addr, z3_constraints, expr))
        else:  # non-loop case
            self._check_range_satisfiable_with_expr(z3_addr, z3_constraints, expr)

    def _report(self, op_type, tensor, violation_address, symbolic_expr=None):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=[],
            symbolic_expr=symbolic_expr,
        )
        if self.abort_on_error:
            # Use the new PDB-style print function if available
            if symbolic_expr is not None or cfg.verbose:
                print_oob_record_pdb_style(oob_record, symbolic_expr)
            else:
                print_oob_record(oob_record)
            raise ValueError(
                "Out-of-bounds access detected. See detailed report above."
            )
        else:
            self.records.append(oob_record)

    def _clear_cache(self):
        self.cache_args.clear()
        self.cache_grid = None

    def pre_run_callback(self, fn: Callable) -> bool:
        if self.cache_grid:
            # First time we launch this program, compute the hash
            fn_cache = _FnSymbolicCache(fn, self.cache_grid, tuple(self.cache_args))
            self._clear_cache()
            if fn_cache not in _fn_symbolic_cache_set:
                _fn_symbolic_cache_set.add(fn_cache)
                # Must continue to run the program at least once
                # We don't clear up tensors at this point
                return True
        # 2nd time we launch this program, depends on whether we need a full grid
        return self.need_full_grid

    def post_run_callback(self, fn: Callable) -> bool:
        if self.grid_idx == self.last_grid or not self.need_full_grid:
            self._clear_cache()
            self.tensors.clear()
            self.tensor_addrs.clear()
        return self.need_full_grid

    def arg_callback(self, name, arg, arg_cvt):
        if not hasattr(arg, "data_ptr"):
            if name not in ["num_warps", "num_stages", "maxnreg", "num_ctas"]:
                self.cache_args.append(arg)
            return
        if arg.is_contiguous() or check_storage_contiguous(arg):
            start = arg.data_ptr()
            end = arg.data_ptr() + (arg.numel() - 1) * arg.element_size()
            tensor_physical_addresses = [(start, end)]
        elif check_inner_stride_equal_to_one(arg):
            tensor_physical_addresses = get_physical_addr_from_tensor_slice(arg)
        else:
            raise ValueError(
                "The address sanitizer only supports contiguouly stored tensors for now!"
            )
        # To uniquely identify the metadata of a tensor
        self.cache_args.append((arg.shape, arg.stride(), arg.dtype))
        self.tensors.append(arg)
        self.tensor_addrs.extend(tensor_physical_addresses)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self.cache_grid = grid
        self.last_grid = _get_last_grid(grid)
        self.grid = tuple(int(g) for g in grid)
        addr = Int("addr")
        self._addr_ok = Or(*[And(addr >= s, addr <= e) for s, e in self.tensor_addrs])
        self._pid_ok = And(
            SymbolicExpr.PID0 < self.grid[0],
            SymbolicExpr.PID1 < self.grid[1],
            SymbolicExpr.PID2 < self.grid[2],
            SymbolicExpr.PID0 >= 0,
            SymbolicExpr.PID1 >= 0,
            SymbolicExpr.PID2 >= 0,
        )
        self._solver = Solver()
        self._solver.add(Not(self._addr_ok))
        self._solver.add(self._pid_ok)
        self._addr_sym = addr

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self.grid_idx = grid_idx

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def op_program_id_overrider(axis):
            assert self.grid, "Grid not initialized!"
            return SymbolicExpr("pid", self.grid, axis)

        def op_raw_load_overrider(ptr, cache_modifier, eviction_policy, is_volatile):
            return op_load_overrider(
                ptr, None, None, cache_modifier, eviction_policy, is_volatile
            )

        def op_load_overrider(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            # deal with indirect loads
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                self.need_full_grid = True
                ptr = _replace_load_subtree(ptr)

            # make sure ptr dtype is valid
            if isinstance(ptr, TensorHandle) and not isinstance(
                ptr.dtype, tl.pointer_type
            ):
                raise ValueError(f"Unsupported ptr dtype: {ptr.dtype}")
            ptr_sym = SymbolicExpr.from_value(ptr)

            if mask is None:
                ret = SymbolicExpr("load", ptr_sym)
            elif other is None:
                ret = SymbolicExpr("load", ptr_sym, mask)
            else:
                ret = SymbolicExpr("load", ptr_sym, mask, other)

            # check memory access using z3 (defer in loops or check immediately)
            self._handle_access_check(ret)
            return ret

        def op_raw_store_overrider(ptr, value, cache_modifier, eviction_policy):
            return op_store_overrider(ptr, value, None, cache_modifier, eviction_policy)

        def op_store_overrider(ptr, value, mask, cache_modifier, eviction_policy):
            # deal with indirect loads
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                self.need_full_grid = True
                ptr = _replace_load_subtree(ptr)

            # make sure ptr is a SymbolicExpr
            if isinstance(ptr, TensorHandle) and not isinstance(
                ptr.dtype, tl.pointer_type
            ):
                raise ValueError(f"Unsupported ptr dtype: {ptr.dtype}")
            ptr_sym = SymbolicExpr.from_value(ptr)

            value = SymbolicExpr.from_value(value)
            if mask is None:
                ret = SymbolicExpr("store", ptr_sym, value)
            else:
                ret = SymbolicExpr("store", ptr_sym, value, mask)

            # check memory access using z3 (defer in loops or check immediately)
            self._handle_access_check(ret)
            return ret

        def op_unary_op_overrider(arg, op):
            _unary_map = {
                np.cos: "cos",
                np.exp: "exp",
                np.exp2: "exp2",
                np.abs: "abs",
                np.floor: "floor",
                np.ceil: "ceil",
                np.log: "log",
                np.log2: "log2",
                np.sqrt: "sqrt",
                np.sin: "sin",
            }
            arg_sym = SymbolicExpr.from_value(arg)
            try:
                name = _unary_map[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported unary operation: {op} on {arg_sym}"
                )
            return SymbolicExpr(name, arg_sym)

        def op_binary_op_overrider(lhs, rhs, op):
            _binary_map = {
                np.add: lambda lhs, rhs: lhs + rhs,
                np.subtract: lambda lhs, rhs: lhs - rhs,
                np.multiply: lambda lhs, rhs: lhs * rhs,
                np.divide: lambda lhs, rhs: lhs / rhs,
                np.less: lambda lhs, rhs: lhs < rhs,
                np.less_equal: lambda lhs, rhs: lhs <= rhs,
                np.greater: lambda lhs, rhs: lhs > rhs,
                np.greater_equal: lambda lhs, rhs: lhs >= rhs,
                np.not_equal: lambda lhs, rhs: lhs != rhs,
                np.equal: lambda lhs, rhs: lhs == rhs,
                np.fmod: lambda lhs, rhs: lhs % rhs,
                np.maximum: lambda lhs, rhs: SymbolicExpr("maximum", lhs, rhs),
                np.minimum: lambda lhs, rhs: SymbolicExpr("minimum", lhs, rhs),
                np.bitwise_and: lambda lhs, rhs: SymbolicExpr("bitwise_and", lhs, rhs),
                np.right_shift: lambda lhs, rhs: SymbolicExpr("right_shift", lhs, rhs),
                np.left_shift: lambda lhs, rhs: SymbolicExpr("left_shift", lhs, rhs),
            }
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            try:
                func = _binary_map[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported binary operation: {op} between {lhs_sym} and {rhs_sym}"
                )
            result = func(lhs_sym, rhs_sym)
            result.binary_numpy_op = op  # Store the numpy operation for later use
            return result

        def op_ternary_op_overrider(lhs, rhs, other, op):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            other = SymbolicExpr.from_value(other)
            if op is np.where:
                return SymbolicExpr("where", lhs_sym, rhs_sym, other)
            else:
                raise NotImplementedError(
                    f"Unsupported ternary operation: {op} between {lhs_sym}, {rhs_sym} and {other}"
                )

        def op_addptr_overrider(ptr, offset):
            ptr_sym = SymbolicExpr.from_value(ptr)
            offset_sym = SymbolicExpr.from_value(offset)
            return SymbolicExpr("addptr", ptr_sym, offset_sym)

        def op_dot_overrider(a, b, d, input_precision, max_num_imprecise_acc):
            a_sym = SymbolicExpr.from_value(a)
            b_sym = SymbolicExpr.from_value(b)
            d_sym = SymbolicExpr.from_value(d)
            return SymbolicExpr("dot", a_sym, b_sym, d_sym)

        def op_make_range_overrider(ret_ty, start, end):
            return SymbolicExpr(
                "arange",
                SymbolicExpr.from_value(ret_ty),
                SymbolicExpr.from_value(start),
                SymbolicExpr.from_value(end),
            )

        def op_expand_dims_overrider(arg, axis):
            return SymbolicExpr("expand_dims", SymbolicExpr.from_value(arg), axis)

        def op_broadcast_overrider(arg, shape):
            return SymbolicExpr("broadcast", SymbolicExpr.from_value(arg), shape)

        def op_reduce_sum_overrider(input, axis=None, keep_dims=False, **kwargs):
            return SymbolicExpr("sum", SymbolicExpr.from_value(input), axis, keep_dims)

        def op_reduce_max_overrider(input, axis=None, keep_dims=False, **kwargs):
            return SymbolicExpr("max", SymbolicExpr.from_value(input), axis, keep_dims)

        def op_reduce_min_overrider(input, axis=None, keep_dims=False, **kwargs):
            return SymbolicExpr("min", SymbolicExpr.from_value(input), axis, keep_dims)

        def op_splat_overrider(shape, arg):
            return SymbolicExpr("splat", shape, SymbolicExpr.from_value(arg))

        def op_make_block_ptr_overrider(
            base, shape, strides, offsets, tensor_shape, order
        ):
            base = SymbolicExpr.from_value(base)
            assert (
                len(shape)
                == len(strides)
                == len(offsets)
                == len(tensor_shape)
                == len(order)
            ), f"Length of shape ({len(shape)}), strides ({len(strides)}), offsets ({len(offsets)}), tensor_shape ({len(tensor_shape)}) and order ({len(order)}) must be the same!"
            shape = [SymbolicExpr.from_value(shape_i) for shape_i in shape]
            strides = [SymbolicExpr.from_value(strides_i) for strides_i in strides]
            offsets = [SymbolicExpr.from_value(offset_i) for offset_i in offsets]
            tensor_shape = [
                SymbolicExpr.from_value(tensor_shape_i)
                for tensor_shape_i in tensor_shape
            ]
            order = [SymbolicExpr.from_value(order_i) for order_i in order]

            ret = SymbolicExpr(
                "make_block_ptr", base, shape, strides, offsets, tensor_shape, order
            )

            ret.dtype_tt = base.get_element_ty()

            return ret

        def op_tensor_pointer_load_overrider(
            ptr,
            boundary_check,
            padding_option,
            cache_modifier,
            eviction_policy,
            is_volatile,
        ):
            raise NotImplementedError("TensorPointerLoad is not supported yet.")

        def op_tensor_pointer_store_overrider(
            ptr, value, boundary_check, cache_modifier, eviction_policy
        ):
            raise NotImplementedError("TensorPointerStore is not supported yet.")

        def op_idiv_overrider(lhs, rhs):
            result = SymbolicExpr.from_value(lhs) // SymbolicExpr.from_value(rhs)
            result.binary_numpy_op = np.floor_divide
            return result

        def op_rsqrt_overrider(arg):
            return SymbolicExpr("rsqrt", SymbolicExpr.from_value(arg))

        def op_cast_impl_overrider(src, dst_type):
            return SymbolicExpr("cast_impl", src, dst_type)

        def op_reshape_overrider(arg, shape, allow_reorder):
            # For symbolic execution, we track the reshape operation
            # arg is the input tensor, shape is the new shape, allow_reorder is a flag
            arg_sym = SymbolicExpr.from_value(arg)
            shape_sym = SymbolicExpr.from_value(shape)
            # Create a reshape symbolic expression
            result = SymbolicExpr("reshape", arg_sym, shape_sym)
            # The dtype should be preserved from the input
            result.dtype_tt = arg_sym.dtype_tt if hasattr(arg_sym, "dtype_tt") else None
            return result

        def op_fabs_overrider(arg):
            arg_sym = SymbolicExpr.from_value(arg)
            return SymbolicExpr("fabs", arg_sym)

        def op_ashr_overrider(lhs, rhs):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            return SymbolicExpr("ashr", lhs_sym, rhs_sym)

        def op_advance_overrider(ptr, offsets):
            # Advance operation for block pointers
            ptr_sym = SymbolicExpr.from_value(ptr)
            offsets_sym = SymbolicExpr.from_value(offsets)
            return SymbolicExpr("advance", ptr_sym, offsets_sym)

        def op_fptofp_overrider(src, dst_type, rounding_mode):
            return SymbolicExpr(
                "fptofp", SymbolicExpr.from_value(src), dst_type, rounding_mode
            )

        def op_umulhi_overrider(lhs, rhs):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            return SymbolicExpr("umulhi", lhs_sym, rhs_sym)

        def op_trans_overrider(arg, perm):
            return SymbolicExpr("trans", SymbolicExpr.from_value(arg), perm)

        def op_cumsum_overrider(input, axis, reverse=False, dtype=None):
            return SymbolicExpr(
                "cumsum", SymbolicExpr.from_value(input), axis, reverse, dtype
            )

        OP_TYPE_TO_OVERRIDER: dict[type[Op], Callable] = {
            ProgramId: op_program_id_overrider,
            RawLoad: op_raw_load_overrider,
            Load: op_load_overrider,
            RawStore: op_raw_store_overrider,
            Store: op_store_overrider,
            UnaryOp: op_unary_op_overrider,
            BinaryOp: op_binary_op_overrider,
            TernaryOp: op_ternary_op_overrider,
            Dot: op_dot_overrider,
            MakeRange: op_make_range_overrider,
            AddPtr: op_addptr_overrider,
            ExpandDims: op_expand_dims_overrider,
            Broadcast: op_broadcast_overrider,
            ReduceSum: op_reduce_sum_overrider,
            ReduceMax: op_reduce_max_overrider,
            ReduceMin: op_reduce_min_overrider,
            Splat: op_splat_overrider,
            MakeBlockPointer: op_make_block_ptr_overrider,
            TensorPointerLoad: op_tensor_pointer_load_overrider,
            TensorPointerStore: op_tensor_pointer_store_overrider,
            Idiv: op_idiv_overrider,
            Rsqrt: op_rsqrt_overrider,
            CastImpl: op_cast_impl_overrider,
            Reshape: op_reshape_overrider,
            Fabs: op_fabs_overrider,
            Ashr: op_ashr_overrider,
            Advance: op_advance_overrider,
            FpToFp: op_fptofp_overrider,
            Umulhi: op_umulhi_overrider,
            Trans: op_trans_overrider,
            CumSum: op_cumsum_overrider,
        }

        if op_type in OP_TYPE_TO_OVERRIDER:
            return OpCallbacks(op_overrider=OP_TYPE_TO_OVERRIDER[op_type])
        else:
            return OpCallbacks()

    def register_for_loop_callback(self):
        def loop_hook_before(lineno, iterable):
            if not isinstance(iterable, range):
                if cfg.verbose:
                    print("not a for-loop, skipping for-loop iterator association.")
                return
            length = len(iterable)
            idx_z3 = Int(f"loop_i_{lineno}")
            self.loop_stack.append(LoopContext(lineno, length, idx_z3))
            if cfg.verbose:
                print(f"[Sanitizer] ▶ enter loop@{lineno}, len={length}")

        def loop_hook_iter_overrider(lineno, idx):
            # collect iterator values
            if self.loop_stack and self.loop_stack[-1].lineno == lineno:
                self.loop_stack[-1].values.append(idx)

            # Make a tagged SymbolicExpr
            sym = SymbolicExpr("const", idx, tl.int32)
            sym._loop_ctx = self.loop_stack[-1]

            return tl.core.tensor(sym, tl.int32)

        def loop_hook_iter_listener(lineno, idx):
            if cfg.verbose:
                print(f"[Sanitizer] ▶ loop@{lineno} idx={idx}")

        def loop_hook_after(lineno: int) -> None:
            ctx = self.loop_stack.pop()
            # add constraints for loop_i
            iterator_constraints: list[BoolRef] = []
            if ctx.values:
                iterator_constraints.append(
                    Or(*[ctx.idx_z3 == v for v in set(ctx.values)])
                )

            # execute pending checks
            for check_tuple in ctx.pending_checks:
                # Handle both old format (2-tuple) and new format (3-tuple)
                if len(check_tuple) == 2:
                    addr_expr, expr_constraints = check_tuple
                    symbolic_expr = None
                else:
                    addr_expr, expr_constraints, symbolic_expr = check_tuple

                if cfg.verbose:
                    print(
                        "[Sanitizer] ▶ checking:",
                        addr_expr,
                        f" with iterator constraints: {iterator_constraints} ",
                        f" and expression-related constraints: {expr_constraints} ",
                    )

                if symbolic_expr is not None:
                    self._check_range_satisfiable_with_expr(
                        addr_expr,
                        expr_constraints + iterator_constraints,
                        symbolic_expr,
                    )
                else:
                    self._check_range_satisfiable(
                        addr_expr, expr_constraints + iterator_constraints
                    )

            if cfg.verbose:
                print(
                    f"[Sanitizer] ▶ leave loop@{lineno} end. "
                    f"(checked {len(ctx.pending_checks)} unique addr patterns)"
                )

        return (
            loop_hook_before,
            loop_hook_iter_overrider,
            loop_hook_iter_listener,
            loop_hook_after,
        )

    def finalize(self) -> list:
        return []


class NullSanitizer(Sanitizer):
    """
    A do-nothing object returned when the sanitizer backend is 'off'.
    Any attribute access raises an explicit error so misuse is obvious.
    """

    def __init__(self, *args, **kwargs):
        pass

    def _disabled(self, method: str):
        raise RuntimeError(
            f"[NullSanitizer] '{method}' was called, "
            "but sanitizer backend is off; no functionality is available."
        )

    def arg_callback(self, *args, **kwargs):
        self._disabled("arg_callback")

    def finalize(self, *args, **kwargs):
        self._disabled("finalize")

    def grid_callback(self, *args, **kwargs):
        self._disabled("grid_callback")

    def grid_idx_callback(self, *args, **kwargs):
        self._disabled("grid_idx_callback")

    def register_op_callback(self, *args, **kwargs):
        self._disabled("register_op_callback")

    def register_for_loop_callback(self, *args, **kwargs):
        self._disabled("register_for_loop_callback")

    def __getattr__(self, name):
        self._disabled(name)
