import traceback
from typing import Tuple, Callable, Optional, Type
import numpy as np
from z3 import Solver, Int, And, Or, Not, sat
import triton
import triton.language as tl
from triton.runtime.interpreter import _get_np_dtype, TensorHandle

from ...core.client import Client
from ...core.data import Op, RawLoad, Load, RawStore, Store, BinaryOp, ProgramId, AddPtr, MakeRange, Splat, Idiv, CastImpl
from ..utils import check_out_of_bounds_access, check_storage_contiguous, get_physical_addr_from_tensor_slice, check_inner_stride_equal_to_one
from .data import TracebackInfo, OutOfBoundsRecord, OutOfBoundsRecordBruteForce, OutOfBoundsRecordZ3
from ...core.config import sanitizer_backend


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
    print(f"Tensor Info: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")
    print(f"Tensor base memory address: {tensor.data_ptr()}")
    print("Valid Access Range: [0, %d)" % (np.prod(tensor.shape) * tensor.element_size()))
    for traceback_info in oob_record.user_code_tracebacks:
        print(f"File: {traceback_info.filename}, Line: {traceback_info.lineno}, in {traceback_info.func_name}")
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
        raise NotImplementedError("Invalid OutOfBoundsRecord type: " + str(type(oob_record)))

    print("============================================================")
    print("            End of Out-Of-Bounds Record Details             ")
    print("============================================================")

def _get_traceback_info():
    """
    Why do both _grid_executor_call and _jit_function_call appear in the call stacks?

    1) Main kernel dispatch (kernel[grid](...)) triggers _grid_executor_call.
    2) Inlined @triton.jit functions trigger _jit_function_call.
    3) Some code sees only _grid_executor_call if no separate JIT function is present or patched.
    4) Complex kernels (e.g., fused_attention) may show both: outer dispatch and inner JIT calls.
    """
    oob_filename, oob_lineno, oob_func_name, oob_line_of_code = "", -1, "", ""

    stack_summary = traceback.extract_stack()

    # record the index of two key functions in core/patch.py
    jit_index = None
    grid_index = None

    user_code_tracebacks = []
    # scan the call stack
    for i, frame in enumerate(stack_summary):
        user_code_index = None
        if ('_jit_function_call' in frame.name
            and 'triton_viz/core/patch.py' in frame.filename):
            user_code_index = i + 2 # _grid_executor_call -> run_grid_loops -> user code
        elif ('_grid_executor_call' in frame.name
            and 'triton_viz/core/patch.py' in frame.filename):
            user_code_index = i + 2 # the same as above

        if user_code_index is not None:
            frame = stack_summary[user_code_index]
            oob_filename = frame.filename
            oob_lineno = frame.lineno
            oob_func_name = frame.name
            oob_line_of_code = frame.line
            traceback_info = TracebackInfo(
                filename=oob_filename,
                lineno=oob_lineno,
                func_name=oob_func_name,
                line_of_code=oob_line_of_code
            )
            user_code_tracebacks.append(traceback_info)

    return user_code_tracebacks

def _get_tensor(tensor_list, data_ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(tensor_list)):
            if data_ptr < tensor_list[i].data_ptr():
                break
            ret_idx = i
        return tensor_list[ret_idx]

class SanitizerBruteForce(Client):
    def __init__(self, callpath: Optional[bool] = True, abort_on_error: Optional[bool] = True):
        self.callpath = callpath
        self.abort_on_error = abort_on_error
        self.tensors: list = []
        self.records: list = []

    def _report(self, op_type, record):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordBruteForce(op_type=op_type, user_code_tracebacks=traceback_info, **record)
        if self.abort_on_error:
            if np.any(oob_record.invalid_access_masks):
                print_oob_record(oob_record)
                assert False, "Out-of-bounds access detected. See detailed report above."
        else:
            self.records.append(oob_record)

    def arg_callback(self, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            assert check_storage_contiguous(
                arg
            ), "The address sanitizer only supports contiguouly stored tensors for now"
            self.tensors.append(arg)

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    def grid_callback(self, grid: Tuple[int]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: Type[Op]) -> Tuple[Optional[Callable], Optional[Callable]]:
        def pre_load_callback(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, oob)
            ptr.data = tensor.data_ptr() + oob['corrected_offsets']

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, check_out_of_bounds_access(ptr.data, mask.data, tensor))
            ptr.data = tensor.data_ptr() + oob['corrected_offsets']

        if op_type is Load:
            return pre_load_callback, None, None
        elif op_type is Store:
            return pre_store_callback, None, None

        return None, None, None

    def finalize(self) -> list:
        return self.records


class SanitizerZ3(Client):
    '''
    Fake Execution Sanitizer. This client is used to detect out-of-bound memory accesses.
    '''
    def __init__(self, abort_on_error):
        self.abort_on_error = abort_on_error
        self.tensors: list = []
        # constraints definition
        self.constraints = []
        self.is_combined_constraint_valid = False
        self.combined_constraint = None

    def _update_constraints(self, new_constraint):
        self.constraints.append(new_constraint)
        self.is_combined_constraint_valid = False

    def _print_constraints(self):
        print('Constraints:')
        for constraint in self.constraints:
            print(constraint)

    def _report(self, op_type, tensor, violation_address):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=self.constraints,
        )
        if self.abort_on_error:
            print_oob_record(oob_record)
            raise ValueError("Out-of-bounds access detected. See detailed report above.")
        else:
            self.records.append(oob_record)

    def _check_if_range_statisfy_constraints(self, start, end, op_type):
        # create a cache for the combined constraint
        if not self.is_combined_constraint_valid:
            self.combined_constraint = Or(*self.constraints)
            self.is_combined_constraint_valid = True

        # create an interval constraint
        x = Int('x')
        lowerbound_constraint = x >= start
        upperbound_constraint = x <= end
        interval_constraint = And(lowerbound_constraint, upperbound_constraint)

        # if the interval does not satisfy the constraints
        # then we have an out-of-bound memory access
        s = Solver()
        s.add(Not(self.combined_constraint))
        s.add(interval_constraint)

        # if found out-of-bound access, report it
        if s.check() == sat:
            tensor = _get_tensor(self.tensors, start)
            self._report(op_type, tensor, s.model()[x])

    def arg_callback(self, arg, arg_cvt):
        if not hasattr(arg, "data_ptr"):
            return
        assert check_storage_contiguous(arg), "The address sanitizer only supports contiguouly stored tensors for now"
        self.tensors.append(arg)

        nbytes = arg.element_size() * arg.numel()

        # add constraints
        x = Int('x')
        lowerbound_constraint = x >= arg.data_ptr()
        upperbound_constraint = x <= arg.data_ptr() + nbytes - arg.element_size()
        self._update_constraints(And(lowerbound_constraint, upperbound_constraint))

    def grid_callback(self, grid: Tuple[int]):
        pass

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    def register_op_callback(self, op_type: Type[Op]) -> Tuple[Optional[Callable], Optional[Callable]]:
        def pre_load_callback(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
            base_array = np.reshape(ptr.data, (-1))
            mask_array = np.reshape(mask.data, (-1))

            valid_addresses = base_array[mask_array]
            if len(valid_addresses) == 0:
                return
            lower_bound = valid_addresses.min()
            upper_bound = valid_addresses.max()
            self._check_if_range_statisfy_constraints(lower_bound, upper_bound, op_type)

        def op_load_overrider(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
            dtype_tt = ptr.get_element_ty()
            dtype_np = _get_np_dtype(dtype_tt)
            return TensorHandle(np.zeros_like(ptr.data, dtype=dtype_np), dtype_tt)

        def op_store_overrider(ptr, value, mask, cache_modifier, eviction_policy):
            pass

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            base_array = np.reshape(ptr.data, (-1))
            mask_array = np.reshape(mask.data, (-1))

            valid_addresses = base_array[mask_array]
            if len(valid_addresses) == 0:
                return
            lower_bound = valid_addresses.min()
            upper_bound = valid_addresses.max()
            self._check_if_range_statisfy_constraints(lower_bound, upper_bound, op_type)

        if op_type is Load:
            return pre_load_callback, None, op_load_overrider
        elif op_type is Store:
            return pre_store_callback, None, op_store_overrider
        else:
            return None, None, None

    def finalize(self) -> list:
        return []

class SymbolicExprDataWrapper:
    '''
    This wrapper is used as a workaround of triton interpreter legacy code.
    In def _get_bool(self) of class tensor,
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Since we replaced TensorHandle with SymbolicExpr,
    we need to wrap SymbolicExpr with a class that has size attribute, and data.size != 1.
    '''
    def __init__(self, value):
        self.value = value

    @property
    def size(self):
        return 2

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class SymbolicExpr:
    BASIC_OPS = ("const", "pid", "arange")
    INDIRECT_OPS = ("load", "store")
    BINARY_OP_SYMBOL_TABLE = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "idiv": "//",
        "mod": "%",
        "less": "<",
        "less_equal": "<=",
        "not_equal": "!=",
    }
    BINARY_OPS = BINARY_OP_SYMBOL_TABLE.keys()
    OP_SYMBOL_TABLE = BINARY_OP_SYMBOL_TABLE
    OP_ARGS_TABLE = {
        "load": ["ptr", "mask", "other"],
    }
    SUPPORTED_OPS = BASIC_OPS + INDIRECT_OPS + tuple(OP_SYMBOL_TABLE.keys())
    def __init__(self, op, *args):
        """
        :param op: Operation type, e.g. "const", "add", "sub", "mul", "div", "pid", "arange"
        :param args: Sub-expressions (for compound operations)
        :param value: For "const" op, the constant value
        :param grid, axis: For "pid" op, the grid and axis
        """
        assert op in self.SUPPORTED_OPS, f"Unsupported op: {op}"
        self.op = op
        # check if the number of arguments is correct
        if self.op == "const":
            assert len(args) == 1, "const op expects one argument!"
            self.value = args[0]
        elif self.op == "pid":
            assert len(args) == 2, "pid op expects two arguments!"
            self.grid = args[0]
            self.axis = args[1]
        elif self.op == "arange":
            assert len(args) == 2, "arange op expects two arguments!"
            self.start = args[0]
            self.end = args[1]
        elif self.op == "load":
            assert len(args) in (1, 2, 3), "load op expects 1, 2 or 3 arguments!"
            self.ptr = args[0]
            self.mask = args[1] if len(args) >= 2 else None
            self.other = args[2] if len(args) >= 3 else None
        elif self.op == "store":
            assert len(args) in (2, 3, 4), "store op expects 2, 3 or 4 arguments!"
            self.ptr = args[0]
            self.value = args[1]
            self.mask = args[2] if len(args) >= 3 else None
            self.other = args[3] if len(args) >= 4 else None
        elif self.op in self.BINARY_OP_SYMBOL_TABLE.keys():
            assert len(args) == 2, f"{self.op} op expects two arguments!"
            self.lhs = args[0]
            self.rhs = args[1]

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

    def to_tree_str(self, indent: int = 0) -> str:
        """Visualize AST using Tree format."""
        indent_str = "  "
        prefix = indent_str * indent

        if self.op == "const":
            s = f"{prefix}const: {self.value}, {type(self.value)}"
        elif self.op == "pid":
            s = f"{prefix}pid_{self.axis}: {self.grid[self.axis]}"
        elif self.op == "arange":
            # For "arange" node, we have two child nodes: start and end
            s = f"{prefix}arange:"
            s += "\n" + self.start.to_tree_str(indent + 1)
            s += "\n" + self.end.to_tree_str(indent + 1)
        elif self.op == "load":
            s = f"{prefix}load:"
            s += f"\n{indent_str}{prefix}ptr:\n" + self.ptr.to_tree_str(indent + 2)
            if self.mask is not None:
                s += f"\n{indent_str}{prefix}mask:\n" + self.mask.to_tree_str(indent + 2)
            if self.other is not None:
                s += f"{indent_str}{prefix}other:\n" + self.other.to_tree_str(indent + 2)
        elif self.op == "store":
            s = f"{prefix}store:"
            s += f"\n{indent_str}{prefix}ptr:\n" + self.ptr.to_tree_str(indent + 2)
            s += f"\n{indent_str}{prefix}value:\n" + self.value.to_tree_str(indent + 2)
            if self.mask is not None:
                s += f"\n{indent_str}{prefix}mask:\n" + self.mask.to_tree_str(indent + 2)
            if self.other is not None:
                s += f"{indent_str}{prefix}other:\n" + self.other.to_tree_str(indent + 2)
        elif self.op in self.BINARY_OPS:
            op_symbol = self.BINARY_OP_SYMBOL_TABLE[self.op]
            s = f"{prefix}{op_symbol}"
            # Call recursively for each operand
            for arg in (self.lhs, self.rhs):
                s += "\n" + arg.to_tree_str(indent + 1)
        else:
            raise ValueError(f"Unsupported op: {self.op}")
        return s

    def __str__(self):
        return self.to_tree_str()

    def __repr__(self):
        return self.__str__()

    @property
    def data(self):
        return SymbolicExprDataWrapper(self.__str__())

    @classmethod
    def from_value(cls, var):
        triton_scala_dtypes = (
            tl.int8, tl.int16, tl.int32, tl.int64,
            tl.uint8, tl.uint16, tl.uint32, tl.uint64,
            tl.float16, tl.float32, tl.float64
        )
        builtin_scala_types = (
            int, float
        )
        # if already symbolic
        if isinstance(var, cls):
            return var

        # if construct from a TensorHandle
        if isinstance(var, TensorHandle):
            # if an immediate
            if var.dtype in triton_scala_dtypes:
                if len(var.data) == 1:
                    return cls("const", var.data.item())
                return cls("const", var.data)
            # if a pointer
            elif isinstance(var.dtype, tl.pointer_type):
                if len(var.data) != 1:
                    raise ValueError("Unsupported tl.pointer_type with length more than one!")
                return cls("const", var.data.item())
            else:
                raise ValueError("Unsupported TensorHandle dtype", var.dtype)

        if isinstance(var, builtin_scala_types):
            return cls("const", var)

        raise ValueError("Unknown type:", type(var))

    def eval(self):
        # ================== Helper Function ==================
        # broadcasting val to a list of length n
        def to_list(val, n):
            if isinstance(val, list):
                if len(val) == n:
                    return val
                elif len(val) == 1:
                    return val * n
                else:
                    raise ValueError(f"Broadcast failed. Expected len(val): 1. Actual len(val): {len(val)}.")
            else:
                return [val] * n

        # judge if a range is a singleton, i.e. low == high
        def is_singleton(interval):
            return interval[0] == interval[1]

        def check_mask_bitmap_monotonous(mask_bitmap, flag):
            # mask bitmaps are like: [True, True, ... True, False, False, ... False]
            # The first piece is all True, and the second piece is all False.
            flipped = False
            for bit in mask_bitmap:
                if flag == bit:
                    continue
                elif not flipped:
                    flag = not flag
                    flipped = True
                else:
                    return False
            return True

        def apply_mask_to_interval(interval, mask_bitmap, true_then_false):
            # check if mask is either
            # [True, True, ... True, False, False, ... False] or
            # [False, False, ... False, True, True, ... True]
            assert check_mask_bitmap_monotonous(mask_bitmap, true_then_false), "Mask bitmap is not monotonous!"
            assert (interval[1] - interval[0]) % (len(mask_bitmap) - 1) == 0, "interval diff must be a multiple of mask bitmap length!"
            word_bytes = (interval[1] - interval[0]) // (len(mask_bitmap) - 1)
            if true_then_false:
                # mask will truncate the end of interval
                return (interval[0], interval[0] + word_bytes * (sum(mask_bitmap) - 1))
            else:
                # mask will truncate the start of interval
                return (interval[1] - word_bytes * (sum(mask_bitmap) - 1), interval[1])

        # ================== Binary Ops Evaluation =======================
        def compute_binary_op(lhs, rhs, op):
            if not is_singleton(lhs) and not is_singleton(rhs):
                raise NotImplementedError("Binary operation does not support two intervals yet.")
            if op == "add":
                # both are singletons
                if is_singleton(lhs) and is_singleton(rhs):
                    lhs = lhs[0]
                    rhs = rhs[0]
                    return (lhs + rhs, lhs + rhs)
                # only lhs is singleton
                elif is_singleton(lhs):
                    lhs = lhs[0]
                    return (lhs + rhs[0], lhs + rhs[1])
                # only rhs is singleton
                elif is_singleton(rhs):
                    rhs = rhs[0]
                    return (lhs[0] + rhs, lhs[1] + rhs)
                # both are intervals
                else:
                    raise NotImplementedError("Add operation does not support two intervals yet.")
            elif op == "sub":
                # both are singletons
                if is_singleton(lhs) and is_singleton(rhs):
                    lhs = lhs[0]
                    rhs = rhs[0]
                    return (lhs - rhs, lhs - rhs)
                # only lhs is singleton
                elif is_singleton(lhs):
                    lhs = lhs[0]
                    return (lhs - rhs[0], lhs - rhs[1])
                # only rhs is singleton
                elif is_singleton(rhs):
                    rhs = rhs[0]
                    return (lhs[0] - rhs, lhs[1] - rhs)
                # both are intervals
                else:
                    raise NotImplementedError("Sub operation does not support two intervals yet.")
            elif op == "mul":
                # both are singletons
                if is_singleton(lhs) and is_singleton(rhs):
                    lhs = lhs[0]
                    rhs = rhs[0]
                    return (lhs * rhs, lhs * rhs)
                # only lhs is singleton
                elif is_singleton(lhs):
                    lhs = lhs[0]
                    return (lhs * rhs[0], lhs * rhs[1])
                # only rhs is singleton
                elif is_singleton(rhs):
                    rhs = rhs[0]
                    return (lhs[0] * rhs, lhs[1] * rhs)
                # both are intervals
                else:
                    raise NotImplementedError("Mul operation does not support two intervals yet.")
            elif op == 'idiv':
                if not is_singleton(rhs):
                    raise NotImplementedError("IDiv operation does not support interval divisor yet.")
                return (lhs[0] // rhs[0], lhs[1] // rhs[0])
            elif op == 'mod':
                if not is_singleton(rhs):
                    raise NotImplementedError("Mod operation does not support interval divisor yet.")
                if not is_singleton(lhs):
                    raise NotImplementedError("Mod operation does not support interval dividend yet.")
                return (lhs[0] % rhs[0], lhs[0] % rhs[0])
            elif op == 'less':
                assert is_singleton(rhs), "rhs must be a singleton for less operation."
                rhs = rhs[0]
                if rhs <= lhs[0]:
                    return (False, False)
                elif rhs > lhs[1]:
                    return (True, True)
                else:
                    mask = tuple(i < rhs for i in range(lhs[0], lhs[1] + 1))
                    return (False, True, mask)
            else:
                raise NotImplementedError(f"Unsupported binary operation: {op}")

        # ================== Evaluation =======================
        if self.op == 'const':
            return (self.value, self.value)
        elif self.op == "pid":
            # expand into a list of (val, val)
            assert self.grid, "Grid not initialized!"
            assert self.axis < len(self.grid), f"axis {self.axis} not found in grid!"
            return [(v, v) for v in range(self.grid[self.axis])]
        elif self.op == "arange":
            start_val, end_val = self.start.eval(), self.end.eval()
            if isinstance(start_val, list):
                raise NotImplementedError("Arange does not support start index with list yet.")
            elif isinstance(start_val, tuple):
                if not is_singleton(start_val):
                    raise NotImplementedError("Arange does not support start index with interval yet.")
                start_val = start_val[0]
            if isinstance(end_val, list):
                raise NotImplementedError("Arange does not support end index with list yet.")
            elif isinstance(end_val, tuple):
                if not is_singleton(end_val):
                    raise NotImplementedError("Arange does not support end index with interval yet.")
                end_val = end_val[0]
            return (start_val, end_val)
        elif self.op in self.BINARY_OPS:
            lhs = self.lhs.eval()
            rhs = self.rhs.eval()
            if isinstance(lhs, list) or isinstance(rhs, list):
                n = len(lhs) if isinstance(lhs, list) else len(rhs)
                lhs = to_list(lhs, n)
                rhs = to_list(rhs, n)
                result = []
                for l, r in zip(lhs, rhs):
                    result.append(compute_binary_op(l, r, self.op))
                return result
            else:
                return compute_binary_op(lhs, rhs, self.op)
        elif self.op == "load" or self.op == "store":
            addrs = self.ptr.eval()
            if not isinstance(addrs, list):
                addrs = [addrs]
            if self.mask:
                mask_values = self.mask.eval()
                assert len(addrs) == len(mask_values), "length of addrs and mask must be the same!"
            else:
                mask_values = [(True, True) for _ in range(len(addrs))]

            masked_addrs = []
            for mask_value, addr in zip(mask_values, addrs):
                if mask_value == (True, True):
                    masked_addrs.append(addr)
                elif mask_value == (False, False):
                    continue
                elif mask_value[:2] == (False, True):
                    masked_addrs.append(apply_mask_to_interval(addr, mask_value[-1], True))
                else:
                    raise ValueError(f"Unsupported mask value: {mask_value}")
            if self.op == "load":
                print("tl.load:", masked_addrs)
            else:
                print("tl.store:", masked_addrs)
            return masked_addrs
        else:
            raise NotImplementedError(f"Unsupported operation: {self.op}")

class SanitizerSymbolicExecution(Client):
    def __init__(self, abort_on_error):
        self.abort_on_error = abort_on_error
        self.grid = None
        self.tensors = []
        self.constraints = []
        self.cached_constraint_valid = False
        self.cached_constraint = None

    def _create_range_constraint(self, addr):
        low, high = addr
        x = Int('x')
        low_constraint = x >= low
        high_constraint = x <= high
        return And(low_constraint, high_constraint)

    def _add_constraints(self, addr):
        self.constraints.append(self._create_range_constraint(addr))
        self.cached_constraint_valid = False

    def _check_range_satisfiable(self, addr, op_type):
        # maintain a cache for the union of all constraints
        if not self.cached_constraint_valid:
            self.cached_constraint = Or(*self.constraints)
            self.cached_constraint_valid = True

        constraint_to_check = self._create_range_constraint(addr)

        # Use Z3 Solver to find constraint conflicts
        s = Solver()
        s.add(Not(self.cached_constraint))
        s.add(constraint_to_check)

        # report out-of-bound access
        if s.check() == sat:
            tensor = _get_tensor(self.tensors, addr[0])
            violation_addr = s.model()[Int('x')]
            self._report(op_type, tensor, violation_addr)

    def _report(self, op_type, tensor, violation_address):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=self.constraints,
        )
        if self.abort_on_error:
            print_oob_record(oob_record)
            raise ValueError("Out-of-bounds access detected. See detailed report above.")
        else:
            self.records.append(oob_record)

    def arg_callback(self, arg, arg_cvt):
        if not hasattr(arg, "data_ptr"):
            return
        if arg.is_contiguous or check_storage_contiguous(arg):
            start = arg.data_ptr()
            end = arg.data_ptr() + (arg.numel() - 1) * arg.element_size()
            tensor_physical_addresses = [(start, end)]
        elif check_inner_stride_equal_to_one(arg):
            tensor_physical_addresses = get_physical_addr_from_tensor_slice(arg)
        else:
            raise ValueError("The address sanitizer only supports contiguouly stored tensors for now!")

        self.tensors.append(arg)

        for tensor_addr in tensor_physical_addresses:
            self._add_constraints(tensor_addr)

    def grid_callback(self, grid: Tuple[int]):
        self.grid = grid

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    def register_op_callback(self, op_type: Type[Op]) -> Tuple[Optional[Callable], Optional[Callable]]:
        def op_program_id_overrider(axis):
            assert self.grid, "Grid not initialized!"
            return SymbolicExpr("pid", self.grid, axis)

        def op_raw_load_overrider(ptr, cache_modifier, eviction_policy, is_volatile):
            return op_load_overrider(ptr, None, None, cache_modifier, eviction_policy, is_volatile)

        def op_load_overrider(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
            # make sure ptr is a SymbolicExpr
            if isinstance(ptr, TensorHandle) and isinstance(ptr.dtype, tl.pointer_type):
                ptr = SymbolicExpr("load", SymbolicExpr.from_value(ptr))
            elif isinstance(ptr, SymbolicExpr):
                ptr = SymbolicExpr("load", ptr)
            else:
                raise ValueError(f"Unsupported ptr type: {type(ptr)}")

            if mask is None:
                ret = SymbolicExpr("load", ptr)
            elif other is None:
                ret = SymbolicExpr("load", ptr, mask)
            else:
                ret = SymbolicExpr("load", ptr, mask, other)

            # check memory access using z3
            for mem_access_addr in ret.eval():
                self._check_range_satisfiable(mem_access_addr, Load)

            return ret

        def op_raw_store_overrider(ptr, value, cache_modifier, eviction_policy):
            return op_store_overrider(ptr, value, None, cache_modifier, eviction_policy)

        def op_store_overrider(ptr, value, mask, cache_modifier, eviction_policy):
            # make sure ptr is a SymbolicExpr
            if isinstance(ptr, TensorHandle) and isinstance(ptr.dtype, tl.pointer_type):
                ptr = SymbolicExpr("load", SymbolicExpr.from_value(ptr))
            elif isinstance(ptr, SymbolicExpr):
                ptr = SymbolicExpr("load", ptr)
            else:
                raise ValueError(f"Unsupported ptr type: {type(ptr)}")

            value = SymbolicExpr.from_value(value)
            if mask is None:
                ret = SymbolicExpr("store", ptr, value)
            else:
                ret = SymbolicExpr("store", ptr, value, mask)
            print('storing:', ret.eval())
            for mem_access_addr in ret.eval():
                self._check_range_satisfiable(mem_access_addr, Store)

        def op_binary_op_overrider(lhs, rhs, op):
            lhs = SymbolicExpr.from_value(lhs)
            rhs = SymbolicExpr.from_value(rhs)
            if op is np.add:
                return lhs + rhs
            elif op is np.subtract:
                return lhs - rhs
            elif op is np.multiply:
                return lhs * rhs
            elif op is np.divide:
                return lhs / rhs
            elif op is np.less:
                return lhs < rhs
            elif op is np.less_equal:
                return lhs <= rhs
            elif op is np.not_equal:
                return lhs != rhs
            elif op is np.fmod:
                return lhs % rhs
            else:
                raise NotImplementedError(f"Unsupported binary operation: {op} between {lhs} and {rhs}")

        def op_addptr_overrider(ptr, offset):
            dtype_tt = ptr.get_element_ty()
            element_bitwidth = dtype_tt.primitive_bitwidth
            element_bytewidth = max(1, element_bitwidth // 8)
            ptr = SymbolicExpr.from_value(ptr)
            offset = SymbolicExpr.from_value(offset)
            element_bytewidth = SymbolicExpr.from_value(element_bytewidth)
            return ptr + offset * element_bytewidth

        def op_make_range_overrider(start, end):
            start = SymbolicExpr.from_value(start)
            end = SymbolicExpr.from_value(end - 1)
            return SymbolicExpr("arange", start, end)

        def op_splat_overrider(arg, shape):
            return arg

        def op_idiv_overrider(lhs, rhs):
            lhs = SymbolicExpr.from_value(lhs)
            rhs = SymbolicExpr.from_value(rhs)
            return lhs // rhs

        def op_cast_impl_overrider(src, dst_type):
            src = SymbolicExpr.from_value(src)
            return src

        if op_type is ProgramId:
            return None, None, op_program_id_overrider
        elif op_type is RawLoad:
            return None, None, op_raw_load_overrider
        elif op_type is Load:
            return None, None, op_load_overrider
        elif op_type is RawStore:
            return None, None, op_raw_store_overrider
        elif op_type is Store:
            return None, None, op_store_overrider
        elif op_type is BinaryOp:
            return None, None, op_binary_op_overrider
        elif op_type is AddPtr:
            return None, None, op_addptr_overrider
        elif op_type is MakeRange:
            return None, None, op_make_range_overrider
        elif op_type is Splat:
            return None, None, op_splat_overrider
        elif op_type is Idiv:
            return None, None, op_idiv_overrider
        elif op_type is CastImpl:
            return None, None, op_cast_impl_overrider
        else:
            return None, None, None

    def finalize(self) -> list:
        return []


def Sanitizer(abort_on_error=False):
    if sanitizer_backend == "brute_force":
        return SanitizerBruteForce(abort_on_error)
    elif sanitizer_backend == "z3":
        return SanitizerZ3(abort_on_error)
    elif sanitizer_backend == "symexec":
        return SanitizerSymbolicExecution(abort_on_error)
    elif sanitizer_backend == "off":
        return None
    else:
        raise ValueError(f"Invalid TRITON_SANITIZER_BACKEND: {sanitizer_backend}")
