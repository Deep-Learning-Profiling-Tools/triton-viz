import traceback
from typing import Tuple, Callable, Optional, Type
import numpy as np
from z3 import Solver, Int, And, Or, Not, sat
import triton
import triton.language as tl
from triton.runtime.interpreter import _get_np_dtype, TensorHandle

from ...core.client import Client
from ...core.data import Op, RawLoad, Load, Store, BinaryOp, ProgramId, AddPtr, MakeRange
from ..utils import check_out_of_bounds_access, check_storage_contiguous
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
            user_code_index = i + 1  # the next stack is triton user code
        elif ('_grid_executor_call' in frame.name
            and 'triton_viz/core/patch.py' in frame.filename):
            user_code_index = i + 1 # the next stack is triton user code

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

class SymbolicPointer:
    """
    Abstract Pointer:
    - base_expr: a string representing base address. e.g. "base_0x1000"
    - offset_interval: a tuple representing *closed* interval. e.g. (low, high)
    """
    def __init__(self, base_expr: str, offset_interval: tuple):
        self.base_expr: str = base_expr
        self.offset_interval: Tuple[int, int] = offset_interval  # (low, high)

    def __str__(self):
        low, high = self.offset_interval
        return (
            f"SymbolicPointer(\n"
            f"  base_expr='{self.base_expr}',\n"
            f"  offset_interval=[{low}, {high}]\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()


class SymbolicScalar:
    """
    Abstract Scalar:
    - expr: a string representing the scalar. e.g. "x", "(x+1)", "pid_0"
    - interval: a tuple representing *closed* interval. e.g. (low, high)
    """
    def __init__(self, expr: str, interval: tuple):
        self.expr: str = expr
        self.interval: Tuple[int, int] = interval  # (low, high)

    def __str__(self):
        low, high = self.interval
        return (
            f"SymbolicScalar(\n"
            f"  expr='{self.expr}',\n"
            f"  interval=[{low}, {high}]\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()


class SymbolicInterpreter:
    def __init__(self):
        self.trace = []

    def to_symbolic(self, var):
        # case 1: Already symbolic
        if isinstance(var, SymbolicPointer) or isinstance(var, SymbolicScalar):
            return var

        # case 2: TensorHandle
        if isinstance(var, TensorHandle):
            scala_dtypes = [
                tl.int8, tl.int16, tl.int32, tl.int64,
                tl.uint8, tl.uint16, tl.uint32, tl.uint64
            ]
            if var.dtype in scala_dtypes:
                # case 2.1: immediate value
                return SymbolicScalar(
                    str(var.data),
                    (var.data.size - 1, var.data.size - 1)
                )
            elif isinstance(var.dtype, tl.pointer_type):
                # case 2.2: pointer
                return SymbolicPointer(
                    f"base_{str(var.data)}",
                    (0, 0)
                )
            else:
                raise ValueError("Unsupported TensorHandle dtype", var.dtype)

        # case default: raise ValueError
        raise ValueError("Unknown handle_or_symbol type", var)

    def symbolic_program_id(self, axis):
        print(axis)

    def symbolic_add(self, lhs: SymbolicScalar, rhs: SymbolicScalar):
        lhs_low, lhs_high = lhs.interval
        rhs_low, rhs_high = rhs.interval

        new_expr = f"({lhs.expr}+{rhs.expr})"
        new_interval = (lhs_low + rhs_low, lhs_high + rhs_high)
        return SymbolicScalar(new_expr, new_interval)

    def symbolic_sub(self, lhs, rhs):
        raise NotImplementedError(f"symbolic_sub: {lhs} - {rhs} not implemented!")

    def symbolic_mul(self, lhs, rhs):
        raise NotImplementedError(f"symbolic_mul: {lhs} * {rhs} not implemented!")

    def symbolic_div(self, lhs, rhs):
        raise NotImplementedError(f"symbolic_div: {lhs} / {rhs} not implemented!")

    def symbolic_addptr(self, ptr: SymbolicPointer, offset: SymbolicScalar):
        """
        ptr + int
        """
        if not isinstance(ptr, SymbolicPointer):
            raise ValueError("symbolic_addptr: LHS is not a pointer", ptr)
        if not isinstance(offset, SymbolicScalar):
            raise ValueError("symbolic_addptr: RHS offset is not a scalar", offset)

        (old_low, old_high) = ptr.offset_interval
        (offset_low, offset_high) = offset.interval

        elem_size = 4 # TODO: get the element size from somewhere
        new_low = old_low + offset_low * elem_size
        new_high = old_high + offset_high * elem_size

        # create a new pointer
        new_ptr = SymbolicPointer(
            base_expr=ptr.base_expr,
            offset_interval=(new_low, new_high)
        )
        return new_ptr

    def symbolic_load(self, ptr_sym: SymbolicPointer):
        if isinstance(ptr_sym, SymbolicPointer):
            info = f"LOAD from {ptr_sym}"
            self.trace.append(info)
            print(info)
        elif isinstance(ptr_sym, SymbolicScalar):
            raise TypeError("Cannot load directly from a scalar")
        else:
            raise TypeError("Unknown pointer type", type(ptr_sym))

class SanitizerSymbolicExecution(Client):
    def __init__(self, abort_on_error):
        self.abort_on_error = abort_on_error
        self.grid = None
        self.symexec = SymbolicInterpreter()

    def arg_callback(self, arg, arg_cvt):
        pass

    def grid_callback(self, grid: Tuple[int]):
        self.grid = grid

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    def register_op_callback(self, op_type: Type[Op]) -> Tuple[Optional[Callable], Optional[Callable]]:
        def op_program_id_overrider(axis):
            assert self.grid, "Grid not initialized!"
            expr = f"pid_{axis}"
            pid_range = (0, self.grid[axis] - 1)
            return SymbolicScalar(expr, pid_range)

        def op_raw_load_overrider(ptr, _0, _1, is_volatile):
            if not isinstance(ptr, SymbolicPointer):
                raise ValueError("symbolic_load: ptr is not a pointer", ptr)
            self.symexec.symbolic_load(ptr)

        def op_load_overrider(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
            raise NotImplementedError("symbolic_masked_load: not implemented")

        def op_store_overrider(ptr, value, mask, cache_modifier, eviction_policy):
            raise NotImplementedError("symbolic_store: not implemented")

        def op_binary_op_overrider(lhs, rhs, op):
            lhs = self.symexec.to_symbolic(lhs)
            rhs = self.symexec.to_symbolic(rhs)
            if op is np.add:
                self.symexec.symbolic_add(lhs, rhs)
            elif op is np.subtract:
                self.symexec.symbolic_sub(lhs, rhs)
            elif op is np.multiply:
                self.symexec.symbolic_mul(lhs, rhs)
            elif op is np.divide:
                self.symexec.symbolic_div(lhs, rhs)
            else:
                print(lhs, rhs, op)
                raise NotImplementedError()

        def op_addptr_overrider(ptr, offset):
            ptr = self.symexec.to_symbolic(ptr)
            offset = self.symexec.to_symbolic(offset)
            return self.symexec.symbolic_addptr(ptr, offset)

        def op_make_range_overrider(start, end):
            raise NotImplementedError("symbolic_make_range: not implemented")

        if op_type is ProgramId:
            return None, None, op_program_id_overrider
        elif op_type is RawLoad:
            return None, None, op_raw_load_overrider
        elif op_type is Load:
            return None, None, op_load_overrider
        elif op_type is Store:
            return None, None, op_store_overrider
        elif op_type is BinaryOp:
            return None, None, op_binary_op_overrider
        elif op_type is AddPtr:
            return None, None, op_addptr_overrider
        elif op_type is MakeRange:
            return None, None, op_make_range_overrider
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
