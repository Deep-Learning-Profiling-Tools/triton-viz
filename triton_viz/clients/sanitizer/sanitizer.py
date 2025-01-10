import traceback
from typing import Tuple, Callable, Optional, Type
import numpy as np
from z3 import Solver, Int, And, Or, Not, sat
from triton.runtime.interpreter import _get_np_dtype, TensorHandle

from ...core.client import Client
from ...core.data import Op, Load, Store
from ..utils import check_out_of_bounds_access, check_storage_contiguous
from .data import OutOfBoundsRecord, OutOfBoundsRecordBruteForce, OutOfBoundsRecordZ3
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
    print(f"File: {oob_record.filename}, Line: {oob_record.lineno}, in {oob_record.func_name}")
    print(f"  Code: {oob_record.line_of_code}")
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
        invalid_index = oob_record.violation_index
        constraints = oob_record.constraints

        # Print OOB details
        print(f"Invalid access detected at index: {invalid_index}")
        print("Constraints:")
        for constraint in constraints:
            print(constraint)

    else:
        raise NotImplementedError("Invalid OutOfBoundsRecord type: " + str(type(oob_record)))

    print("============================================================")
    print("            End of Out-Of-Bounds Record Details             ")
    print("============================================================")

def _get_traceback_info():
    oob_filename, oob_lineno, oob_func_name, oob_line_of_code = "", -1, "", ""
    stack_summary = traceback.extract_stack()
    for i, frame in enumerate(stack_summary):
        if '_jit_function_call' in frame.name \
            and 'triton_viz/core/patch.py' in frame.filename:
            oob_stack_index = i + 1
            if oob_stack_index >= 0:
                oob_filename = stack_summary[oob_stack_index].filename
                oob_lineno = stack_summary[oob_stack_index].lineno
                oob_func_name = stack_summary[oob_stack_index].name
                oob_line_of_code = stack_summary[oob_stack_index].line
            break
    return {
        'filename': oob_filename,
        'lineno': oob_lineno,
        'func_name': oob_func_name,
        'line_of_code': oob_line_of_code
    }

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
        oob_record = OutOfBoundsRecordBruteForce(op_type=op_type, **record, **traceback_info)
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
            ptr.data = tensor.data_ptr() + oob[-1]

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, check_out_of_bounds_access(ptr.data, mask.data, tensor))
            ptr.data = tensor.data_ptr() + oob[-1]

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
        self.word_length = 4    # dtype=float32, word_length = 4 (bytes). TODO: support other dtypes
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

    def _report(self, op_type, tensor, violation_index):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            tensor=tensor,
            violation_index=violation_index,
            constraints=self.constraints,
            **traceback_info)
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
        upperbound_constraint = x <= arg.data_ptr() + nbytes - self.word_length
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

def Sanitizer(abort_on_error=False):
    if sanitizer_backend == "brute_force":
        return SanitizerBruteForce(abort_on_error)
    elif sanitizer_backend == "z3":
        return SanitizerZ3(abort_on_error)
    elif sanitizer_backend == "off":
        return None
    else:
        raise ValueError(f"Invalid TRITON_SANITIZER_BACKEND: {sanitizer_backend}")
