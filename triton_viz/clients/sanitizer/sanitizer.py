from ...core.client import Client
from ...core.data import Op, Load, Store
from ..utils import check_out_of_bounds_access, check_storage_contiguous
from .data import OutOfBoundsRecord
from typing import Tuple, Callable, Optional, Type
import numpy as np
from z3 import Solver, Int, And, Or, Not, sat
from ...core.config import sanitizer_backend, global_warning_toggled
from triton.runtime.interpreter import _get_np_dtype, TensorHandle


class SanitizerBruteForce(Client):
    def __init__(self, callpath: Optional[bool] = True, abort_on_error: Optional[bool] = True):
        self.callpath = callpath
        self.abort_on_error = abort_on_error
        self.tensors: list = []
        self.records: list = []

    def _get_tensor(self, data_ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(self.tensors)):
            if data_ptr < self.tensors[i].data_ptr():
                break
            ret_idx = i
        return self.tensors[ret_idx]

    def _report(self, op_type, record):
        if self.abort_on_error:
            if np.any(record[4]):
                raise ValueError(f"Out of bounds access detected: {record}")
        else:
            self.records.append(OutOfBoundsRecord(op_type, *record))

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
            tensor = self._get_tensor(first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, oob)
            ptr.data = tensor.data_ptr() + oob[-1]

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = self._get_tensor(first_ptr)
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

    def _check_if_range_statisfy_constraints(self, start, end):
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

        if s.check() == sat:
            self._print_constraints()
            print('out-of-bound memory access detected: {s.model()[x]}')
            # assert False, f'out-of-bound memory access detected: {s.model()[x]}'

    def arg_callback(self, arg, arg_cvt):
        if not hasattr(arg, "data_ptr"):
            return
        assert check_storage_contiguous(arg), "The address sanitizer only supports contiguouly stored tensors for now"
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
            if len(valid_addresses) != 0:
                lower_bound = valid_addresses.min()
                upper_bound = valid_addresses.max()
                self._check_if_range_statisfy_constraints(lower_bound, upper_bound)

        def op_load_callback(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
            dtype_tt = ptr.get_element_ty()
            dtype_np = _get_np_dtype(dtype_tt)
            return TensorHandle(np.zeros_like(ptr.data, dtype=dtype_np), dtype_tt)

        def op_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            pass

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            base_array = np.reshape(ptr.data, (-1))
            mask_array = np.reshape(mask.data, (-1))

            valid_addresses = base_array[mask_array]
            lower_bound = valid_addresses.min()
            upper_bound = valid_addresses.max()

            self._check_if_range_statisfy_constraints(lower_bound, upper_bound)

        if op_type is Load:
            return pre_load_callback, None, op_load_callback
        elif op_type is Store:
            return pre_store_callback, None, op_store_callback
        else:
            return None, None, None

    def finalize(self) -> list:
        return []

def Sanitizer(abort_on_error=False):
    available_backends = ["null", "brute_force", "z3"]
    if sanitizer_backend == "brute_force":
        return SanitizerBruteForce(abort_on_error)
    elif sanitizer_backend == "z3":
        return SanitizerZ3(abort_on_error)
    else:
        if not global_warning_toggled['sanitizer']:
            global_warning_toggled['sanitizer'] = True
            print("TRITON_SANITIZER_BACKEND not set. Defaulting to 'brute_force'.")
            print(f"Available backends are: {', '.join(available_backends)}")
        return SanitizerBruteForce(abort_on_error)