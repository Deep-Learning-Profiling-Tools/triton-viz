from ...core.client import Client
from ...core.data import Op, Load, Store, MakeRange
from ..utils import check_storage_contiguous
from typing import Tuple, Callable, Optional, Type
import numpy as np
from z3 import Solver, Int, And, Or, Not, sat


class FakeExecutor(Client):
    name: str

    def __init__(self):
        self.name = "fake_executor"

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
            assert False, f'out-of-bound memory access detected: {s.model()[x]}'

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
            lower_bound = valid_addresses.min()
            upper_bound = valid_addresses.max()

            self._check_if_range_statisfy_constraints(lower_bound, upper_bound)

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            base_array = np.reshape(ptr.data, (-1))
            mask_array = np.reshape(mask.data, (-1))

            valid_addresses = base_array[mask_array]
            lower_bound = valid_addresses.min()
            upper_bound = valid_addresses.max()

            self._check_if_range_statisfy_constraints(lower_bound, upper_bound)

        if op_type is Load:
            return pre_load_callback, None
        elif op_type is Store:
            return pre_store_callback, None
        else:
            return None, None

    def finalize(self) -> list:
        return []