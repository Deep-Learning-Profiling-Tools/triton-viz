from ...core.client import Client
from ...core.data import Op, Load, Store
from ..utils import check_out_of_bounds_access, check_storage_contiguous
from .data import OutOfBoundsRecord
from typing import Tuple, Callable, Optional, Type
import numpy as np


class Sanitizer(Client):
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
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, oob)
            ptr.data = first_ptr + oob[-1]

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, check_out_of_bounds_access(ptr.data, mask.data, tensor))
            ptr.data = first_ptr + oob[-1]

        if op_type is Load:
            return pre_load_callback, None
        elif op_type is Store:
            return pre_store_callback, None

        return None, None

    def finalize(self) -> list:
        return self.records
