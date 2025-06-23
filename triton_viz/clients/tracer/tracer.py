from ...core.client import Client
from ...core.data import Op, Load, Store, ReduceSum, Dot
from typing import Tuple, Callable, Optional, Type, Union
import numpy as np


def _convert_grid_idx(grid_idx) -> Optional[Tuple[int, int, int]]:
    if grid_idx is None:
        return grid_idx

    grid_idx = (grid_idx, 0, 0) if isinstance(grid_idx, int) else grid_idx
    if len(grid_idx) == 1:
        grid_idx = (grid_idx[0], 0, 0)
    elif len(grid_idx) == 2:
        grid_idx = (grid_idx[0], grid_idx[1], 0)
    return grid_idx


class Tracer(Client):
    def __init__(
        self,
        callpath: Optional[bool] = True,
        grid_idx: Optional[Union[Tuple[int], int]] = None,
    ):
        self.callpath = callpath
        self.grid_idx = _convert_grid_idx(grid_idx)
        self.records: list = []
        self.tensors: list = []
        self.sample = True

    def _get_tensor(self, data_ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(self.tensors)):
            if data_ptr < self.tensors[i].data_ptr():
                break
            ret_idx = i
        return self.tensors[ret_idx]

    def arg_callback(self, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            self.tensors.append(arg)

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        if self.grid_idx is not None and grid_idx != self.grid_idx:
            self.sample = False
        else:
            self.sample = True

    def grid_callback(self, grid: Tuple[int]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(
        self, op_type: Type[Op]
    ) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]:
        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            self.records.append(
                Load(tensor.data_ptr(), ptr.data - tensor.data_ptr(), mask.data)
            )

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            self.records.append(
                Store(tensor.data_ptr(), ptr.data - tensor.data_ptr(), mask.data)
            )

        def post_reduce_sum_callback(ret, input, axis=None, keep_dims=False):
            if not self.sample:
                return
            input_shape = input.handle.data.shape
            output_shape = ret.handle.data.shape
            self.records.append(ReduceSum(input_shape, axis, keep_dims, output_shape))

        def post_dot_callback(ret, input, other, *args):
            if not self.sample:
                return
            input_shape = input.data.shape
            other_shape = other.data.shape
            ret_shape = ret.data.shape
            self.records.append(Dot(input_shape, other_shape, ret_shape))

        if op_type is Load:
            return pre_load_callback, None, None
        elif op_type is Store:
            return pre_store_callback, None, None
        elif op_type is ReduceSum:
            return None, post_reduce_sum_callback, None
        elif op_type is Dot:
            return None, post_dot_callback, None

        return None, None, None

    def finalize(self) -> list:
        return self.records
