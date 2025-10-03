from ...core.client import Client
from ...core.callbacks import OpCallbacks
from ...core.data import Op, Load, Store, ReduceSum, Dot, Grid
import numpy as np


def _convert_grid_idx(grid_idx) -> tuple[int, int, int] | None:
    if grid_idx is None:
        return grid_idx

    grid_idx = (grid_idx, 0, 0) if isinstance(grid_idx, int) else grid_idx
    if len(grid_idx) == 1:
        grid_idx = (grid_idx[0], 0, 0)
    elif len(grid_idx) == 2:
        grid_idx = (grid_idx[0], grid_idx[1], 0)
    return grid_idx


class Tracer(Client):
    NAME = "tracer"

    def __init__(
        self,
        callpath: bool = True,
        grid_idx: tuple[int] | int | None = None,
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

    def grid_idx_callback(self, grid_idx: tuple[int]):
        if self.grid_idx is not None and grid_idx != self.grid_idx:
            self.sample = False
        else:
            self.sample = True

        # Create a Grid record for this grid index
        self.records.append(Grid(idx=grid_idx))

    def grid_callback(self, grid: tuple[int]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def pre_load_callback(
            ptr, mask, *ignore_args, **ignore_kwargs
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
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)
        elif op_type is ReduceSum:
            return OpCallbacks(after_callback=post_reduce_sum_callback)
        elif op_type is Dot:
            return OpCallbacks(after_callback=post_dot_callback)

        return OpCallbacks()

    def finalize(self) -> list:
        return self.records
