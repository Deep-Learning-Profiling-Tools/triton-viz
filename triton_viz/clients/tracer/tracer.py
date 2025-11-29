from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import Op, Load, Store, ReduceSum, Dot, Grid, RawLoad, RawStore
from typing import Callable, Optional, Union
import numpy as np
import threading


def _convert_grid_idx(grid_idx) -> Optional[tuple[int, int, int]]:
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
        grid_idx: Optional[Union[tuple[int], int]] = None,
    ):
        super().__init__()  # Initialize parent class
        self.callpath = callpath
        self.grid_idx = _convert_grid_idx(grid_idx)
        self.records: list = []
        self.tensors: list = []
        self.sample = True
        self._records_lock = threading.Lock()

    def _get_tensor(self, data_ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(self.tensors)):
            if data_ptr < self.tensors[i].data_ptr():
                break
            ret_idx = i
        return self.tensors[ret_idx]

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        return False

    def post_warmup_callback(self, jit_fn, ret) -> None:
        pass

    def arg_callback(self, name, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            self.tensors.append(arg)

    def _append_record(self, record) -> None:
        # Serialize writes to shared records list across threads
        with self._records_lock:
            self.records.append(record)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        if self.grid_idx is not None and grid_idx != self.grid_idx:
            self.sample = False
        else:
            self.sample = True

        # Create a Grid record for this grid index
        self._append_record(Grid(idx=grid_idx))

    def grid_callback(self, grid: tuple[int, ...]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            self._append_record(
                Load(tensor.data_ptr(), ptr.data - tensor.data_ptr(), mask.data)
            )

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            self._append_record(
                Store(tensor.data_ptr(), ptr.data - tensor.data_ptr(), mask.data)
            )

        # Raw (unmasked) ops: synthesize a full True mask based on ptr shape
        def pre_raw_load_callback(ptr, *args, **kwargs):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            offsets = ptr.data - tensor.data_ptr()
            true_mask = np.ones_like(offsets, dtype=bool)
            self._append_record(Load(tensor.data_ptr(), offsets, true_mask))

        def pre_raw_store_callback(ptr, value, *args, **kwargs):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            offsets = ptr.data - tensor.data_ptr()
            true_mask = np.ones_like(offsets, dtype=bool)
            self._append_record(Store(tensor.data_ptr(), offsets, true_mask))

        def post_reduce_sum_callback(ret, input, axis=None, keep_dims=False):
            if not self.sample:
                return
            input_shape = input.handle.data.shape
            output_shape = ret.handle.data.shape
            self._append_record(ReduceSum(input_shape, axis, keep_dims, output_shape))

        def post_dot_callback(ret, input, other, *args):
            if not self.sample:
                return
            input_shape = input.data.shape
            other_shape = other.data.shape
            ret_shape = ret.data.shape
            self._append_record(Dot(input_shape, other_shape, ret_shape))

        if op_type is Load:
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)
        elif op_type is RawLoad:
            return OpCallbacks(before_callback=pre_raw_load_callback)
        elif op_type is RawStore:
            return OpCallbacks(before_callback=pre_raw_store_callback)
        elif op_type is ReduceSum:
            return OpCallbacks(after_callback=post_reduce_sum_callback)
        elif op_type is Dot:
            return OpCallbacks(after_callback=post_dot_callback)

        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    @property
    def sample(self) -> bool:
        return self._get_thread_local("sample", True)

    @sample.setter
    def sample(self, value: bool) -> None:
        self._set_thread_local("sample", value)

    def finalize(self) -> list:
        self.tensors.clear()
        return self.records
