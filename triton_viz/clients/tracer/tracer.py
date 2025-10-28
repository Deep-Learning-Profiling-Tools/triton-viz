from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import Op, Load, Store, ReduceSum, Dot, Grid, RawLoad, RawStore, Array
from triton_viz.core.nki_masked_load import masked_load
from typing import Callable, Optional, Union
import numpy as np


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

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        if self.grid_idx is not None and grid_idx != self.grid_idx:
            self.sample = False
        else:
            self.sample = True

        # Create a Grid record for this grid index
        self.records.append(Grid(idx=grid_idx))

    def grid_callback(self, grid: tuple[int, ...]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def pre_load_callback(ptr, mask, *ignore_args, **ignore_kwargs):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            self.records.append(
                Load(tensor.data_ptr(), ptr.data - tensor.data_ptr(), mask.data)
            )

        def _convert_keys_to_numpy(keys):
            """Convert any NDArrays in keys to numpy arrays."""
            if isinstance(keys, (tuple, list)):
                return tuple(_convert_keys_to_numpy(k) for k in keys)
            elif hasattr(keys, "data"):
                return keys.data
            else:
                return keys

        def post_array_callback(ret, *ignore_args, **ignore_kwargs):
            assert hasattr(ret, "data")
            self.tensors.append(ret)

        def pre_masked_load_callback(
            ptr, keys, mask=None, *ignore_args, **ignore_kwargs
        ):
            if not self.sample:
                return
            keys = _convert_keys_to_numpy(keys)

            self.records.append(
                Load(
                    ptr.data_ptr(),
                    masked_load(ptr.get_offsets().data, keys, mask=mask.data),
                    mask.data,
                )
            )

        def pre_store_callback(ptr, value, mask, *ignore_args, **ignore_kwargs):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            self.records.append(
                Store(tensor.data_ptr(), ptr.data - tensor.data_ptr(), mask.data)
            )

        def pre_masked_store_callback(
            ptr, keys, value, mask=None, *ignore_args, **ignore_kwargs
        ):
            if not self.sample:
                return
            keys = _convert_keys_to_numpy(keys)
            offsets = masked_load(ptr.get_offsets().data, keys)
            if mask is None:
                mask_data = np.ones_like(offsets).astype(bool)
            else:
                mask_data = mask.data

            self.records.append(Store(ptr.data_ptr(), offsets, mask_data))

        # Raw (unmasked) ops: synthesize a full True mask based on ptr shape
        def pre_raw_load_callback(ptr):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            offsets = ptr.data - tensor.data_ptr()
            true_mask = np.ones_like(offsets, dtype=bool)
            self.records.append(Load(tensor.data_ptr(), offsets, true_mask))

        def pre_raw_store_callback(ptr, value):
            if not self.sample:
                return
            first_ptr = np.reshape(ptr.data, (-1))[0]
            tensor = self._get_tensor(first_ptr)
            offsets = ptr.data - tensor.data_ptr()
            true_mask = np.ones_like(offsets, dtype=bool)
            self.records.append(Store(tensor.data_ptr(), offsets, true_mask))

        def post_reduce_sum_callback(
            ret, input, axis=None, keep_dims=False, *ignore_args, **ignore_kwargs
        ):
            if not self.sample:
                return
            input_shape = input.handle.data.shape
            output_shape = ret.handle.data.shape
            self.records.append(ReduceSum(input_shape, axis, keep_dims, output_shape))

        def post_dot_callback(ret, input, other, *ignore_args, **ignore_kwargs):
            if not self.sample:
                return
            input_shape = input.data.shape
            other_shape = other.data.shape
            ret_shape = ret.data.shape
            self.records.append(
                Dot(input_shape, other_shape, ret_shape, input.data, other.data)
            )

        if op_type is Array:  # THTODO: only for NKI
            return OpCallbacks(after_callback=post_array_callback)
        if op_type is Load:
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)
        # if op_type is Load: # THTODO: only for NKI
        #    return OpCallbacks(before_callback=pre_masked_load_callback)
        # elif op_type is Store: # THTODO: only for NKI
        #    return OpCallbacks(before_callback=pre_masked_store_callback)
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

    def finalize(self) -> list:
        self.tensors.clear()
        return self.records
