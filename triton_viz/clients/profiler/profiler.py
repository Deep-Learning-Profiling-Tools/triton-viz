from ...core.client import Client
from ...core.callbacks import OpCallbacks
from ...core.data import Op, Load, Store
from .data import LoadStoreBytes
from triton.runtime.interpreter import _get_np_dtype, TensorHandle
import numpy as np


class Profiler(Client):
    NAME = "profiler"

    def __init__(self, callpath: bool = True):
        self.callpath = callpath
        self.load_bytes = LoadStoreBytes("load", 0, 0)
        self.store_bytes = LoadStoreBytes("store", 0, 0)

    def arg_callback(self, arg, arg_cvt):
        pass

    def grid_idx_callback(self, grid_idx: tuple[int]):
        pass

    def grid_callback(self, grid: tuple[int]):
        pass

    def _report_load_store_bytes(self, type, ptr: TensorHandle, mask: TensorHandle):
        dtype_tt = ptr.get_element_ty()
        dtype_np: np.dtype = _get_np_dtype(dtype_tt)
        mask_true = np.count_nonzero(mask.data)
        mask_false = np.count_nonzero(np.logical_not(mask.data))
        total_bytes_true = mask_true * dtype_np.itemsize
        total_bytes_attempted = (mask_true + mask_false) * dtype_np.itemsize
        if type == "load":
            self.load_bytes.total_bytes_attempted += total_bytes_attempted
            self.load_bytes.total_bytes_true += total_bytes_true
        elif type == "store":
            self.store_bytes.total_bytes_attempted += total_bytes_attempted
            self.store_bytes.total_bytes_true += total_bytes_true

    def register_op_callback(self, op: type[Op]) -> OpCallbacks:
        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            self._report_load_store_bytes("load", ptr, mask)

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            self._report_load_store_bytes("store", ptr, mask)

        if isinstance(op, Load):
            return OpCallbacks(before_callback=pre_load_callback)
        elif isinstance(op, Store):
            return OpCallbacks(before_callback=pre_store_callback)

        return OpCallbacks()

    def finalize(self) -> list:
        return [self.load_bytes, self.store_bytes]
