from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Load,
    Store,
    ReduceSum,
    Dot,
    Grid,
    Allocate,
    Flip,
)
from triton_viz.core.masked_load import masked_load
from typing import Callable, Optional, Union
import numpy as np
import traceback


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
        with self._lock:
            self.records.append(Grid(idx=grid_idx))

    def grid_callback(self, grid: tuple[int, ...]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def _extract_user_frames() -> list[traceback.FrameSummary]:
            stack: list[traceback.FrameSummary] = list(traceback.extract_stack())
            # drop current frames (this function and callers)
            stack = stack[:-2]
            cleaned: list[traceback.FrameSummary] = []
            for f in stack:
                fn = f.filename.replace("\\", "/")
                if any(
                    s in fn
                    for s in [
                        "triton_viz/core/",
                        "triton_viz/clients/",
                        "triton/runtime/",
                        "triton/language/",
                        "site-packages/triton/",
                        "runpy.py",
                        "IPython",
                    ]
                ):
                    continue
                cleaned.append(f)
            if cleaned:
                return cleaned
            # fallback to last non "<...>" frame
            for f in reversed(stack):
                if not f.filename.startswith("<"):
                    return [f]
            return stack[-1:]

        def post_allocate_callback(ret):
            assert hasattr(ret, "data")
            self.tensors.append(ret)

        def _convert_keys_to_numpy(keys):
            """Convert any NDArrays in keys to numpy arrays."""
            if isinstance(keys, (tuple, list)):
                return tuple(_convert_keys_to_numpy(k) for k in keys)
            elif hasattr(keys, "data"):
                return keys.data
            else:
                return keys

        @self.lock_fn
        def pre_load_callback(ptr, mask, keys):
            if not self.sample:
                return

            if keys is None:  # i.e. for triton, ptr = pointer + offsets
                first_ptr = np.reshape(ptr.data, (-1))[0]
                tensor = self._get_tensor(first_ptr)
                offsets = ptr.data - tensor.data_ptr()
            else:
                keys = _convert_keys_to_numpy(keys)
                offsets = masked_load(ptr.get_offsets().data, keys, mask=mask.data)
                tensor = ptr

            rec = Load(tensor.data_ptr(), offsets, mask.data)
            rec.call_path = _extract_user_frames()
            self.records.append(rec)

        @self.lock_fn
        def pre_store_callback(ptr, mask, keys):
            if not self.sample:
                return

            if keys is None:  # i.e. for triton, ptr = pointer + offsets, so keys=None
                first_ptr = np.reshape(ptr.data, (-1))[0]
                tensor = self._get_tensor(first_ptr)
                offsets = ptr.data - tensor.data_ptr()
                mask_data = mask.data
            else:
                keys = _convert_keys_to_numpy(keys)
                if mask is None:
                    offsets = masked_load(ptr.get_offsets().data, keys)
                    mask_data = np.ones_like(offsets).astype(bool)
                else:
                    mask_data = mask.data
                    offsets = masked_load(ptr.get_offsets().data, keys, mask=mask_data)
                tensor = ptr

            rec = Store(tensor.data_ptr(), offsets, mask_data)
            rec.call_path = _extract_user_frames()
            self.records.append(rec)

        @self.lock_fn
        def post_reduce_sum_callback(ret, input, axis=None, keep_dims=False):
            if not self.sample:
                return
            input_shape = input.handle.data.shape
            output_shape = ret.handle.data.shape
            self.records.append(ReduceSum(input_shape, axis, keep_dims, output_shape))

        @self.lock_fn
        def post_dot_callback(ret, input, other):
            if not self.sample:
                return
            input_shape = input.data.shape
            other_shape = other.data.shape
            ret_shape = ret.data.shape
            # Pass input/other raw arrays so draw.py can render MatMul
            rec = Dot(input_shape, other_shape, ret_shape, input.data, other.data)
            rec.call_path = _extract_user_frames()
            self.records.append(rec)

        def post_flip_callback(ret, x, *args, **kwargs):
            if not self.sample:
                return
            # Try to capture dim argument
            dim = None
            if args:
                dim = args[0]
            if "dim" in kwargs:
                dim = kwargs.get("dim")
            try:
                in_shape = tuple(x.data.shape)
                out_shape = tuple(ret.data.shape)
            except Exception:
                in_shape = getattr(getattr(x, "handle", None), "data", None)
                out_shape = getattr(getattr(ret, "handle", None), "data", None)
                in_shape = tuple(getattr(in_shape, "shape", []) or [])
                out_shape = tuple(getattr(out_shape, "shape", []) or [])
            rec = Flip(in_shape, out_shape, int(dim) if dim is not None else 0)
            rec.call_path = _extract_user_frames()
            self.records.append(rec)

        if op_type is Allocate:
            return OpCallbacks(after_callback=post_allocate_callback)
        elif op_type is Load:
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)
        elif op_type is ReduceSum:
            return OpCallbacks(after_callback=post_reduce_sum_callback)
        elif op_type is Dot:
            return OpCallbacks(after_callback=post_dot_callback)
        # Flip is wrapped at tl.flip; we don't have an interpreter op to hook here.
        # The wrapper in patch_lang will append Flip records directly to tracer.

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
        with self._lock:
            self.tensors.clear()
            return self.records
