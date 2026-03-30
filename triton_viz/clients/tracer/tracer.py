from collections.abc import Callable

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
    NkiDmaCopy,
    NkiTensorCopy,
    NkiTensorScalar,
    NkiTensorTensor,
)
from ...utils.traceback_utils import extract_user_frames
from triton_viz.core.masked_load_store import masked_load
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

    def _next_time_idx(self) -> int:
        time_idx = int(self._get_thread_local("time_idx", 0) or 0)
        self._set_thread_local("time_idx", time_idx + 1)
        return time_idx

    @staticmethod
    def _shape_tuple(value) -> tuple:
        data = getattr(value, "data", value)
        shape = getattr(data, "shape", ())
        return tuple(shape) if shape is not None else ()

    @staticmethod
    def _buffer_name(value) -> str:
        buffer = str(getattr(value, "buffer", "") or "").upper()
        if buffer:
            return buffer
        return "HBM" if hasattr(getattr(value, "data", value), "shape") else ""

    @staticmethod
    def _nbytes(value) -> int:
        data = getattr(value, "data", value)
        return int(getattr(data, "nbytes", 0) or 0)

    @staticmethod
    def _root_view(value):
        root = value
        seen: set[int] = set()
        while getattr(root, "_parent", None) is not None and id(root) not in seen:
            seen.add(id(root))
            root = root._parent
        return root

    @staticmethod
    def _copy_numpy(value):
        data = getattr(value, "data", value)
        return np.asarray(data).copy()

    @classmethod
    def _view_metadata(cls, value):
        root = cls._root_view(value)
        view_data = getattr(value, "data", value)
        root_data = getattr(root, "data", root)
        view_arr = np.asarray(view_data)
        root_arr = np.asarray(root_data)
        root_copy = root_arr.copy()
        if view_arr.ndim == 0 or root_arr.ndim == 0:
            return root_copy, tuple(root_arr.shape), []
        if view_arr.shape == root_arr.shape and view_arr.ctypes.data == root_arr.ctypes.data:
            return root_copy, tuple(root_arr.shape), cls._full_tensor_coords(root_arr.shape)

        itemsize = max(int(root_arr.dtype.itemsize), 1)
        base_linear = int((view_arr.ctypes.data - root_arr.ctypes.data) // itemsize)
        stride_elems = tuple(int(stride // itemsize) for stride in view_arr.strides)
        coords: list[tuple[float, ...]] = []
        for idx in np.ndindex(view_arr.shape):
            linear = base_linear + sum(i * s for i, s in zip(idx, stride_elems))
            coord = np.unravel_index(linear, root_arr.shape)
            coords.append(tuple(float(v) for v in coord))
        return root_copy, tuple(root_arr.shape), coords

    @staticmethod
    def _full_tensor_coords(shape: tuple[int, ...]) -> list[tuple[float, ...]]:
        if not shape:
            return []
        if len(shape) == 1:
            return [(float(x),) for x in range(shape[0])]
        return [tuple(float(v) for v in idx) for idx in np.ndindex(shape)]

    @staticmethod
    def _op_name(*ops) -> str:
        names = [getattr(op, "name", "") for op in ops if op is not None]
        return " -> ".join(name for name in names if name)

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
        self._set_thread_local("time_idx", 0)

        # Create a Grid record for this grid index
        with self._lock_context():
            self.records.append(Grid(idx=grid_idx))

    def grid_callback(self, grid: tuple[int, ...]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
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
            rec.call_path = extract_user_frames(num_frames=1)
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
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_reduce_sum_callback(ret, input, axis=None, keep_dims=False):
            if not self.sample:
                return
            input_data = getattr(getattr(input, "handle", None), "data", None)
            if input_data is None:
                input_data = getattr(input, "data", None)
            output_data = getattr(getattr(ret, "handle", None), "data", None)
            if output_data is None:
                output_data = getattr(ret, "data", None)
            input_shape = input_data.shape if input_data is not None else ()
            output_shape = output_data.shape if output_data is not None else ()
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
            rec.mem_src = "SBUF"
            rec.mem_dst = "PSUM"
            rec.bytes = self._nbytes(input) + self._nbytes(other)
            rec.time_idx = self._next_time_idx()
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_nki_dma_copy_callback(ret, dst, src, dst_rmw_op=None):
            del ret, dst_rmw_op
            if not self.sample:
                return
            src_root_data, src_root_shape, src_view_coords = self._view_metadata(src)
            dst_root_data, dst_root_shape, dst_view_coords = self._view_metadata(dst)
            rec = NkiDmaCopy(
                self._shape_tuple(src),
                self._shape_tuple(dst),
                mem_src=self._buffer_name(src),
                mem_dst=self._buffer_name(dst),
                bytes=max(self._nbytes(src), self._nbytes(dst)),
                time_idx=self._next_time_idx(),
            )
            rec.input_data = self._copy_numpy(src)
            rec.output_data = self._copy_numpy(dst)
            rec.src_root_data = src_root_data
            rec.src_root_shape = src_root_shape
            rec.src_view_coords = src_view_coords
            rec.dst_root_data = dst_root_data
            rec.dst_root_shape = dst_root_shape
            rec.dst_view_coords = dst_view_coords
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_nki_tensor_copy_callback(ret, dst, src):
            del ret
            if not self.sample:
                return
            src_root_data, src_root_shape, src_view_coords = self._view_metadata(src)
            dst_root_data, dst_root_shape, dst_view_coords = self._view_metadata(dst)
            rec = NkiTensorCopy(
                self._shape_tuple(src),
                self._shape_tuple(dst),
                mem_src=self._buffer_name(src),
                mem_dst=self._buffer_name(dst),
                bytes=max(self._nbytes(src), self._nbytes(dst)),
                time_idx=self._next_time_idx(),
            )
            rec.input_data = self._copy_numpy(src)
            rec.output_data = self._copy_numpy(dst)
            rec.src_root_data = src_root_data
            rec.src_root_shape = src_root_shape
            rec.src_view_coords = src_view_coords
            rec.dst_root_data = dst_root_data
            rec.dst_root_shape = dst_root_shape
            rec.dst_view_coords = dst_view_coords
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_nki_tensor_scalar_callback(ret, dst, data, op0=None, op1=None):
            del ret
            if not self.sample:
                return
            rec = NkiTensorScalar(
                self._shape_tuple(data),
                self._shape_tuple(dst),
                op=self._op_name(op0, op1),
                mem_src=self._buffer_name(data),
                mem_dst=self._buffer_name(dst),
                bytes=max(self._nbytes(data), self._nbytes(dst)),
                time_idx=self._next_time_idx(),
            )
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_nki_tensor_tensor_callback(ret, dst, data1, data2, op=None):
            del ret
            if not self.sample:
                return
            lhs_buffer = self._buffer_name(data1)
            rhs_buffer = self._buffer_name(data2)
            mem_src = lhs_buffer if lhs_buffer == rhs_buffer else ("PSUM" if "PSUM" in (lhs_buffer, rhs_buffer) else lhs_buffer or rhs_buffer)
            rec = NkiTensorTensor(
                self._shape_tuple(data1),
                self._shape_tuple(data2),
                self._shape_tuple(dst),
                op=self._op_name(op),
                mem_src=mem_src,
                mem_dst=self._buffer_name(dst),
                bytes=self._nbytes(data1) + self._nbytes(data2),
                time_idx=self._next_time_idx(),
            )
            rec.call_path = extract_user_frames(num_frames=1)
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
            rec.call_path = extract_user_frames(num_frames=1)
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
        elif op_type is NkiDmaCopy:
            return OpCallbacks(after_callback=post_nki_dma_copy_callback)
        elif op_type is NkiTensorCopy:
            return OpCallbacks(after_callback=post_nki_tensor_copy_callback)
        elif op_type is NkiTensorScalar:
            return OpCallbacks(after_callback=post_nki_tensor_scalar_callback)
        elif op_type is NkiTensorTensor:
            return OpCallbacks(after_callback=post_nki_tensor_tensor_callback)
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
        with self._lock_context():
            self.tensors.clear()
            return self.records
