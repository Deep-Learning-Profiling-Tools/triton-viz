from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import Op, Load, Store, AddPtr
from .data import LoadStoreBytes
from triton.runtime.interpreter import _get_np_dtype, TensorHandle
import numpy as np
from typing import Callable


class Profiler(Client):
    NAME = "profiler"

    def __init__(
        self,
        callpath: bool = True,
        CHECK_BUFFER_LOAD: bool = False,
        CHECK_LOAD_MASK_PERCENTAGE: bool = False,
    ):
        super().__init__()  # Initialize parent class
        # Enable ASM collection for the profiler
        self.callpath = callpath
        self.load_bytes = LoadStoreBytes("load", 0, 0)
        self.store_bytes = LoadStoreBytes("store", 0, 0)
        self.has_buffer_load = False
        self.CHECK_BUFFER_LOAD = CHECK_BUFFER_LOAD
        self.CHECK_LOAD_MASK_PERCENTAGE = CHECK_LOAD_MASK_PERCENTAGE

        # Counters for mask statistics
        self.total_loads = 0  # Total number of load operations
        self.masked_loads = 0  # Number of loads with mask
        self.total_stores = 0  # Total number of store operations
        self.masked_stores = 0  # Number of stores with mask

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        # TODO: optionally proceed the warmup. For now, always proceed.
        return True

    def post_warmup_callback(self, jit_fn, ret) -> None:
        if not ret:
            return

        if hasattr(ret, "asm") and "amdgcn" in ret.asm:
            self.has_buffer_load = "buffer_load" in ret.asm["amdgcn"]
            if self.has_buffer_load:
                print("Detected buffer_load instruction in kernel ASM!")

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        pass

    def grid_callback(self, grid: tuple[int, ...]):
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

    def _check_32bit_range(
        self, byte_offset: np.ndarray, element_bytewidth: int, offset_data: np.ndarray
    ):
        """Check if byte offsets are within 32-bit signed integer range and print statistics."""
        assert isinstance(
            byte_offset, np.ndarray
        ), f"byte_offset must be np.ndarray, got {type(byte_offset)}"

        # Check if within 32-bit signed range (-2^31 to 2^31 - 1)
        INT32_MIN = -(2**31)
        INT32_MAX = 2**31 - 1

        within_32bit = np.logical_and(
            byte_offset >= INT32_MIN, byte_offset <= INT32_MAX
        )
        outside_32bit = np.logical_not(within_32bit)

        num_outside = np.count_nonzero(outside_32bit)

        # Check if ALL offsets are within 32-bit range
        if num_outside == 0:
            # All offsets are within 32-bit range
            # If we're on AMD GPU and buffer_load is NOT found, this is an error
            if self.has_buffer_load is False:
                assert False, "Buffer Load optimization should be used when offsets are within 32-bit range!"

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def _is_mask_all_true(mask: TensorHandle) -> bool:
            return np.all(mask.data)

        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            self._report_load_store_bytes("load", ptr, mask)
            if self.CHECK_LOAD_MASK_PERCENTAGE:
                if not _is_mask_all_true(mask):
                    self.masked_loads += 1
                self.total_loads += 1

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            self._report_load_store_bytes("store", ptr, mask)
            if self.CHECK_LOAD_MASK_PERCENTAGE:
                if not _is_mask_all_true(mask):
                    self.masked_stores += 1
                self.total_stores += 1

        def pre_addptr_callback(ptr, offset):
            dtype_tt = ptr.get_element_ty()
            element_bitwidth = dtype_tt.primitive_bitwidth
            element_bytewidth = max(1, element_bitwidth // 8)

            # Get offset data
            offset_data = offset.data if isinstance(offset, TensorHandle) else offset

            # Calculate byte offset
            byte_offset = offset_data * element_bytewidth

            # Check if byte offsets are within 32-bit range
            if self.CHECK_BUFFER_LOAD:
                self._check_32bit_range(byte_offset, element_bytewidth, offset_data)

        if op_type is Load:
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)
        elif op_type is AddPtr:
            return OpCallbacks(before_callback=pre_addptr_callback)

        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self) -> list:
        # Calculate and print mask statistics only if CHECK_LOAD_MASK_PERCENTAGE is enabled
        if self.CHECK_LOAD_MASK_PERCENTAGE:
            print("\n" + "=" * 60)
            print("Profiler: Mask Usage Statistics")
            print("=" * 60)

            # Load statistics
            if self.total_loads > 0:
                masked_load_percentage = (self.masked_loads / self.total_loads) * 100
                print(f"\nLoad Operations:")
                print(f"  Total loads:        {self.total_loads}")
                print(f"  Masked loads:       {self.masked_loads}")
                print(f"  Unmasked loads:     {self.total_loads - self.masked_loads}")
                print(f"  Masked percentage:  {masked_load_percentage:.2f}%")
            else:
                print(f"\nLoad Operations:")
                print(f"  No load operations detected")

            # Store statistics
            if self.total_stores > 0:
                masked_store_percentage = (self.masked_stores / self.total_stores) * 100
                print(f"\nStore Operations:")
                print(f"  Total stores:       {self.total_stores}")
                print(f"  Masked stores:      {self.masked_stores}")
                print(f"  Unmasked stores:    {self.total_stores - self.masked_stores}")
                print(f"  Masked percentage:  {masked_store_percentage:.2f}%")
            else:
                print(f"\nStore Operations:")
                print(f"  No store operations detected")

            print("=" * 60 + "\n")

        return [self.load_bytes, self.store_bytes]
