from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import Op, Load, Store, AddPtr
from .data import LoadStoreBytes
from triton.runtime.interpreter import _get_np_dtype, TensorHandle
import numpy as np
from typing import Callable, Optional


class Profiler(Client):
    NAME = "profiler"

    def __init__(
        self,
        callpath: bool = True,
        disable_buffer_load_check: bool = False,
        disable_for_loop_unroll_check: bool = False,
        disable_load_mask_percentage_check: bool = False,
        block_sampling: bool = False,
        k: int | None = None,
    ):
        super().__init__()  # Initialize parent class
        # Enable ASM collection for the profiler
        self.callpath = callpath
        self.load_bytes = LoadStoreBytes("load", 0, 0)
        self.store_bytes = LoadStoreBytes("store", 0, 0)
        self.has_buffer_load = False
        self.disable_buffer_load_check = disable_buffer_load_check
        self.disable_for_loop_unroll_check = disable_for_loop_unroll_check

        # For-loop statistics
        self.loop_info: list[
            tuple[int, int]
        ] = []  # List of (lineno, total_steps) tuples
        self.loop_linenos_seen: set[
            int
        ] = set()  # Set to track already seen line numbers
        self.disable_load_mask_percentage_check = disable_load_mask_percentage_check
        self.block_sampling = block_sampling
        self.k = k

        # Counters for mask statistics
        self.load_mask_total_count = (
            0  # Total number of mask elements in all load operations
        )
        self.load_mask_false_count = (
            0  # Total number of False elements in all load masks
        )
        self.store_mask_total_count = (
            0  # Total number of mask elements in all store operations
        )
        self.store_mask_false_count = (
            0  # Total number of False elements in all store masks
        )

        # Block sampling state
        self.sampled_blocks: Optional[set[tuple[int, ...]]] = None
        self.current_grid_idx: Optional[tuple[int, ...]] = None

    def pre_run_callback(self, fn: Callable) -> bool:
        # If block sampling is enabled, check if current block is selected
        if self.block_sampling and self.sampled_blocks is not None:
            return self.current_grid_idx in self.sampled_blocks
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        # TODO: optionally proceed the warmup. For now, always proceed.
        return True

    def post_warmup_callback(self, jit_fn, ret) -> None:
        if not ret:
            return

        if (
            not self.disable_buffer_load_check
            and hasattr(ret, "asm")
            and "amdgcn" in ret.asm
        ):
            self.has_buffer_load = "buffer_load" in ret.asm["amdgcn"]
            if self.has_buffer_load:
                print("Detected buffer_load instruction in kernel ASM!")

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        # Store the current grid index
        self.current_grid_idx = grid_idx

    def grid_callback(self, grid: tuple[int, ...]):
        # If block sampling is enabled, determine which blocks to sample
        if self.block_sampling:
            # Generate all possible indices
            all_indices = []
            for x in range(grid[0] if len(grid) > 0 else 1):
                for y in range(grid[1] if len(grid) > 1 else 1):
                    for z in range(grid[2] if len(grid) > 2 else 1):
                        all_indices.append((x, y, z))

            # Sample k blocks randomly
            total_blocks = len(all_indices)
            k = min(self.k if self.k is not None else 1, total_blocks)

            # Randomly select k indices
            perm = np.random.permutation(total_blocks)[:k]
            sampled_indices = [all_indices[i] for i in perm]
            self.sampled_blocks = set(sampled_indices)
        else:
            # No sampling - all blocks will be executed
            self.sampled_blocks = None

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
        def _get_mask_stats(mask: TensorHandle) -> tuple[int, int]:
            """Get mask statistics: total count and false count.

            Args:
                mask: TensorHandle containing boolean mask data

            Returns:
                Tuple of (total_count, false_count)
            """
            total_count = mask.data.size
            false_count = np.count_nonzero(np.logical_not(mask.data))
            return total_count, false_count

        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            self._report_load_store_bytes("load", ptr, mask)
            if not self.disable_load_mask_percentage_check:
                total_count, false_count = _get_mask_stats(mask)
                self.load_mask_total_count += total_count
                self.load_mask_false_count += false_count

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            self._report_load_store_bytes("store", ptr, mask)
            if not self.disable_load_mask_percentage_check:
                total_count, false_count = _get_mask_stats(mask)
                self.store_mask_total_count += total_count
                self.store_mask_false_count += false_count

        def pre_addptr_callback(ptr, offset):
            dtype_tt = ptr.get_element_ty()
            element_bitwidth = dtype_tt.primitive_bitwidth
            element_bytewidth = max(1, element_bitwidth // 8)

            # Get offset data
            offset_data = offset.data if isinstance(offset, TensorHandle) else offset

            # Calculate byte offset
            byte_offset = offset_data * element_bytewidth

            # Check if byte offsets are within 32-bit range
            if not self.disable_buffer_load_check:
                self._check_32bit_range(byte_offset, element_bytewidth, offset_data)

        if op_type is Load:
            return OpCallbacks(before_callback=pre_load_callback)
        elif op_type is Store:
            return OpCallbacks(before_callback=pre_store_callback)
        elif op_type is AddPtr:
            return OpCallbacks(before_callback=pre_addptr_callback)

        return OpCallbacks()

    def register_for_loop_callback(self):
        def loop_hook_before(lineno, iterable):
            if self.disable_for_loop_unroll_check:
                return

            if not isinstance(iterable, range):
                return

            # Only record each unique loop (by line number) once
            # Different blocks will execute the same loop, so we deduplicate by lineno
            if lineno in self.loop_linenos_seen:
                return

            # Record loop information: line number and total steps
            length = len(iterable)
            self.loop_info.append((lineno, length))
            self.loop_linenos_seen.add(lineno)

        def loop_hook_after(lineno: int) -> None:
            # No action needed after loop for profiler
            pass

        return ForLoopCallbacks(
            before_loop_callback=loop_hook_before,
            after_loop_callback=loop_hook_after,
        )

    def finalize(self) -> list:
        # Print for-loop statistics if enabled
        if not self.disable_for_loop_unroll_check and self.loop_info:
            print("\n" + "=" * 60)
            print("Profiler: For-Loop Statistics")
            print("=" * 60)
            print(f"\nTotal for-loops detected: {len(self.loop_info)}\n")

            for idx, (lineno, total_steps) in enumerate(self.loop_info, 1):
                print(f"Loop #{idx}:")
                print(f"  Line number:    {lineno}")
                print(f"  Total steps:    {total_steps}")

        # Calculate and print mask statistics only if load mask percentage check is enabled
        if not self.disable_load_mask_percentage_check:
            print("\n" + "=" * 60)
            print("Profiler: Mask Usage Statistics")
            print("=" * 60)

            # Load statistics
            if self.load_mask_total_count > 0:
                load_masked_percentage = (
                    self.load_mask_false_count / self.load_mask_total_count
                ) * 100
                print("\nLoad Operations:")
                print(f"  Total mask elements:     {self.load_mask_total_count}")
                print(f"  False elements:          {self.load_mask_false_count}")
                print(f"  Masked percentage:       {load_masked_percentage:.2f}%")
            else:
                print("\nLoad Operations:")
                print("  No load operations detected")

            # Store statistics
            if self.store_mask_total_count > 0:
                store_masked_percentage = (
                    self.store_mask_false_count / self.store_mask_total_count
                ) * 100
                print("\nStore Operations:")
                print(f"  Total mask elements:     {self.store_mask_total_count}")
                print(f"  False elements:          {self.store_mask_false_count}")
                print(f"  Masked percentage:       {store_masked_percentage:.2f}%")
            else:
                print("\nStore Operations:")
                print("  No store operations detected")

            print("=" * 60 + "\n")

        return [self.load_bytes, self.store_bytes]
