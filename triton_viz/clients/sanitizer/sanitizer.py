from ...core.client import Client
from ...core.data import Op, Load, Store
from ..utils import check_out_of_bounds_access, check_storage_contiguous
from .data import OutOfBoundsRecord
from typing import Tuple, Callable, Optional, Type
import numpy as np
import traceback


def print_oob_record(oob_record: OutOfBoundsRecord, max_display=10):
    """
    Print detailed logs for a given OOB record.

    Parameters
    ----------
    oob_record : OutOfBoundsRecord
        The record containing information about out-of-bounds accesses.
    max_display : int
        Maximum number of invalid accesses to display in detail.
    """
    if issubclass(oob_record.op_type, Store):
        op_type = "Store"
    elif issubclass(oob_record.op_type, Load):
        op_type = "Load"
    else:
        assert False, "Not supported op type: " + str(oob_record.op_type)

    # Read the tensor from the record
    tensor = oob_record.tensor

    # Convert memoryviews to NumPy arrays
    offsets_arr = np.array(oob_record.offsets)
    invalid_access_masks_arr = np.array(oob_record.invalid_access_masks)

    # Basic info about the OOB event
    print("============================================================")
    print("                 Out-Of-Bounds Access Detected              ")
    print("============================================================")
    print(f"Operation: {op_type}")
    print(f"Tensor Info: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")
    print(f"Tensor base memory address: {tensor.data_ptr()}")
    print("Valid Access Range: [0, %d)" % (np.prod(tensor.shape) * tensor.element_size()))
    print(f"File: {oob_record.filename}, Line: {oob_record.lineno}, in {oob_record.func_name}")
    print(f"  Code: {oob_record.line_of_code}")
    print("------------------------------------------------------------")

    # Determine all invalid indices
    invalid_indices = np.where(invalid_access_masks_arr.flatten())[0]

    assert len(invalid_indices) != 0, "No invalid accesses found in this record."

    print(f"Total invalid accesses: {len(invalid_indices)}")

    invalid_offsets = offsets_arr.flatten()[invalid_indices]

    print("Invalid offsets:")
    print(invalid_offsets)

    print("============================================================")
    print("            End of Out-Of-Bounds Record Details             ")
    print("============================================================")

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
        oob_filename, oob_lineno, oob_func_name, oob_line_of_code = "", -1, "", ""
        stack_summary = traceback.extract_stack()
        for i, frame in enumerate(stack_summary):
            if '_jit_function_call' in frame.name \
                and 'triton_viz/core/patch.py' in frame.filename:
                oob_stack_index = i + 1
                if oob_stack_index >= 0:
                    oob_filename = stack_summary[oob_stack_index].filename
                    oob_lineno = stack_summary[oob_stack_index].lineno
                    oob_func_name = stack_summary[oob_stack_index].name
                    oob_line_of_code = stack_summary[oob_stack_index].line
                break
        oob_record = OutOfBoundsRecord(op_type, *record, oob_filename, oob_lineno, oob_func_name, oob_line_of_code)
        if self.abort_on_error:
            if np.any(record[4]):
                print_oob_record(oob_record)
                assert False, "Out-of-bounds access detected. See detailed report above."
        else:
            self.records.append(oob_record)

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
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = self._get_tensor(first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, oob)
            ptr.data = tensor.data_ptr() + oob[-1]

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = self._get_tensor(first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, check_out_of_bounds_access(ptr.data, mask.data, tensor))
            ptr.data = tensor.data_ptr() + oob[-1]

        if op_type is Load:
            return pre_load_callback, None
        elif op_type is Store:
            return pre_store_callback, None

        return None, None

    def finalize(self) -> list:
        return self.records
