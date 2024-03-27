from collections import defaultdict
import numpy as np
from .interpreter import record_builder
from .data import Load, Store, Grid


# Function to compute metrics in the analysis shown during visualization.
def analyzer():
    launch_data = record_builder.launches[0]
    op_type_counts = defaultdict(int)
    grid_size = launch_data.grid
    total_load_bytes_true = 0
    total_store_bytes_true = 0
    total_load_bytes_attempted = 0
    total_store_bytes_attempted = 0
    tensor_ptr_to_element_size = {
        tensor.ptr: tensor.element_size for tensor in launch_data.tensors
    }
    for record in launch_data.records:
        if isinstance(record, Grid):
            pass
        else:
            op_type_counts[type(record).__name__] += 1
        if isinstance(record, Load):
            element_size = tensor_ptr_to_element_size[record.ptr]
            mask_true = np.count_nonzero(record.access_masks)
            mask_false = np.count_nonzero(np.logical_not(record.access_masks))
            total_load_bytes_true += mask_true * element_size
            total_load_bytes_attempted += (mask_true + mask_false) * element_size
        elif isinstance(record, Store):
            element_size = tensor_ptr_to_element_size[record.ptr]
            mask_true = np.count_nonzero(record.access_masks)
            mask_false = np.count_nonzero(np.logical_not(record.access_masks))
            total_store_bytes_true += mask_true * element_size
            total_store_bytes_attempted += (mask_true + mask_false) * element_size
    overall_load_ratio = (
        total_load_bytes_true / total_load_bytes_attempted
        if total_load_bytes_attempted > 0
        else 0
    )
    overall_store_ratio = (
        total_store_bytes_true / total_store_bytes_attempted
        if total_store_bytes_attempted > 0
        else 0
    )
    data = [["Grid Size", tuple(grid_size)]]
    data += [[op_type, count] for op_type, count in op_type_counts.items()]
    data.append(["Masked Load Ratio", overall_load_ratio])
    data.append(["Masked Store Ratio", overall_store_ratio])
    return data
