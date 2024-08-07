from triton_viz.data import (
    Tensor,
    Grid,
    Store,
    Load,
    Dot
)
from .trace import Trace
from .interpreter import record_builder
import numpy as np
import torch
from typing import Dict, Tuple, List
import ctypes

def get_tensor_slice_coordinates(x: Load, tensor_table: Dict[int, Tuple[Tensor, int]]) -> List[Tuple[int, int]]:

    
    if x.ptr not in tensor_table:
        raise KeyError(f"Tensor with ptr {x.ptr} not found in tensor_table")

    tensor, _ = tensor_table[x.ptr]
    global_shape = tensor.shape


    if len(global_shape) != 2 or len(x.shape) != 2:
        raise ValueError("This function only supports 2D tensors")

    # Extract the row and column offsets
    row_offsets = x.offsets[:, 0]
    col_offsets = x.offsets[0, :]

    # Find the start and end coordinates
    start_y, end_y = np.min(row_offsets), np.max(row_offsets) + x.shape[0]
    start_x, end_x = np.min(col_offsets), np.max(col_offsets) + x.shape[1]



    # Ensure coordinates are within the global tensor bounds
    start_y = max(0, min(start_y, global_shape[0] - 1))
    start_x = max(0, min(start_x, global_shape[1] - 1))
    end_y = max(start_y + 1, min(end_y, global_shape[0]))
    end_x = max(start_x + 1, min(end_x, global_shape[1]))


    return [
        (int(start_y), int(start_x)),
        (int(start_y), int(end_x) - 1),
        (int(end_y) - 1, int(start_x)),
        (int(end_y) - 1, int(end_x) - 1)
    ]
   

def collect_grid():
    for launch in record_builder.launches[-1:]:
        records, tensor_table, failures = collect_launch(launch)
    return records, tensor_table, failures

def collect_launch(launch):
    tensor_table = {}
    for i, t in enumerate(launch.tensors):
        tensor_table[t.ptr] = (t, i)
    failures = {}
    all_grids = {}
    last_grid = None
    program_records = []
    for r in launch.records:
        if isinstance(r, Grid):
            if last_grid is not None:
                all_grids[last_grid.idx] = program_records
                program_records = []
            last_grid = r
        program_records.append(r)
        if (
            isinstance(r, (Store, Load))
            and (r.invalid_access_masks & r.original_masks).any()
        ):
            failures[last_grid.idx] = True
    all_grids[last_grid.idx] = program_records
    return all_grids, tensor_table, failures


def get_tensor_data(tensor: Tensor):
    # Create a NumPy array from the memory address
    np_array = np.frombuffer(
        (ctypes.c_float * np.prod(tensor.shape)).from_address(tensor.ptr),
        dtype=np.float32
    ).reshape(tensor.shape)
    
    # Convert NumPy array to PyTorch tensor
    return torch.from_numpy(np_array)




def prepare_visualization_data(program_records, tensor_table, getValues=False):
    """Prepare visualization data for the frontend."""
    visualization_data = []
    
    for record in program_records:
        if isinstance(record, Dot):
            intermediate_results = {}
            if getValues:
                intermediate_results = {
                    f"{row},{col}": {
                        'result': result
                    }
                    for (row, col), result in record.intermediate_results.items()
                }
            
            visualization_data.append({
                'type': 'Dot',
                'input_shape': record.input_shape,
                'input_data': record.input_data,
                'other_data': record.other_data,
                'other_shape': record.other_shape,
                'output_shape': record.output_shape,
                'intermediate_results': intermediate_results
            })
            
        elif isinstance(record, (Load, Store)):
            global_tensor, _ = tensor_table[record.ptr]

            slice_coords = get_tensor_slice_coordinates(record, tensor_table)
            slice_tensor = torch.zeros(record.shape)

            global_values = []
            if getValues:
                global_values = global_tensor.data.tolist()

            visualization_data.append({
                'type': 'Load' if isinstance(record, Load) else 'Store',
                'global_values': global_values,
                'global_shape': global_tensor.shape,
                'slice_shape': slice_tensor.shape,
                'slice_coords': slice_coords
            })
            
    return visualization_data

def get_visualization_data(getValues=False):
    """Return the visualization data in a format suitable for JSON serialization."""
    records, tensor_table, failures = collect_grid()
    visualization_data = {}
    for grid_idx, program_records in records.items():
        visualization_data[str(grid_idx)] = prepare_visualization_data(program_records, tensor_table)
    
    # Get the kernel source code
    kernel_src = ""
    if record_builder.launches and isinstance(record_builder.launches[0], Trace):
        kernel_src = record_builder.launches[0].get_src()
    
    return {
        "visualization_data": visualization_data,
        "failures": failures,
        "kernel_src": kernel_src
    }

def serialize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')