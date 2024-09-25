from triton_viz.data import Tensor, Grid, Store, Load, Dot, ExpandDims
import uuid
import numpy as np
import torch
from .interpreter import record_builder

from typing import Tuple, List


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


def extract_load_coords(
    record: Load, global_tensor: Tensor
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    # Extract coordinates for the global tensor
    global_shape = make_3d(global_tensor.shape)
    global_z, global_y, global_x = delinearized(
        global_shape,
        record.original_offsets,
        global_tensor.dtype,
        record.original_masks,
    )

    global_coords = [
        (float(xi), float(yi), float(zi))
        for xi, yi, zi in zip(global_z, global_y, global_x)
        if xi != -1 and yi != -1 and zi != -1
    ]

    # Extract coordinates for the slice tensor
    slice_shape = make_3d(record.shape)
    slice_z, slice_y, slice_x = record.original_masks.reshape(*slice_shape).nonzero()

    slice_coords = [
        (float(xi), float(yi), float(zi))
        for xi, yi, zi in zip(slice_x, slice_y, slice_z)
    ]

    return global_coords, slice_coords


def make_3d(shape: Tuple[int, ...]):
    if len(shape) == 1:
        return (1, 1, shape[0])
    if len(shape) == 2:
        return (1, shape[0], shape[1])
    return shape


def delinearized(
    shape: Tuple[int, int, int], x: np.ndarray, dtype, mask
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.copy() // (dtype.element_ty.primitive_bitwidth // 8)
    z = ((x // (shape[1] * shape[2])) * mask - (1 - mask)).ravel()
    y = (((x // shape[2]) % shape[1]) * mask - (1 - mask)).ravel()
    x = ((x % shape[2]) * mask - (1 - mask)).ravel()
    return z, y, x


def prepare_visualization_data(program_records, tensor_table):
    """Prepare visualization data for the frntend and raw tensor data for the server."""
    # global idx
    visualization_data = []
    raw_tensor_data = {}
    for record in program_records:
        record_uuid = str(uuid.uuid4())[:8]

        if isinstance(record, ExpandDims):
            print(record.input_shape, record.output_shape, record.index)
        if isinstance(record, Dot):
            visualization_data.append(
                {
                    "type": "Dot",
                    "input_shape": record.input_shape,
                    "other_shape": record.other_shape,
                    "output_shape": record.output_shape,
                    "uuid": record_uuid,
                }
            )

            raw_tensor_data[record_uuid] = {
                "input_data": torch.tensor(record.input_data),
                "other_data": torch.tensor(record.other_data),
                "intermediate_results": record.intermediate_results,
            }

        elif isinstance(record, Load):
            global_tensor, slice_tensor = tensor_table[record.ptr]
            print(global_tensor)
            global_coords, slice_coords = extract_load_coords(record, global_tensor)

            visualization_data.append(
                {
                    "type": "Load",
                    "global_shape": global_tensor.shape,
                    "slice_shape": record.shape,
                    "global_coords": global_coords,
                    "slice_coords": slice_coords,
                    "uuid": record_uuid,
                }
            )

            raw_tensor_data[record_uuid] = {
                "global_tensor": global_tensor.data.cpu(),  # Ensure it's on CPU
                "dims": len(global_tensor.data.cpu().shape),
            }
            print(record.shape)

        elif isinstance(record, Store):
            global_tensor, slice_tensor = tensor_table[record.ptr]

            global_coords, slice_coords = extract_load_coords(record, global_tensor)

            visualization_data.append(
                {
                    "type": "Store",
                    "global_shape": global_tensor.shape,
                    "slice_shape": record.shape,
                    "global_coords": global_coords,
                    "slice_coords": slice_coords,
                    "uuid": record_uuid,
                }
            )

    return visualization_data, raw_tensor_data, ""


def get_visualization_data():
    """Return the visualization data and raw tensor data."""
    records, tensor_table, failures = collect_grid()
    visualization_data = {}
    raw_tensor_data = {}

    for grid_idx, program_records in records.items():
        viz_data, raw_data, kernel_src = prepare_visualization_data(
            program_records, tensor_table
        )
        visualization_data[str(grid_idx)] = viz_data
        raw_tensor_data.update(raw_data)

    # Get the kernel source code

    return {
        "visualization_data": visualization_data,
        "raw_tensor_data": raw_tensor_data,
        "failures": failures,
        "kernel_src": kernel_src,
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
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
