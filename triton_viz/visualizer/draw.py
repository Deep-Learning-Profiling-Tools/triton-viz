from triton_viz.core.data import (
    Tensor,
    Grid,
    ExpandDims,
    Dot,
    Load,
    Store,
    Flip,
)
from triton_viz.clients.sanitizer.data import OutOfBoundsRecordBruteForce
import numpy as np
import sys
import torch
import uuid

sys.setrecursionlimit(100000)


LAST_RECORD_ONLY = True

# Generic render helpers


def collect_grid():
    # If imported at module level, it may capture an empty launches list before trace.py completes initialization.
    # By importing here, we ensure we get the current state of launches with all traced kernel executions.
    from ..core.trace import launches as current_launches

    records = []
    tensor_tables = []
    failures = []
    for launch in current_launches:
        cur_records, cur_tensor_table, cur_failures = collect_launch(launch)
        records.append(cur_records)
        tensor_tables.append(cur_tensor_table)
        failures.append(cur_failures)
    if len(records) == 0:
        # Gracefully handle when there are no launches yet
        return {}, {}, {}
    assert LAST_RECORD_ONLY, "Only last record is supported for now"
    return records[-1], tensor_tables[-1], failures[-1]


def collect_launch(launch):
    tensor_table = {}
    for i, t in enumerate(launch.tensors):
        tensor_table[t.data_ptr()] = (
            Tensor(
                ptr=t.data_ptr(),
                dtype=str(t.dtype),
                stride=tuple(t.stride()),
                shape=tuple(t.shape),
                element_size=t.element_size(),
                data=t,
            ),
            i,
        )
    failures = {}
    all_grids: dict[tuple, list] = {}
    current_idx: tuple | None = None
    for r in launch.records:
        if isinstance(r, Grid):
            current_idx = getattr(r, "idx", None)
            if current_idx is not None and current_idx not in all_grids:
                all_grids[current_idx] = []
            continue
        if current_idx is None:
            current_idx = (0, 0, 0)
            all_grids.setdefault(current_idx, [])
        # append non-Grid ops
        all_grids[current_idx].append(r)
        if isinstance(r, OutOfBoundsRecordBruteForce):
            try:
                if (r.invalid_access_masks & r.op.masks).any():
                    failures[current_idx] = True
            except Exception:
                pass
    return all_grids, tensor_table, failures


def extract_load_coords(
    record, global_tensor: Tensor
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    # Extract coordinates for the global tensor
    global_shape = make_3d(global_tensor.shape)
    global_z, global_y, global_x = delinearized(
        global_shape,
        record.offsets,
        global_tensor.dtype,
        record.masks,
    )

    global_coords = [
        (float(xi), float(yi), float(zi))
        for xi, yi, zi in zip(global_z, global_y, global_x)
        if xi != -1 and yi != -1 and zi != -1
    ]

    # Extract coordinates for the slice tensor
    # Infer shape from masks array
    slice_shape = make_3d(record.masks.shape)
    slice_z, slice_y, slice_x = record.masks.reshape(*slice_shape).nonzero()

    slice_coords = [
        (float(xi), float(yi), float(zi))
        for xi, yi, zi in zip(slice_x, slice_y, slice_z)
    ]

    return global_coords, slice_coords


def make_3d(shape: tuple[int, ...]):
    if len(shape) == 1:
        return (1, 1, shape[0])
    if len(shape) == 2:
        return (1, shape[0], shape[1])
    return shape


def delinearized(
    shape: tuple[int, int, int], x: np.ndarray, dtype, mask
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Handle dtype as string by extracting element size
    if isinstance(dtype, str):
        # Common dtype sizes in bytes
        dtype_sizes = {
            "torch.float32": 4,
            "float32": 4,
            "torch.float64": 8,
            "float64": 8,
            "torch.int32": 4,
            "int32": 4,
            "torch.int64": 8,
            "int64": 8,
            "torch.float16": 2,
            "float16": 2,
            "torch.bfloat16": 2,
            "bfloat16": 2,
            "torch.int8": 1,
            "int8": 1,
            "torch.uint8": 1,
            "uint8": 1,
        }
        element_size = dtype_sizes.get(dtype, 4)  # Default to 4 bytes
    else:
        element_size = dtype.element_ty.primitive_bitwidth // 8

    x = x.copy() // element_size
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
                    # provide C values for color-by-value
                    "c_shape": record.output_shape,
                }
            )

            raw_tensor_data[record_uuid] = {
                "input_data": torch.tensor(record.input_data),
                "other_data": torch.tensor(record.other_data),
                "intermediate_results": record.intermediate_results,
                "tracebacks": [
                    {
                        "filename": f.filename,
                        "lineno": f.lineno,
                        "line": f.line,
                        "name": f.name,
                    }
                    for f in getattr(record, "call_path", [])
                ],
                # prepare C values after kernel (if available from intermediate or recompute best-effort)
            }

        elif isinstance(record, Flip):
            visualization_data.append(
                {
                    "type": "Flip",
                    "input_shape": record.input_shape,
                    "output_shape": record.output_shape,
                    "dim": int(getattr(record, "dim", 0)),
                    "uuid": record_uuid,
                }
            )

            raw_tensor_data[record_uuid] = {
                "tracebacks": [
                    {
                        "filename": f.filename,
                        "lineno": f.lineno,
                        "line": f.line,
                        "name": f.name,
                    }
                    for f in getattr(record, "call_path", [])
                ],
                # best-effort payload for potential future value viz
                "input_shape": list(record.input_shape),
                "output_shape": list(record.output_shape),
                "dim": int(getattr(record, "dim", 0)),
                # optionally include data for hover value queries
                "input_data": None
                if getattr(record, "input_data", None) is None
                else torch.tensor(record.input_data),
                "output_data": None
                if getattr(record, "output_data", None) is None
                else torch.tensor(record.output_data),
            }

        elif isinstance(record, Load):
            global_tensor, slice_tensor = tensor_table[record.ptr]
            print(global_tensor)
            global_coords, slice_coords = extract_load_coords(record, global_tensor)

            visualization_data.append(
                {
                    "type": "Load",
                    "global_shape": global_tensor.shape,
                    "slice_shape": record.masks.shape,
                    "global_coords": global_coords,
                    "slice_coords": slice_coords,
                    "uuid": record_uuid,
                }
            )

            raw_tensor_data[record_uuid] = {
                "global_tensor": global_tensor.data.cpu(),  # Ensure it's on CPU
                "dims": len(global_tensor.data.cpu().shape),
                "tracebacks": [
                    {
                        "filename": f.filename,
                        "lineno": f.lineno,
                        "line": f.line,
                        "name": f.name,
                    }
                    for f in getattr(record, "call_path", [])
                ],
            }
            print(record.masks.shape)

        elif isinstance(record, Store):
            global_tensor, slice_tensor = tensor_table[record.ptr]

            global_coords, slice_coords = extract_load_coords(record, global_tensor)

            visualization_data.append(
                {
                    "type": "Store",
                    "global_shape": global_tensor.shape,
                    "slice_shape": record.masks.shape,
                    "global_coords": global_coords,
                    "slice_coords": slice_coords,
                    "uuid": record_uuid,
                }
            )

            raw_tensor_data[record_uuid] = {
                "tracebacks": [
                    {
                        "filename": f.filename,
                        "lineno": f.lineno,
                        "line": f.line,
                        "name": f.name,
                    }
                    for f in getattr(record, "call_path", [])
                ],
            }

    return visualization_data, raw_tensor_data, ""


def get_visualization_data():
    """Return the visualization data and raw tensor data."""
    records, tensor_table, failures = collect_grid()
    visualization_data = {}
    raw_tensor_data = {}
    kernel_src = ""

    for grid_idx, program_records in records.items():
        viz_data, raw_data, kernel_src = prepare_visualization_data(
            program_records, tensor_table
        )
        visualization_data[str(grid_idx)] = viz_data
        raw_tensor_data.update(raw_data)

    # Ensure failures dict has JSON-serializable keys
    safe_failures = {str(k): v for k, v in failures.items()}

    return {
        "visualization_data": visualization_data,
        "raw_tensor_data": raw_tensor_data,
        "failures": safe_failures,
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
