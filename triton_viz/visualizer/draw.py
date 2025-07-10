from colour import Color
from triton_viz.core.data import (
    Tensor,
    Grid,
    Store,
    Load,
    Dot,
    ExpandDims,
)
import numpy as np
import planar
import math
import chalk
from chalk import Diagram, rectangle
from ..core.trace import launches
from ..clients.sanitizer.data import OutOfBoundsRecordBruteForce
import sys
import torch
import uuid

sys.setrecursionlimit(100000)


planar.EPSILON = 0.0
chalk.set_svg_draw_height(500)
BG = Color("white")
WHITE = Color("white")
DEFAULT = Color("grey")
BLACK = Color("black")
GREY = Color("grey")
palette = [
    "#f29f05",
    "#f25c05",
    "#d6568c",
    "#4d8584",
    "#a62f03",
    "#400d01",
    "#274001",
    "#828a00",
]
ACTIVE = [Color(p) for p in palette]

MRATIO = 1 / 3

LAST_RECORD_ONLY = True

# Generic render helpers


def box(d: Diagram, width: float, height: float, outer=0.2) -> Diagram:
    "Put diagram in a box of shape height, width"
    h, w = d.get_envelope().height, d.get_envelope().width
    m = max(h, w)
    back = rectangle(outer + m, outer + m).line_width(0).fill_color(BG).center_xy()
    d = (back + d.center_xy()).with_envelope(back)
    return d.scale_x(width / m).scale_y(height / m)


def reshape(d: Diagram) -> Diagram:
    "Use log-scale if ratio is too sharp"
    h, w = d.get_envelope().height, d.get_envelope().width
    if (h / w > MRATIO) or (w / h > MRATIO):
        d = d.scale_y(math.log(h + 1, 2) / h).scale_x(math.log(w + 1, 2) / w)
    return d


def collect_grid():
    records = []
    tensor_tables = []
    failures = []
    for launch in launches:
        cur_records, cur_tensor_table, cur_failures = collect_launch(launch)
        records.append(cur_records)
        tensor_tables.append(cur_tensor_table)
        failures.append(cur_failures)
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
            isinstance(r, (OutOfBoundsRecordBruteForce))
            and (r.invalid_access_masks & r.op.masks).any()
        ):
            failures[last_grid.idx] = True
    all_grids[last_grid.idx] = program_records
    return all_grids, tensor_table, failures


def extract_load_coords(
    record: Load, global_tensor: Tensor
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
                    "slice_shape": record.masks.shape,
                    "global_coords": global_coords,
                    "slice_coords": slice_coords,
                    "uuid": record_uuid,
                }
            )

            raw_tensor_data[record_uuid] = {
                "global_tensor": global_tensor.data.cpu(),  # Ensure it's on CPU
                "dims": len(global_tensor.data.cpu().shape),
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
