from triton_viz.core.data import (
    Tensor,
    Grid,
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
) -> list[tuple[float, float, float]]:
    # Extract coordinates for the global tensor
    global_shape = make_3d(global_tensor.shape)
    global_z, global_y, global_x = delinearized(
        global_shape,
        record.offsets,
        global_tensor.dtype,
        record.masks,
    )

    # Report (x, y, z); delinearized returns (z, y, x)
    global_coords = [
        (float(xi), float(yi), float(zi))
        for xi, yi, zi in zip(global_x, global_y, global_z)
        if xi != -1 and yi != -1 and zi != -1
    ]

    return global_coords


def make_3d(shape: tuple[int, ...]):
    if len(shape) == 1:
        return (1, 1, shape[0])
    if len(shape) == 2:
        return (1, shape[0], shape[1])
    return shape


def build_slice_tensor(record, tensor: Tensor, global_arr: np.ndarray):
    """Reconstruct slice-tensor values by sampling the global tensor via recorded offsets."""
    try:
        mask = np.asarray(record.masks, dtype=bool)
        offsets = np.asarray(record.offsets, dtype=np.int64)
        if offsets.shape != mask.shape:
            offsets = np.broadcast_to(offsets, mask.shape)
        elem_size = getattr(tensor, "element_size", None)
        if not elem_size:
            try:
                elem_size = tensor.data.element_size()
            except Exception:
                elem_size = 1
        elem_size = max(int(elem_size), 1)

        flat_mask = mask.reshape(-1)
        flat_offsets = offsets.reshape(-1)
        slice_flat = np.zeros_like(flat_offsets, dtype=float)

        if flat_mask.any() and global_arr.size:
            elem_indices = (flat_offsets[flat_mask] // elem_size).astype(np.int64)
            elem_indices = np.clip(elem_indices, 0, global_arr.size - 1)
            global_flat = global_arr.reshape(-1)
            values = global_flat[elem_indices]
            slice_flat[flat_mask] = values
            slice_min = float(values.min())
            slice_max = float(values.max())
        else:
            slice_min = slice_max = 0.0

        return slice_flat.reshape(mask.shape), slice_min, slice_max
    except Exception:
        shape = np.asarray(record.masks).shape
        return np.zeros(shape, dtype=float), 0.0, 0.0


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
    mask = np.asarray(mask, dtype=np.int64, order="C")
    z = ((x // (shape[1] * shape[2])) * mask - (1 - mask)).ravel()
    y = (((x // shape[2]) % shape[1]) * mask - (1 - mask)).ravel()
    x = ((x % shape[2]) * mask - (1 - mask)).ravel()
    return z, y, x


def prepare_visualization_data(program_records, tensor_table):
    """Prepare visualization data for the frntend and raw tensor data for the server."""
    # global idx
    visualization_data = []
    raw_tensor_data = {}
    sbuf_events = []
    load_overall = {}
    store_overall = {}
    for record in program_records:
        record_uuid = str(uuid.uuid4())[:8]
        # Use the current length of visualization_data as the default time_idx
        # This ensures that symmetric programs have matching time_idx for the same logical operation.
        current_time = len(visualization_data)

        # if isinstance(record, ExpandDims):
        #    print(record.input_shape, record.output_shape, record.index)
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
                    # NKI meta (accumulate to PSUM)
                    "mem_src": getattr(record, "mem_src", None),
                    "mem_dst": getattr(record, "mem_dst", None),
                    "tile_shape": getattr(record, "tile_shape", None),
                    "k": int(getattr(record, "k", 0)),
                    # TEMP: the b_swizzle flag is injected only for demo until tracer can infer automatically
                    "b_swizzle": bool(getattr(record, "b_swizzle", False)),
                    "bytes": int(getattr(record, "bytes", 0)),
                    "time_idx": int(getattr(record, "time_idx", current_time)),
                }
            )

            # Normalize Dot operands to NumPy arrays for downstream endpoints
            try:
                import numpy as _np

                def _to_numpy_cpu(x):
                    if isinstance(x, _np.ndarray):
                        return x
                    if hasattr(x, "cpu"):
                        try:
                            return x.detach().cpu().numpy()
                        except Exception:
                            return _np.asarray(x)
                    if hasattr(x, "data"):
                        return _np.asarray(getattr(x, "data"))
                    if hasattr(x, "_value"):
                        return _np.asarray(getattr(x, "_value"))
                    return _np.asarray(x)

                a_np = _to_numpy_cpu(record.input_data)
                b_np = _to_numpy_cpu(record.other_data)
            except Exception:
                import numpy as _np

                a_np = _np.asarray(record.input_data)
                b_np = _np.asarray(record.other_data)

            raw_tensor_data[record_uuid] = {
                "input_data": a_np,
                "other_data": b_np,
                # TEMP: same as above, remember whether demo flagged swizzle
                "b_swizzle": bool(getattr(record, "b_swizzle", False)),
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
                    "time_idx": int(getattr(record, "time_idx", current_time)),
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
            # print(global_tensor)
            # Calculate global_coords for overall map, but lazy load for detailed view
            global_coords = extract_load_coords(record, global_tensor)

            ptr_key = f"LOAD:{int(getattr(record, 'ptr', id(global_tensor)))}"
            time_idx = int(getattr(record, "time_idx", current_time))

            visualization_data.append(
                {
                    "type": "Load",
                    "global_shape": global_tensor.shape,
                    "slice_shape": record.masks.shape,
                    # "global_coords": Lazy load
                    "uuid": record_uuid,
                    "overall_key": ptr_key,
                    "time_idx": time_idx,
                    # NKI flow meta (optional)
                    "mem_src": getattr(record, "mem_src", None),
                    "mem_dst": getattr(record, "mem_dst", None),
                    "bytes": int(getattr(record, "bytes", 0)),
                }
            )

            entry = load_overall.setdefault(
                ptr_key,
                {
                    "shape": list(global_tensor.shape),
                    "dims": len(global_tensor.shape),
                    "slice_shape": list(record.masks.shape),
                    "tiles": [],
                },
            )
            entry["tiles"].append(
                {
                    "uuid": record_uuid,
                    "global_coords": global_coords,
                    "ptr_key": ptr_key,
                    "time_idx": time_idx,
                }
            )

            # Normalize to NumPy array for downstream APIs, and cache basic stats
            try:
                import numpy as _np

                gt = global_tensor.data
                if hasattr(gt, "cpu") and callable(getattr(gt, "cpu", None)):
                    try:
                        arr = gt.detach().cpu().numpy()
                    except Exception:
                        arr = _np.asarray(gt)
                elif hasattr(gt, "data"):
                    arr = _np.asarray(getattr(gt, "data"))
                elif hasattr(gt, "_value"):
                    arr = _np.asarray(getattr(gt, "_value"))
                else:
                    arr = _np.asarray(gt)
            except Exception:
                import numpy as _np

                arr = _np.asarray([])

            # Normalize any NDArray/object payload to plain numpy numeric array
            try:
                import numpy as _np

                def _unwrap(v):
                    # Recursively unwrap objects exposing .data (e.g., NDArray)
                    vv = getattr(v, "data", v)
                    return _unwrap(vv) if hasattr(vv, "data") else vv

                # First unwrap the outer container
                arr = _unwrap(arr)
                arr = _np.asarray(arr)
                if arr.dtype == object:
                    flat = [_unwrap(x) for x in arr.ravel()]
                    # Try direct float cast
                    try:
                        arr = _np.asarray(flat, dtype=float).reshape(arr.shape)
                    except Exception:
                        # Fallback: for nested arrays, take mean as scalar proxy
                        try:
                            scalars = []
                            for x in flat:
                                xx = _np.asarray(_unwrap(x))
                                if xx.size == 0:
                                    scalars.append(0.0)
                                else:
                                    scalars.append(float(xx.astype(float).ravel()[0]))
                            arr = _np.asarray(scalars, dtype=float).reshape(arr.shape)
                        except Exception:
                            arr = _np.zeros(arr.shape, dtype=float)
            except Exception:
                pass

            t_min = float(np.min(arr)) if getattr(arr, "size", 0) else 0.0
            t_max = float(np.max(arr)) if getattr(arr, "size", 0) else 0.0

            raw_tensor_data[record_uuid] = {
                "global_tensor": arr,
                "dims": int(arr.ndim),
                "shape": list(arr.shape),
                "min": t_min,
                "max": t_max,
                "tracebacks": [
                    {
                        "filename": f.filename,
                        "lineno": f.lineno,
                        "line": f.line,
                        "name": f.name,
                    }
                    for f in getattr(record, "call_path", [])
                ],
                # Lazy load metadata
                "global_shape": list(global_tensor.shape),
                "global_dtype": str(global_tensor.dtype),
                "offsets": serialize_for_json(record.offsets),
                "masks": serialize_for_json(record.masks),
            }
            sbuf_events.append(
                {
                    "time_idx": time_idx,
                    "delta": int(getattr(record, "bytes", arr.nbytes)),
                    "label": "load",
                    "uuid": record_uuid,
                }
            )

        elif isinstance(record, Store):
            global_tensor, slice_tensor = tensor_table[record.ptr]

            global_coords = extract_load_coords(record, global_tensor)

            ptr_key = f"STORE:{int(getattr(record, 'ptr', id(global_tensor)))}"
            time_idx = int(getattr(record, "time_idx", current_time))

            visualization_data.append(
                {
                    "type": "Store",
                    "global_shape": global_tensor.shape,
                    "slice_shape": record.masks.shape,
                    # "global_coords": Lazy load
                    "uuid": record_uuid,
                    "overall_key": ptr_key,
                    "time_idx": time_idx,
                    "mem_src": getattr(record, "mem_src", None),
                    "mem_dst": getattr(record, "mem_dst", None),
                    "bytes": int(getattr(record, "bytes", 0)),
                }
            )

            entry = store_overall.setdefault(
                ptr_key,
                {
                    "shape": list(global_tensor.shape),
                    "dims": len(global_tensor.shape),
                    "slice_shape": list(record.masks.shape),
                    "tiles": [],
                },
            )
            entry["tiles"].append(
                {
                    "uuid": record_uuid,
                    "global_coords": global_coords,
                    "ptr_key": ptr_key,
                    "time_idx": time_idx,
                }
            )

            # Reuse the same normalization path as Load so histogram/sampling works
            try:
                import numpy as _np

                gt = global_tensor.data
                if hasattr(gt, "cpu") and callable(getattr(gt, "cpu", None)):
                    try:
                        arr = gt.detach().cpu().numpy()
                    except Exception:
                        arr = _np.asarray(gt)
                elif hasattr(gt, "data"):
                    arr = _np.asarray(getattr(gt, "data"))
                elif hasattr(gt, "_value"):
                    arr = _np.asarray(getattr(gt, "_value"))
                else:
                    arr = _np.asarray(gt)
            except Exception:
                import numpy as _np

                arr = _np.asarray([])

            try:
                import numpy as _np

                def _unwrap(v):
                    vv = getattr(v, "data", v)
                    return _unwrap(vv) if hasattr(vv, "data") else vv

                arr = _unwrap(arr)
                arr = _np.asarray(arr)
                if arr.dtype == object:
                    flat = [_unwrap(x) for x in arr.ravel()]
                    try:
                        arr = _np.asarray(flat, dtype=float).reshape(arr.shape)
                    except Exception:
                        try:
                            scalars = []
                            for x in flat:
                                xx = _np.asarray(_unwrap(x))
                                if xx.size == 0:
                                    scalars.append(0.0)
                                else:
                                    scalars.append(float(xx.astype(float).ravel()[0]))
                            arr = _np.asarray(scalars, dtype=float).reshape(arr.shape)
                        except Exception:
                            arr = _np.zeros(arr.shape, dtype=float)
            except Exception:
                pass

            t_min = float(np.min(arr)) if getattr(arr, "size", 0) else 0.0
            t_max = float(np.max(arr)) if getattr(arr, "size", 0) else 0.0

            raw_tensor_data[record_uuid] = {
                "global_tensor": arr,
                "dims": int(arr.ndim),
                "shape": list(arr.shape),
                "min": t_min,
                "max": t_max,
                "tracebacks": [
                    {
                        "filename": f.filename,
                        "lineno": f.lineno,
                        "line": f.line,
                        "name": f.name,
                    }
                    for f in getattr(record, "call_path", [])
                ],
                # Lazy load metadata
                "global_shape": list(global_tensor.shape),
                "global_dtype": str(global_tensor.dtype),
                "offsets": serialize_for_json(record.offsets),
                "masks": serialize_for_json(record.masks),
            }
            sbuf_events.append(
                {
                    "time_idx": time_idx,
                    "delta": -int(getattr(record, "bytes", arr.nbytes)),
                    "label": "store",
                    "uuid": record_uuid,
                }
            )

    raw_tensor_data["__sbuf_events__"] = sbuf_events
    return visualization_data, raw_tensor_data, "", load_overall, store_overall


def get_visualization_data():
    """Return the visualization data and raw tensor data."""
    records, tensor_table, failures = collect_grid()
    visualization_data = {}
    raw_tensor_data = {}
    kernel_src = ""
    load_overall_maps = {}
    store_overall_maps = {}

    for grid_idx, program_records in records.items():
        (
            viz_data,
            raw_data,
            kernel_src,
            load_overall,
            store_overall,
        ) = prepare_visualization_data(program_records, tensor_table)
        visualization_data[str(grid_idx)] = viz_data
        raw_tensor_data.update(raw_data)
        for key, val in load_overall.items():
            if key in load_overall_maps:
                load_overall_maps[key]["tiles"].extend(val.get("tiles", []))
            else:
                load_overall_maps[key] = val
        for key, val in store_overall.items():
            if key in store_overall_maps:
                store_overall_maps[key]["tiles"].extend(val.get("tiles", []))
            else:
                store_overall_maps[key] = val

    # Ensure failures dict has JSON-serializable keys
    safe_failures = {str(k): v for k, v in failures.items()}

    return {
        "visualization_data": visualization_data,
        "raw_tensor_data": raw_tensor_data,
        "failures": safe_failures,
        "kernel_src": kernel_src,
        "load_overall": load_overall_maps,
        "store_overall": store_overall_maps,
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
