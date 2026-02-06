import threading
from typing import Any
import sys
from flask import Flask, render_template, jsonify, request
from .analysis import analyze_records
from .draw import get_visualization_data, delinearized, make_3d
from ..utils.traceback_utils import safe_read_file_segment
import os
import torch
import numpy as np
from flask_cloudflared import _run_cloudflared
import requests
import time

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)

# Global variables to store the data
global_data = None
raw_tensor_data = None
precomputed_c_values: dict[str, dict[tuple[int, int], list[float]]] = {}
current_fullscreen_op = None
last_public_url = None
last_local_port = None
last_launch_snapshot = None
sbuf_events = []
load_overall_maps = {}
store_overall_maps = {}
DEVICE_LIMITS = {
    "TRN1_NC_V2": 24 * 1024 * 1024,
    "TRN1_CHIP": 48 * 1024 * 1024,
    "TRN1_2XL": 48 * 1024 * 1024,
    "TRN1_32XL": 768 * 1024 * 1024,
    "TRN2_NC_V3": 28 * 1024 * 1024,
    "TRN2_CHIP": 224 * 1024 * 1024,
    "TRN2_48XL": 3584 * 1024 * 1024,
}


def _compute_sbuf_timeline(events: list[dict]) -> tuple[list[dict], int]:
    if not events:
        return [], 0
    usage = 0
    timeline = []
    usage_values = []
    for ev in sorted(events, key=lambda e: e.get("time_idx", 0)):
        usage += int(ev.get("delta", 0))
        usage_values.append(usage)
        timeline.append(
            {
                "time_idx": int(ev.get("time_idx", 0)),
                "usage": usage,
                "label": ev.get("label"),
                "uuid": ev.get("uuid"),
            }
        )
    max_usage = max(usage_values) if usage_values else 0
    return timeline, max_usage


# Server state management
class ServerState:
    """Encapsulates server state to avoid global variables."""

    def __init__(self):
        self.last_public_url = None
        self.last_local_port = None

    def set_public_url(self, url):
        self.last_public_url = url

    def set_local_port(self, port):
        self.last_local_port = port

    def get_public_url(self):
        return self.last_public_url

    def get_local_port(self):
        return self.last_local_port


# Create a single instance for the module
_server_state = ServerState()


def _to_numpy_array(value):
    """Best-effort conversion to a numpy array."""
    if value is None:
        return np.asarray([])
    if isinstance(value, np.ndarray):
        return value
    try:
        if hasattr(value, "detach") and callable(getattr(value, "detach", None)):
            value = value.detach()
        if hasattr(value, "cpu") and callable(getattr(value, "cpu", None)):
            value = value.cpu()
        if hasattr(value, "numpy") and callable(getattr(value, "numpy", None)):
            return value.numpy()
        if hasattr(value, "data"):
            return _to_numpy_array(value.data)
        return np.asarray(value)
    except Exception:
        try:
            return np.asarray(value)
        except Exception:
            return np.asarray([])


def precompute_c_values(op_data):
    # Normalize to torch.Tensor (Dot operands may arrive as NumPy arrays)
    input_tensor = torch.as_tensor(_to_numpy_array(op_data["input_data"]))
    other_tensor = torch.as_tensor(_to_numpy_array(op_data["other_data"]))

    if input_tensor.ndim != 2 or other_tensor.ndim != 2:
        return {}

    rows, inner_dim = input_tensor.shape
    cols = other_tensor.shape[1]
    if rows == 0 or cols == 0 or inner_dim == 0:
        return {}

    contrib = torch.einsum("ik,kj->kij", input_tensor, other_tensor)
    cum = torch.cumsum(contrib, dim=0).cpu()

    precomputed = {}
    base = [0.0]
    for i in range(rows):
        for j in range(cols):
            precomputed[(i, j)] = base + cum[:, i, j].tolist()

    return precomputed


def update_global_data(force: bool = False):
    global global_data
    global raw_tensor_data
    global precomputed_c_values
    global sbuf_events
    global load_overall_maps
    global store_overall_maps
    global last_launch_snapshot

    # Collect all records from launches
    from ..core.trace import launches

    total_records = 0
    for launch in launches:
        total_records += len(launch.records)
    snapshot = (len(launches), total_records)
    if not force and global_data is not None and snapshot == last_launch_snapshot:
        return
    last_launch_snapshot = snapshot

    all_records: list = []
    for launch in launches:
        # Include the Launch object itself so analysis can attach per-kernel
        # metadata (e.g., grid size) before consuming client records.
        all_records.append(launch)
        all_records.extend(launch.records)

    # Pass the records to analyze_records
    analysis_data = analyze_records(all_records)
    viz_data = get_visualization_data()
    try:
        keys = list(viz_data.get("visualization_data", {}).keys())
        print(f"[viz] grids: {keys}")
        for k in keys:
            ops = viz_data["visualization_data"].get(k, [])
            print(f"[viz] grid {k} ops: {[op.get('type') for op in ops]}")
    except Exception as e:
        print("[viz] debug logging failed:", e)
    global_data = {
        "ops": {
            "visualization_data": viz_data["visualization_data"],
            "failures": viz_data["failures"],
            "kernel_src": viz_data["kernel_src"],
        }
    }
    raw_tensor_data = viz_data["raw_tensor_data"]
    sbuf_events = raw_tensor_data.pop("__sbuf_events__", [])
    load_overall_maps = viz_data.get("load_overall", {})
    store_overall_maps = viz_data.get("store_overall", {})

    # Precompute C values for each Dot operation
    precomputed_c_values = {}
    for uuid, op_data in raw_tensor_data.items():
        if "input_data" in op_data and "other_data" in op_data:
            precomputed_c_values[uuid] = precompute_c_values(op_data)

    # Convert analysis_data to a dictionary format similar to pandas DataFrame.to_dict()
    # analysis_data is a list of lists where each inner list contains [metric, value] pairs
    df_dict: dict[str, list[Any]] = {"Metric": [], "Value": []}
    for record in analysis_data:
        for metric, value in record:
            df_dict["Metric"].append(metric)
            df_dict["Value"].append(value)

    global_data["analysis"] = df_dict


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    global global_data
    update_global_data()
    return jsonify(global_data)


@app.route("/api/sbuf")
def get_sbuf_usage():
    global sbuf_events
    device = (request.args.get("device") or "A100").upper()
    limit = DEVICE_LIMITS.get(device)
    if limit is None:
        limit = int(request.args.get("limit_bytes", 256 * 1024))
    timeline, max_usage = _compute_sbuf_timeline(sbuf_events)
    overflow = [pt for pt in timeline if pt["usage"] > limit]
    return jsonify(
        {
            "timeline": timeline,
            "limit_bytes": limit,
            "max_usage": max_usage,
            "overflow_points": overflow,
        }
    )


@app.route("/api/update_data")
def update_data():
    update_global_data(force=True)
    return jsonify({"status": "Data updated successfully"})


@app.route("/api/setop", methods=["POST"])
def set_current_op():
    global current_fullscreen_op
    data = request.json
    current_fullscreen_op = data.get("uuid")
    return jsonify(
        {"status": "Current op set successfully", "uuid": current_fullscreen_op}
    )


@app.route("/api/op_code", methods=["POST"])
def get_op_code():
    global raw_tensor_data, current_fullscreen_op
    data = request.json or {}
    # ensure data prepared
    if raw_tensor_data is None:
        update_global_data()
    uuid = data.get("uuid") or current_fullscreen_op
    context = int(data.get("context", 8))
    if not uuid or raw_tensor_data is None or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    tb_list = raw_tensor_data[uuid].get("tracebacks") or []
    if not tb_list:
        return jsonify({"error": "Traceback not available"}), 200
    # Last frame = innermost user code, closest to tl.load/tl.store
    tb = tb_list[-1]
    filename = tb.get("filename")
    lineno = int(tb.get("lineno", 0))
    line_of_code = tb.get("line")
    seg = safe_read_file_segment(filename, lineno, context)
    if seg is None:
        # fallback with single line
        seg = {
            "filename": filename,
            "lineno": lineno,
            "start": lineno,
            "end": lineno,
            "highlight": lineno,
            "lines": [{"no": lineno, "text": line_of_code or ""}],
        }
    return jsonify(seg)


@app.route("/api/getMatmulC", methods=["POST"])
def get_matmul_c():
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")
    if not uuid or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    op = raw_tensor_data[uuid]
    a = op.get("input_data")
    b = op.get("other_data")
    if a is None or b is None:
        return jsonify({"error": "MatMul tensors not available"}), 200
    try:
        import numpy as _np

        a_np = _np.asarray(a)
        b_np = _np.asarray(b)
        c_np = a_np @ b_np
        cmin = float(_np.min(c_np)) if c_np.size else 0.0
        cmax = float(_np.max(c_np)) if c_np.size else 0.0
        return jsonify(
            {
                "shape": list(c_np.shape),
                "min": cmin,
                "max": cmax,
                "values": c_np.tolist(),
            }
        )
    except Exception as e:
        return jsonify({"error": f"MatMul compute failed: {e}"}), 200


@app.route("/api/getMatmulA", methods=["POST"])
def get_matmul_a():
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")
    if not uuid or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    op = raw_tensor_data[uuid]
    a = op.get("input_data")
    if a is None:
        return jsonify({"error": "MatMul A tensor not available"}), 200
    try:
        import numpy as _np

        a_np = _np.asarray(a)
        amin = float(_np.min(a_np)) if a_np.size else 0.0
        amax = float(_np.max(a_np)) if a_np.size else 0.0
        return jsonify(
            {
                "shape": list(a_np.shape),
                "min": amin,
                "max": amax,
                "values": a_np.tolist(),
            }
        )
    except Exception as e:
        return jsonify({"error": f"MatMul A fetch failed: {e}"}), 200


@app.route("/api/getMatmulB", methods=["POST"])
def get_matmul_b():
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")
    if not uuid or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    op = raw_tensor_data[uuid]
    b = op.get("other_data")
    if b is None:
        return jsonify({"error": "MatMul B tensor not available"}), 200
    try:
        import numpy as _np

        b_np = _np.asarray(b)
        bmin = float(_np.min(b_np)) if b_np.size else 0.0
        bmax = float(_np.max(b_np)) if b_np.size else 0.0
        return jsonify(
            {
                "shape": list(b_np.shape),
                "min": bmin,
                "max": bmax,
                "values": b_np.tolist(),
            }
        )
    except Exception as e:
        return jsonify({"error": f"MatMul B fetch failed: {e}"}), 200


@app.route("/api/getSBufData", methods=["POST"])
@app.route("/api/histogram", methods=["POST"])
def get_value_histogram():
    """Return histogram statistics for tensors associated with an operation."""
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")
    if not uuid or not raw_tensor_data or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404

    source = (data.get("source") or "").upper()
    bins = max(2, min(512, int(data.get("bins", 64))))
    max_samples = max(1000, int(data.get("max_samples", 200000)))

    op_payload = raw_tensor_data[uuid]
    arr = None

    if source in {"GLOBAL", "LOAD_GLOBAL"}:
        arr = op_payload.get("global_tensor")
    elif source == "A":
        arr = op_payload.get("input_data")
    elif source == "B":
        arr = op_payload.get("other_data")
    elif source == "C":
        arr = op_payload.get("output_data")
        if arr is None:
            a = op_payload.get("input_data")
            b = op_payload.get("other_data")
            if a is not None and b is not None:
                try:
                    arr = np.matmul(_to_numpy_array(a), _to_numpy_array(b))
                    op_payload["output_data"] = arr
                except Exception:
                    arr = None
    else:
        return jsonify({"error": f"Unsupported source '{source}'"}), 400

    arr_np = _to_numpy_array(arr)
    if arr_np.size == 0:
        return jsonify(
            {
                "counts": [],
                "edges": [],
                "min": 0.0,
                "max": 0.0,
                "n": 0,
                "sampled": 0,
            }
        )

    flat = arr_np.astype(np.float64).ravel()
    total = flat.size

    if total > max_samples:
        rng = np.random.default_rng()
        idx = rng.choice(total, size=max_samples, replace=False)
        sample = flat[idx]
    else:
        sample = flat

    counts, edges = np.histogram(sample, bins=bins)

    return jsonify(
        {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "n": int(total),
            "sampled": int(sample.size),
        }
    )


@app.route("/api/getValue", methods=["POST"])
def get_value():
    global raw_tensor_data, precomputed_c_values, current_fullscreen_op
    print(current_fullscreen_op)
    data = request.json
    uuid = data.get("uuid")
    matrix_name = data.get("matrixName")
    row = data.get("row")
    col = data.get("col")

    if uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404

    op_data = raw_tensor_data[uuid]

    if matrix_name == "A":
        value = (
            op_data["input_data"][row, col].item() if "input_data" in op_data else None
        )
        return jsonify({"value": value})
    elif matrix_name == "B":
        value = (
            op_data["other_data"][row, col].item() if "other_data" in op_data else None
        )
        return jsonify({"value": value})
    elif matrix_name == "C":
        current_step = data.get("currentStep", 0)

        if uuid not in precomputed_c_values:
            return jsonify({"error": "Precomputed values not found"}), 404

        precomputed = precomputed_c_values[uuid]
        current_value = precomputed[(row, col)][current_step]

        return jsonify(
            {
                "value": current_value,
            }
        )
    else:
        return jsonify({"error": "Invalid matrix name"}), 400


@app.route("/api/getLoadValue", methods=["POST"])
def get_load_value():
    global raw_tensor_data, current_fullscreen_op

    data = request.json
    uuid = data.get("uuid")
    x = data.get("x")
    y = data.get("y")
    z = data.get("z")
    print(x, y, z)
    if uuid is None or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404

    op_data = raw_tensor_data[uuid]

    if "global_tensor" in op_data and (
        x is not None and y is not None and z is not None
    ):
        try:
            import numpy as _np

            arr = _np.asarray(op_data["global_tensor"])  # already NumPy in draw.py
            yy, xx, zz = int(y), int(x), int(z)
            if arr.ndim >= 3:
                value = float(arr[yy, xx, zz])
            elif arr.ndim == 2:
                value = float(arr[yy, xx])
            elif arr.ndim == 1:
                value = float(arr[xx])
            else:
                value = 0.0
            return jsonify({"value": value})
        except IndexError:
            return jsonify({"error": "Coordinates out of bounds"}), 200
    else:
        return jsonify({"error": "Global tensor data not found"}), 200


@app.route("/api/getFlipValue", methods=["POST"])
def get_flip_value():
    """Return a value from input or output of a Flip op by linear index or 2D coords.

    Request JSON: { uuid, which: "input"|"output", x, y }
    - If tensor is 1D, use x only; if 2D, use x (col), y (row)
    """
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")
    which = (data.get("which") or "input").lower()
    x = data.get("x")
    y = data.get("y")
    if not uuid or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    op = raw_tensor_data[uuid]
    t = op.get("input_data") if which == "input" else op.get("output_data")
    shape = op.get("input_shape") if which == "input" else op.get("output_shape")
    if t is None or shape is None:
        return jsonify({"error": "Flip tensor data unavailable"}), 200
    try:
        if len(shape) <= 1:
            idx = int(x or 0)
            return jsonify({"value": t[idx].item()})
        elif len(shape) == 2:
            rr = int(y or 0)
            cc = int(x or 0)
            return jsonify({"value": t[rr, cc].item()})
        else:
            # higher dims not directly supported; fallback to flat index
            idx = int(x or 0)
            flat = t.view(-1)
            return jsonify({"value": flat[idx].item()})
    except Exception as e:
        return jsonify({"error": f"Flip get value failed: {e}"}), 200


@app.route("/api/getLoadTensor", methods=["POST"])
def get_load_tensor():
    """Return entire global tensor for a given Load/Store op, with min/max.

    Response schema:
      {
        "shape": [d0, d1, d2?],
        "dims": 1|2|3,
        "min": float,
        "max": float,
        "values": nested_list  # Python list converted from torch tensor
      }
    """
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")

    if raw_tensor_data is None or uuid is None or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404

    op_data = raw_tensor_data[uuid]
    if "global_tensor" not in op_data:
        return jsonify({"error": "Global tensor data not found"}), 200

    import numpy as _np

    try:
        arr = _np.asarray(op_data["global_tensor"])  # already NumPy in draw.py
        t_shape = op_data.get("shape") or list(arr.shape)
        t_dims = op_data.get("dims") or int(arr.ndim)
        t_min = op_data.get("min")
        t_max = op_data.get("max")
        if t_min is None:
            t_min = float(_np.min(arr)) if arr.size else 0.0
        if t_max is None:
            t_max = float(_np.max(arr)) if arr.size else 0.0
        payload = {
            "shape": t_shape,
            "dims": int(t_dims),
            "min": float(t_min),
            "max": float(t_max),
            "values": arr.tolist(),
        }

        slice_arr = op_data.get("slice_tensor")
        if slice_arr is not None:
            slice_np = _np.asarray(slice_arr)
            slice_shape = op_data.get("slice_shape") or list(slice_np.shape)
            slice_dims = op_data.get("slice_dims") or int(slice_np.ndim)
            slice_min = op_data.get("slice_min")
            slice_max = op_data.get("slice_max")
            if slice_min is None:
                slice_min = float(_np.min(slice_np)) if slice_np.size else 0.0
            if slice_max is None:
                slice_max = float(_np.max(slice_np)) if slice_np.size else 0.0
            payload["slice"] = {
                "shape": slice_shape,
                "dims": int(slice_dims),
                "min": float(slice_min),
                "max": float(slice_max),
                "values": slice_np.tolist(),
            }

        # Calculate highlights
        if "offsets" in op_data:
            offsets = np.asarray(op_data["offsets"])
            masks = np.asarray(op_data["masks"])
            shape = tuple(op_data["global_shape"])
            dtype = op_data["global_dtype"]

            shape_3d = make_3d(shape)
            gz, gy, gx = delinearized(shape_3d, offsets, dtype, masks)

            # Filter invalid
            valid = (gx != -1) & (gy != -1) & (gz != -1)
            if valid.any():
                gx = gx[valid]
                gy = gy[valid]
                gz = gz[valid]
                coords = np.stack([gx, gy, gz], axis=1)

                min_c = coords.min(axis=0)
                max_c = coords.max(axis=0)
                dims = max_c - min_c + 1
                volume = np.prod(dims)
                unique_coords = np.unique(coords, axis=0)

                if len(unique_coords) == volume:
                    # Robust check: all coordinates in the box [min_c, max_c] must be present
                    # We already know the count matches volume, so we just need to verify they are all distinct
                    # and within bounds (which unique + bounding box logic already implies).
                    # However, to be absolutely sure it's not a sparse set that happens to have 'volume' points:
                    # Generate the dense set and compare.
                    is_dense = True
                    # Optimization: if it's really dense, every integer in the range [0, volume)
                    # of linear indices relative to min_c must be present.
                    # Linear index = (z-sz)*dx*dy + (y-sy)*dx + (x-sx)
                    offsets_rel = unique_coords - min_c
                    lin_indices = (
                        offsets_rel[:, 2] * dims[1] * dims[0]
                        + offsets_rel[:, 1] * dims[0]
                        + offsets_rel[:, 0]
                    )
                    if not (np.sort(lin_indices) == np.arange(volume)).all():
                        is_dense = False

                    if is_dense:
                        payload["highlights"] = {
                            "type": "descriptor",
                            "start": min_c.tolist(),
                            "shape": dims.tolist(),
                        }
                    else:
                        payload["highlights"] = {
                            "type": "array",
                            "data": coords.tolist(),
                        }
                else:
                    payload["highlights"] = {"type": "array", "data": coords.tolist()}

        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": f"getLoadTensor failed: {e}"}), 200


def _parse_grid_key(raw_key: str) -> tuple[int, int, int] | None:
    key = str(raw_key or "").strip()
    key = key.lstrip("(").rstrip(")")
    parts = [p.strip() for p in key.split(",") if p.strip()]
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (TypeError, ValueError):
        return None


def _collect_load_store_program_subsets(op_type, overall_key, op_index, time_idx):
    if not global_data or "ops" not in global_data:
        return {
            "coords": [],
            "subsets": {},
            "subset_count": 0,
            "counts": [],
            "max_count": 0,
            "shape": [],
        }
    viz = global_data["ops"].get("visualization_data") or {}
    uuid_to_pid: dict[str, tuple[int, int, int]] = {}
    for grid_key, ops in viz.items():
        pid = _parse_grid_key(grid_key)
        if pid is None:
            continue
        for op in ops or []:
            is_match = (
                op.get("type") == op_type and op.get("overall_key") == overall_key
            )
            if not is_match:
                continue
            if op_index is not None:
                is_match = op.get("op_index") == op_index
            else:
                is_match = op.get("time_idx") == time_idx
            if is_match:
                uuid = op.get("uuid")
                if uuid:
                    uuid_to_pid[uuid] = pid

    overall_maps = load_overall_maps if op_type == "Load" else store_overall_maps
    entry = overall_maps.get(overall_key)
    if not entry or not uuid_to_pid:
        return {
            "coords": [],
            "subsets": {},
            "subset_count": 0,
            "counts": [],
            "max_count": 0,
            "shape": list(entry.get("shape") or []) if entry else [],
        }
    shape = list(entry.get("shape") or [])
    uuid_to_coords = {
        tile.get("uuid"): tile.get("global_coords")
        for tile in entry.get("tiles", [])
        if tile.get("uuid") in uuid_to_pid
    }

    coord_subsets: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for uuid, pid in uuid_to_pid.items():
        coords = uuid_to_coords.get(uuid) or []
        if not coords:
            continue
        unique_coords = {(int(x), int(y), int(z)) for x, y, z in coords}
        for coord in unique_coords:
            coord_subsets.setdefault(coord, set()).add(pid)

    subsets: dict[str, list[list[int]]] = {}
    coords_list = []
    counts_list = []
    max_count = 0
    for (x, y, z), pid_set in coord_subsets.items():
        pid_list = sorted(pid_set)
        subset_key = "|".join([f"{px},{py},{pz}" for px, py, pz in pid_list])
        if subset_key not in subsets:
            subsets[subset_key] = [[px, py, pz] for px, py, pz in pid_list]
        count = len(pid_list)
        max_count = max(max_count, count)
        coords_list.append([x, y, z, subset_key])
        counts_list.append([x, y, z, count])

    return {
        "coords": coords_list,
        "subsets": subsets,
        "subset_count": len(subsets),
        "counts": counts_list,
        "max_count": max_count,
        "shape": shape,
    }


@app.route("/api/getLoadStoreAllPrograms", methods=["POST"])
def get_load_store_all_programs():
    data = request.json or {}
    if global_data is None or raw_tensor_data is None:
        update_global_data()
    op_type = data.get("type")
    overall_key = data.get("overall_key")
    time_idx = data.get("time_idx")
    op_index = data.get("op_index")
    if not op_type or not overall_key or time_idx is None:
        return jsonify({"error": "Missing type, overall_key, or time_idx"}), 400
    op_type = str(op_type).strip().capitalize()
    if op_type not in {"Load", "Store"}:
        return jsonify({"error": "Unsupported type"}), 400
    try:
        time_idx = int(time_idx)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid time_idx"}), 400
    try:
        op_index = int(op_index) if op_index is not None else None
    except (TypeError, ValueError):
        op_index = None
    payload = _collect_load_store_program_subsets(
        op_type, overall_key, op_index, time_idx
    )
    return jsonify(payload)


@app.route("/api/load_overall", methods=["POST"])
def get_load_overall():
    data = request.json or {}
    key = data.get("key")
    time_idx = data.get("time_idx")
    if not key:
        return jsonify({"error": "Missing key"}), 400
    entry = load_overall_maps.get(key)
    if not entry:
        return jsonify({"error": "Overall data not found"}), 404

    if time_idx is not None:
        try:
            time_idx = int(time_idx)
            # Filter tiles by time_idx
            filtered_tiles = [
                t for t in entry.get("tiles", []) if t.get("time_idx") == time_idx
            ]
            # Return a copy of the entry with only the matching tiles
            return jsonify({**entry, "tiles": filtered_tiles})
        except (ValueError, TypeError):
            pass

    return jsonify(entry)


@app.route("/api/store_overall", methods=["POST"])
def get_store_overall():
    data = request.json or {}
    key = data.get("key")
    time_idx = data.get("time_idx")
    if not key:
        return jsonify({"error": "Missing key"}), 400
    entry = store_overall_maps.get(key)
    if not entry:
        return jsonify({"error": "Overall data not found"}), 404

    if time_idx is not None:
        try:
            time_idx = int(time_idx)
            # Filter tiles by time_idx
            filtered_tiles = [
                t for t in entry.get("tiles", []) if t.get("time_idx") == time_idx
            ]
            # Return a copy of the entry with only the matching tiles
            return jsonify({**entry, "tiles": filtered_tiles})
        except (ValueError, TypeError):
            pass

    return jsonify(entry)


def run_flask_with_cloudflared(port: int = 8000, tunnel_port: int | None = None):
    """
    Run the Flask app on a given port and expose it via Cloudflared.

    :param port: Local Flask port to bind to. Defaults to 8000.
    :param tunnel_port: Local tunnel control port for cloudflared. Defaults to port + 1.
    """
    cloudflared_port = port
    if tunnel_port is None:
        tunnel_port = cloudflared_port + 1
    tunnel_url = _run_cloudflared(cloudflared_port, tunnel_port)
    _server_state.set_public_url(tunnel_url)
    _server_state.set_local_port(cloudflared_port)
    print(f"Cloudflare tunnel URL: {tunnel_url}")
    app.run(host="0.0.0.0", port=cloudflared_port, debug=False, use_reloader=False)


def _is_interactive() -> bool:
    if getattr(sys, "ps1", None):
        return True
    if getattr(sys, "flags", None) and sys.flags.interactive:
        return True
    return "ipykernel" in sys.modules


def launch(share: bool = True, port: int | None = None, block: bool | None = None):
    """
    Launch the Triton-Viz Flask server.

    :param block: Whether to block the caller when share=True. Defaults to
                  True outside interactive sessions.
    """
    print("Launching Triton viz tool")
    default_port = 8000 if share else 5001
    actual_port = port or int(os.getenv("TRITON_VIZ_PORT", default_port))
    if block is None:
        block = share and not _is_interactive()

    if share:
        print("--------")
        flask_thread = threading.Thread(
            target=run_flask_with_cloudflared,
            args=(actual_port, None),
            daemon=False,
        )
        flask_thread.start()

        # Wait for the server to start
        time.sleep(5)

        # Try to get the tunnel URL by making a request to the local server
        local_url = f"http://localhost:{actual_port}"
        public_url = _server_state.get_public_url()

        try:
            # touch local server to ensure it's up
            _ = requests.get(local_url)
            print(f"Running on local URL:  {local_url}")
            if public_url:
                print(f"Running on public URL: {public_url}")
            print(
                "\nThis share link expires in 72 hours. For free permanent hosting and GPU upgrades, check out Spaces: https://huggingface.co/spaces"
            )
            print("--------")
        except requests.exceptions.RequestException:
            print("Setting up public URL... Please wait.")

        if block:
            try:
                flask_thread.join()
            except KeyboardInterrupt:
                pass

        return local_url, public_url
    else:
        print("--------")
        local_url = f"http://localhost:{actual_port}"
        print(f"Running on local URL:  {local_url}")
        print("--------")
        _server_state.set_local_port(actual_port)

        # For share=False, we want to block and keep the server running
        # This is the traditional behavior for local web servers
        app.run(host="0.0.0.0", port=actual_port, debug=True, use_reloader=False)

        return local_url, None


def get_last_public_url():
    """Return the last Cloudflare public URL created by launch(share=True)."""
    return _server_state.get_public_url()


@app.route("/shutdown", methods=["POST", "GET"])
def _shutdown():
    """Shutdown Flask development server (useful for notebooks)."""
    from flask import request as _req

    func = _req.environ.get("werkzeug.server.shutdown")
    if func is None:
        return jsonify(
            {"status": "error", "message": "Not running with the Werkzeug Server"}
        ), 400
    func()
    return jsonify({"status": "ok", "message": "Server shutting down..."})


def stop_server(port: int | None = None):
    """
    Stop the running Flask server by calling the /shutdown endpoint.
    If port is None, it will try the last used local port.
    """
    target_port = port or _server_state.get_local_port()
    if target_port is None:
        return False
    try:
        requests.post(f"http://127.0.0.1:{target_port}/shutdown", timeout=2)
        return True
    except Exception:
        return False
