import threading
from flask import Flask, render_template, jsonify, request
from .analysis import analyze_records
from .draw import get_visualization_data
import os
import torch
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
precomputed_c_values = {}
current_fullscreen_op = None
last_public_url = None
last_local_port = None


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


def precompute_c_values(op_data):
    input_data = op_data["input_data"]
    other_data = op_data["other_data"]
    rows, inner_dim = input_data.shape
    cols = other_data.shape[1]

    precomputed = {}
    for i in range(rows):
        for j in range(cols):
            precomputed[(i, j)] = [0] * (inner_dim + 1)
            for k in range(1, inner_dim + 1):
                precomputed[(i, j)][k] = torch.dot(
                    input_data[i, :k], other_data[:k, j]
                ).item()

    return precomputed


def update_global_data():
    global global_data, raw_tensor_data, precomputed_c_values

    # Collect all records from launches
    from ..core.trace import launches

    all_records = []
    for launch in launches:
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

    # Precompute C values for each Dot operation
    precomputed_c_values = {}
    for uuid, op_data in raw_tensor_data.items():
        if "input_data" in op_data and "other_data" in op_data:
            precomputed_c_values[uuid] = precompute_c_values(op_data)

    # Convert analysis_data to a dictionary format similar to pandas DataFrame.to_dict()
    # analysis_data is a list of lists where each inner list contains [metric, value] pairs
    df_dict = {"Metric": [], "Value": []}
    for record in analysis_data:
        for metric, value in record:
            df_dict["Metric"].append(metric)
            df_dict["Value"].append(value)

    global_data["analysis"] = df_dict


def _safe_read_file_segment(filename: str, lineno: int, context: int = 8):
    try:
        # only allow files under current working dir for safety
        cwd = os.path.realpath(os.getcwd())
        path = os.path.realpath(filename)
        if not path.startswith(cwd):
            return None
        start = max(1, lineno - context)
        end = lineno + context
        lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if i > end:
                    break
                lines.append({"no": i, "text": line.rstrip("\n")})
        return {
            "filename": path,
            "lineno": lineno,
            "start": start,
            "end": end,
            "highlight": lineno,
            "lines": lines,
        }
    except Exception:
        return None


@app.route("/")
def index():
    update_global_data()
    return render_template("index.html")


@app.route("/debug")
def debug_page():
    update_global_data()
    return render_template("debug.html")


@app.route("/calibrate")
def calibrate_page():
    # A minimal page to calibrate mouse picking dx/dy
    return render_template("calibrate.html")


@app.route("/api/data")
def get_data():
    global global_data
    update_global_data()
    return jsonify(global_data)


@app.route("/api/update_data")
def update_data():
    update_global_data()
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
    frame_idx = int(data.get("frame_idx", 0))
    context = int(data.get("context", 8))
    if not uuid or raw_tensor_data is None or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    tb_list = raw_tensor_data[uuid].get("tracebacks") or []
    if not tb_list:
        return jsonify({"error": "Traceback not available"}), 200
    # Heuristic: pick the BEST user frame under CWD (closest to op)
    cwd = os.path.realpath(os.getcwd())

    def _score(tb: dict) -> int:
        fn = tb.get("filename") or ""
        line = (tb.get("line") or "").strip()
        name = tb.get("name") or ""
        p = os.path.realpath(fn)
        score = 0

        # 强优先：明确的 Triton 语言内核操作符
        if "tl.load" in line or "tl.store" in line:
            score += 100
        elif "tl." in line:
            score += 50

        # 位置相关：项目代码优先，三方/框架降权
        if p.startswith(cwd):
            score += 5
        if any(
            s in p
            for s in ["site-packages", "triton_viz/", "triton/", "runpy.py", "IPython"]
        ):
            score -= 10

        # 语义相关：函数名看起来像 kernel 的加分
        if name.endswith("_kernel") or "kernel" in name:
            score += 3

        # 体验相关：examples 目录小幅加分
        if "examples" in p:
            score += 1

        return score

    # Prefer frames from tail (closest to current) and highest score
    best = None
    best_score = -(10**9)
    for tb in reversed(tb_list):
        sc = _score(tb)
        if sc > best_score:
            best, best_score = tb, sc
    chosen = best if best is not None else None
    if chosen is None:
        # fallback: use requested frame or last non-<string> frame
        frame_idx = max(0, min(frame_idx, len(tb_list) - 1))
        chosen = tb_list[frame_idx]
        for tb in reversed(tb_list):
            fn = tb.get("filename") or ""
            if not fn.startswith("<"):
                chosen = tb
                break
    tb = chosen
    filename = tb.get("filename")
    lineno = int(tb.get("lineno", 0))
    line_of_code = tb.get("line")
    seg = _safe_read_file_segment(filename, lineno, context)
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
        # Compute C on CPU
        c = a @ b
        c_cpu = c.cpu()
        cmin = float(c_cpu.min().item())
        cmax = float(c_cpu.max().item())
        return jsonify(
            {
                "shape": list(c_cpu.shape),
                "min": cmin,
                "max": cmax,
                "values": c_cpu.numpy().tolist(),
            }
        )
    except Exception as e:
        return jsonify({"error": f"MatMul compute failed: {e}"}), 200


@app.route("/api/getMatmulVectors", methods=["POST"])
def get_matmul_vectors():
    """Return A[row, :] and B[:, col] for a given Dot op.

    Request JSON: { uuid, row, col }
    Response JSON: {
        "row": int,
        "col": int,
        "a_row": [float, ...],
        "b_col": [float, ...],
        "k": int
    }
    """
    global raw_tensor_data
    data = request.json or {}
    uuid = data.get("uuid")
    row = int(data.get("row", 0))
    col = int(data.get("col", 0))
    if not uuid or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404
    op = raw_tensor_data[uuid]
    a = op.get("input_data")
    b = op.get("other_data")
    if a is None or b is None:
        return jsonify({"error": "MatMul tensors not available"}), 200
    try:
        a_row = a[row, :].cpu().numpy().tolist()
        b_col = b[:, col].cpu().numpy().tolist()
        k = len(a_row)
        return jsonify({"row": row, "col": col, "a_row": a_row, "b_col": b_col, "k": k})
    except Exception as e:
        return jsonify({"error": f"MatMul vectors failed: {e}"}), 200


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
            value = 0.0
            if op_data["dims"] == 3:
                value = op_data["global_tensor"][x, y, z].item()
            elif op_data["dims"] == 2:
                value = op_data["global_tensor"][x, y].item()
            elif op_data["dims"] == 1:
                value = op_data["global_tensor"][x].item()

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
    data = request.json
    uuid = data.get("uuid")

    if uuid is None or uuid not in raw_tensor_data:
        return jsonify({"error": "Operation not found"}), 404

    op_data = raw_tensor_data[uuid]
    if "global_tensor" not in op_data:
        return jsonify({"error": "Global tensor data not found"}), 200

    t = op_data["global_tensor"].cpu()
    try:
        t_min = float(t.min().item())
        t_max = float(t.max().item())
    except Exception:
        # In case of empty tensor
        t_min = 0.0
        t_max = 0.0

    return jsonify(
        {
            "shape": list(t.shape),
            "dims": len(t.shape),
            "min": t_min,
            "max": t_max,
            "values": t.numpy().tolist(),
        }
    )


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


def launch(share: bool = True, port: int | None = None):
    """
    Launch the Triton-Viz Flask server.

    """
    print("Launching Triton viz tool")
    default_port = 8000 if share else 5001
    actual_port = port or int(os.getenv("TRITON_VIZ_PORT", default_port))

    if share:
        print("--------")
        flask_thread = threading.Thread(
            target=run_flask_with_cloudflared, args=(actual_port, None)
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

        return local_url, public_url
    else:
        print("--------")
        local_url = f"http://localhost:{actual_port}"
        print(f"Running on local URL:  {local_url}")
        print("--------")
        _server_state.set_local_port(actual_port)

        # Run Flask in a background thread so callers can continue (non-blocking)
        def _run_local():
            app.run(host="0.0.0.0", port=actual_port, debug=True, use_reloader=False)

        flask_thread = threading.Thread(target=_run_local, daemon=True)
        flask_thread.start()
        # Give the server a moment to bind the port
        time.sleep(0.5)
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
