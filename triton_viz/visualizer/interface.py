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


@app.route("/")
def index():
    update_global_data()
    return render_template("index.html")


@app.route("/debug")
def debug_page():
    update_global_data()
    return render_template("debug.html")


@app.route("/api/data")
def get_data():
    global global_data
    if global_data is None:
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


def run_flask_with_cloudflared(port: int = 8000, tunnel_port: int | None = None):
    """
    Run the Flask app on a given port and expose it via Cloudflared.

    :param port: Local Flask port to bind to. Defaults to 8000.
    :param tunnel_port: Local tunnel control port for cloudflared. Defaults to port + 1.
    """
    cloudflared_port = port
    if tunnel_port is None:
        tunnel_port = cloudflared_port + 1
    global last_public_url, last_local_port
    tunnel_url = _run_cloudflared(cloudflared_port, tunnel_port)
    last_public_url = tunnel_url
    last_local_port = cloudflared_port
    print(f"Cloudflare tunnel URL: {tunnel_url}")
    app.run(host="0.0.0.0", port=cloudflared_port, debug=False, use_reloader=False)


def launch(share: bool = True, port: int | None = None):
    """
    Launch the Triton-Viz Flask server.

    :param share: If True, expose via cloudflared (public URL). If False, serve locally only.
    :param port: Optional port override. Defaults to 8000 when share=True, else 5001.
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
        try:
            local_url = f"http://localhost:{actual_port}"
            # touch local server to ensure it's up
            _ = requests.get(local_url)
            public_url = last_public_url
            print(f"Running on local URL:  {local_url}")
            if public_url:
                print(f"Running on public URL: {public_url}")
            print(
                "\nThis share link expires in 72 hours. For free permanent hosting and GPU upgrades, check out Spaces: https://huggingface.co/spaces"
            )
            print("--------")
            return local_url, public_url
        except requests.exceptions.RequestException:
            print("Setting up public URL... Please wait.")
    else:
        print("--------")
        local_url = f"http://localhost:{actual_port}"
        print(f"Running on local URL:  {local_url}")
        print("--------")
        global last_local_port
        last_local_port = actual_port
        app.run(host="0.0.0.0", port=actual_port, debug=False, use_reloader=False)
        return local_url, None


def get_last_public_url():
    """Return the last Cloudflare public URL created by launch(share=True)."""
    return last_public_url


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
    target_port = port or last_local_port
    if target_port is None:
        return False
    try:
        requests.post(f"http://127.0.0.1:{target_port}/shutdown", timeout=2)
        return True
    except Exception:
        return False
