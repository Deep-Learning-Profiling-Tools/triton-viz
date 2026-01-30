# Visualizer Architecture

This folder owns the Flask server and the data-prep pipeline that turns traced records into frontend-ready payloads.

## File map (start here)
- `triton_viz/visualizer/interface.py`: Flask app, API endpoints, and data caching.
- `triton_viz/visualizer/draw.py`: record-to-visualization conversion and tensor extraction.
- `triton_viz/visualizer/analysis.py`: lightweight metrics over records.
- `triton_viz/visualizer/tooltip.py`: tooltip payload helpers.

## Terms defined here
- **visualization data**: per-program op payloads used to render the UI.
- **raw tensor data**: per-op tensor/value payloads used for histogram and details.
- **launch snapshot**: (launch count, record count) used to skip redundant recomputation.
- **sbuf events**: scratch-buffer usage events emitted by NKI flows.

## Subsystems used
- **Core trace state**: `triton_viz.core.trace.launches` is the source of truth.
- **Frontend contracts**: payload shapes are consumed by `frontend/` and documented in `frontend/GLOSSARY.md`.
- **Numpy/Torch**: used for payload computation and sampling.

## Main logic flows
- **Data refresh**:
  - `update_global_data()` collects all launch records.
  - `analysis.analyze_records()` builds a metrics table.
  - `draw.get_visualization_data()` constructs op payloads + raw tensor payloads.
  - Results are cached and served by `/api/data` and related endpoints.
- **On-demand data**:
  - Histogram and tensor endpoints pull from `raw_tensor_data`.
  - `/api/op_code` resolves tracebacks to source snippets.

## Extension crash-course: add a new visualization
1. Add a new `Op` record (core) and emit it from a client.
2. Update `draw.py` to translate that record into an op payload and any raw tensor data.
3. Update the frontend ops registry to render the new payload.
4. If needed, add an API endpoint in `interface.py` for extra data.

## Required vs optional patterns
- **Required**: op payloads must include a stable `uuid` and a `type` the frontend understands.
- **Optional**: raw tensor payloads can be omitted for ops that do not render tensors.

## Gotchas and invariants
- The visualizer currently assumes "last launch only" in `draw.py`.
- `update_global_data()` only recomputes if the launch snapshot changed.
- `raw_tensor_data` is keyed by op UUID; keep UUIDs stable across derived payloads.

## Debug recipe
- Hit `/api/data` first and inspect the `visualization_data` payload.
- For missing ops, verify they appear in `draw.collect_launch()` and are not filtered.
