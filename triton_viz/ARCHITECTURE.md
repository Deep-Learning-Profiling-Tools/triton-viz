# Triton-Viz Frontend Architecture

This file summarizes the static frontend modules and how they map to the UI described in `GLOSSARY.md`.

## Static Module Map
- `static/main.js`: browser entrypoint compiled from `src/main.ts`.
- `static/visualization.js`: bootstraps the UI and fetches `/api/data`.
- `static/op_workspace.js`: Active Program Workspace, Op Tabs, and Code Peek Panel.
- `static/tensor_view.js`: Tensor View rendering, legends, and histogram overlay.
- `static/load_utils.js`: camera controls, scene helpers, and tensor mesh setup.
- `static/dimension_utils.js`: CAD-style dimension lines and vector text.
- `static/ops/registry.js`: op visualizer registry.
- `static/ops/defaults.js`: built-in op visualizers.
- `static/logger.js`: action logging with `logAction`.
- `static/state.js`: shared Active Program and toggle state.
- `static/utils/dispose.js`: listener cleanup utilities.
- `static/histogram.js`: histogram overlay UI and `/api/histogram` calls.
- `static/sbuf_panel.js`: NKI scratch-buffer timeline panel.

## Build Pipeline
- Source: `src/*.ts`.
- Output: `triton_viz/static/*.js`.
- Command: `npm run build:frontend`.

## Data Sources
- `/api/data` initializes the UI and op registry data.
- `/api/op_code` feeds the Code Peek Panel.
- `/api/histogram` feeds the histogram overlay.
- Tensor endpoints (for example `/api/getLoadTensor`) feed the Tensor View.
