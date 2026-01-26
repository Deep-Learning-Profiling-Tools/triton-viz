# Architecture

This document describes how the Triton-Viz frontend is structured and how data flows from the backend into the UI.
Terms are defined in `GLOSSARY.md` and used consistently below.

## Overview
- Backend (Flask) serves the HTML template and JSON APIs from `triton_viz/visualizer/interface.py`.
- Frontend TypeScript lives in `src/` and compiles to `triton_viz/static/` via `npm run build:frontend`.
- The browser entrypoint is `triton_viz/static/main.js`, compiled from `src/main.ts`.

## Data Flow
1. `src/visualization.ts` fetches `/api/data` and initializes the Active Program Workspace.
2. Program ID Controls update shared state and re-render the active op views.
3. Op Tabs switch between operation visualizers registered in `src/ops/registry.ts`.
4. Tensor View requests tensors via `/api/getLoadTensor`, `/api/getStoreTensor`, or op-specific endpoints.
5. Histogram Overlay posts to `/api/histogram` for value distributions.
6. Code Peek Panel posts to `/api/op_code` for source context.
7. Optional NKI panels fetch `/api/sbuf` to render scratch-buffer timelines.

## UI Layout
- Control Panel: Program ID Controls, Operation Controls, and Code Peek Panel.
- Active Program Workspace: main canvas that renders Op Tabs, Tensor View, Flow View, and legends.
- Legends and panels: Shape Legend, Value Legend, and Side Info Panel are managed by Tensor View.

## Key Modules
- `src/visualization.ts`: app bootstrap, state wiring, program slider setup, and global data fetch.
- `src/op_workspace.ts`: Op Tabs, view switching, and Code Peek Panel rendering.
- `src/tensor_view.ts`: Three.js scene setup, tensor meshes, highlights, and histogram overlay.
- `src/load_utils.ts`: camera controls and shared 3D scene helpers.
- `src/dimension_utils.ts`: CAD-style dimension lines and vector text overlays.
- `src/state.ts`: shared state for active program and toggle values.
- `src/logger.ts`: action and info logging with `logAction` and `logInfo`.
- `src/ops/*`: op-specific visualizers registered in the op registry.

## Build and Static Assets
- Source of truth: `src/` TypeScript modules.
- Output target: `triton_viz/static/` (including `main.js`).
- Command: `npm run build:frontend`.

## Diagnostics
- Action logs are emitted via `logAction` using canonical event names in `GLOSSARY.md`.
- Errors during `/api/data` fetch are surfaced in the control panel.
