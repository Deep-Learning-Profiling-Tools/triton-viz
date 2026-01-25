# Architecture

## Front-end Annotation Layer
The front-end utilizes Three.js for 3D tensor visualization and a custom annotation layer for dimension lines and legends.

### CAD Dimension Overlays
Dimension lines are rendered as CAD-style annotations:
- **Extension Lines**: Perpendicular lines from tensor boundaries.
- **Dimension Lines**: Parallel lines showing the span.
- **Arrowheads**: Located at intersections, with automatic inside/outside placement logic.
- **Text Rendering**: High-quality vector text rendering using Multi-channel Signed Distance Fields (MSDF) via `Troika-Three-Text`, ensuring readability at all zoom levels without pixelation.
- **Logic**: Implemented in `triton_viz/static/dimension_utils.js`.

### Shape Legend
A floating DOM overlay that lists all active tensors and their shapes, color-coded to match the 3D meshes.

### Active Program Workspace
Program ID sliders select the active program (X/Y/Z). The workspace renders op tabs, Tensor/Flow views, and code peek for that active program.
Logic lives in `triton_viz/static/op_workspace.js`, with wiring in `triton_viz/static/visualization.js`.

### Front-end TypeScript Build
Frontend sources live in `src/` and compile to `triton_viz/static/` via `npm run build:frontend`. The browser entrypoint is `src/main.ts`, which outputs `triton_viz/static/main.js` for `index.html`.

### Front-end Core Modules
- `triton_viz/static/api.js`: base-aware JSON client for frontend requests.
- `triton_viz/static/state.js`: single source of truth for active program/op/toggles.
- `triton_viz/static/ops/registry.js`: op visualizer registry; `ops/defaults.js` wires built-ins.
- `triton_viz/static/utils/dispose.js`: shared cleanup helper for listeners and timers.
- `triton_viz/static/logger.js`: centralized action logging.
