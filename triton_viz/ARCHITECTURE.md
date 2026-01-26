# Triton-Viz Static Modules

- `load.js`: Handles load operation visualization.
- `store.js`: Handles store operation visualization.
- `matmul.js`: Handles matrix multiplication visualization.
- `op_workspace.js`: Active program workspace that renders op tabs and views.
- `ops/registry.js`: Registry for op visualizers with a shared create interface.
- `ops/defaults.js`: Built-in op visualizer registrations.
- `api.js`: Base-aware JSON client for frontend requests.
- `state.js`: Shared UI state for active program/op/toggles.
- `utils/dispose.js`: Cleanup helper for listeners and timers.
- `logger.js`: Lightweight action logging.
- `dimension_utils.js`: Shared utilities for CAD-style dimensioning, legends, and high-quality vector text rendering.
- `load_utils.js`: General 3D scene and tensor setup helpers.
- `visualization.js`: Fetches data, wires controls, and drives the active program workspace.
- `src/`: TypeScript sources that compile to `triton_viz/static/`.
- `static/main.js`: Browser entrypoint compiled from `src/main.ts`.
