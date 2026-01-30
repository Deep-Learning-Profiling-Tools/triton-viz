# Frontend Architecture

The frontend renders visualization payloads from the Flask server and provides UI controls for exploring kernels.

## File map (start here)
- `frontend/main.ts`: entry point, imports visualization bootstrap.
- `frontend/core/visualization.ts`: app bootstrap, controls, and data fetch.
- `frontend/core/state.ts`: shared UI state for active program and toggles.
- `frontend/components/`: UI components (workspace, tensor view, histograms).
- `frontend/ops/`: per-op renderers and registry.
- `frontend/utils/`: rendering helpers and geometry utilities.
- `frontend/types/`: shared types and payload contracts.
- `frontend/GLOSSARY.md`: canonical UI terms and payload names.

## Terms defined here
- **op renderer**: a function/module that knows how to render a specific op payload.
- **active program**: selected (x, y, z) program id that drives the workspace.
- **toggle state**: shared UI flags (heatmap, histogram, all programs, code peek).
- **op workspace**: the main surface containing op tabs and tensor view.

## Subsystems used
- **API client**: `core/api.ts` fetches server endpoints.
- **State + controls**: `core/state.ts` and `core/visualization.ts` coordinate UI.
- **Renderer registry**: `frontend/ops/registry.ts` binds op type -> renderer.
- **Types**: `frontend/types/types.ts` defines payload shapes.

## Main logic flows
- **Bootstrap**:
  - `main.ts` imports `core/visualization.ts`.
  - `visualization.ts` builds controls and fetches `/api/data`.
- **Render**:
  - `OpWorkspace` creates op tabs for the active program.
  - Each op tab looks up its renderer in the ops registry.
  - Tensor view fetches extra data (histograms, tensor values) on demand.
- **Interaction**:
  - Control toggles update global state and re-render.
  - Code peek fetches `/api/op_code` for the active op.

## Extension crash-course: add a new op renderer
1. Add the payload shape to `frontend/types/types.ts` (or reuse existing types).
2. Implement a renderer in `frontend/ops/<name>.ts`.
3. Register it in `frontend/ops/registry.ts`.
4. If the renderer needs new API data, add a helper in `frontend/core/api.ts`.
5. Update `frontend/GLOSSARY.md` with new payload terms if needed.

## Required vs optional patterns
- **Required**: registry entries must match `type` values emitted by the server.
- **Optional**: renderer can opt into histogram or all-programs controls.

## Gotchas and invariants
- The UI assumes `visualization_data` is grouped by program id strings.
- `window.setOpControlHandlers` is used by op renderers to bind controls.
- If an op renderer is missing, the tab still renders but shows a fallback.

## Debug recipe
- Start with `/api/data` and confirm the op `type` matches your registry key.
- Use the dev overlay (`Ctrl+Shift+D`) to locate component roots.
