# Glossary

This glossary defines the canonical UI terms, data types, and events used across the Triton-Viz frontend.

## UI Surfaces
- Control Panel: left sidebar with program controls, operation controls, and the code peek panel.
- Program ID Controls: sliders for Program Axes (X/Y/Z) plus the Reset button.
- Operation Controls: buttons for Color by Value, All Program IDs, and Value Histogram.
- Active Program Workspace: main canvas region driven by the active program selection.
- Op Tabs: tabs inside the workspace, one per operation in the active program.
- Tensor View: 3D tensor view with legends, highlights, and a side info panel.
- Flow View: flow diagram view when an op provides a flow visualization.
- Code Peek Panel: right-side panel in the control panel that renders code context.
- Theme Toggle: light/dark toggle in the control panel header.
- Dev Overlay: optional badge layer that labels `data-component` roots.

## Core Concepts
- Active Program: the currently selected program coordinate triple (x, y, z).
- Program Axes: the three axes used for program IDs: x, y, z.
- Toggle State: shared on/off UI state for colorize, histogram, all programs, and code panel visibility.
- Data Component: `data-component` attribute on a UI root used by the Dev Overlay.

## Data Types and Payloads
- OpRecord: operation metadata for a single op, including shapes and UUIDs.
- TensorPayload: tensor values, shape, min/max, and optional highlights for the Tensor View.
- TensorHighlights: optional highlight region and sampled data for a tensor.
- ProgramCountsPayload: per-program counts for the all-programs view.
- ProgramSubsetsPayload: optional subsets and counts for filtered program selections.
- OpCodePayload: source code context for an op (filename, lines, highlight line).
- SbufTimelinePayload: scratch-buffer usage timeline and capacity details.
- HistogramPayload: histogram statistics for tensor values (counts, edges, min, max, sample sizes).
- ApiErrorPayload: error-only response payload for failed requests.

## API Endpoints
- GET /api/data: top-level dataset used to initialize the UI and visualizations.
- POST /api/op_code: fetch code context for a specific op UUID and frame.
- POST /api/histogram: return histogram stats for the active tensor selection.
- POST /api/getLoadTensor, /api/getStoreTensor, /api/getMatmulA/B/C: fetch tensor values for the active op.
- POST /api/getLoadStoreAllPrograms: counts for the all-programs view.
- GET /api/sbuf?device=...: scratch-buffer timeline data for NKI flows.

## Events and Logs
Events are logged with `logAction` and include an action name plus a details object.
- program_reset: reset program ID sliders to the default selection.
- program_slider: user changed a program axis slider (axis, value).
- toggle_colorize: toggled Color by Value (next: boolean).
- toggle_histogram: toggled Value Histogram (next: boolean).
- toggle_all_programs: toggled All Program IDs (next: boolean).
- code_peek_toggle: toggled the code peek panel (visible, uuid, source).
- dev_overlay_toggle: toggled the Dev Overlay (enabled, source).

## Canonical File Mapping
- frontend/core/visualization.ts: UI bootstrap, data fetch, and control wiring.
- frontend/components/op_workspace.ts: op tabs, workspace layout, and code peek panel.
- frontend/components/tensor_view.ts: 3D tensor rendering and histogram overlay integration.
- frontend/utils/dimension_utils.ts: CAD-style dimension lines and vector text.
- frontend/core/state.ts: shared active program and toggle state.
- frontend/core/logger.ts: action and info logging.
