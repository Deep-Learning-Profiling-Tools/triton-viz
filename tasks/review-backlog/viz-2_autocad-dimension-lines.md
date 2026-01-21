Context:
Replace the existing dimension markers with CAD-style dimension lines:
    - Draw Extension Lines perpendicular to the dimension axis at both terminal points. Optional: Add a 2-unit offset so they don't touch the object.
    - Draw a Dimension Line parallel to the object.
    - Place Arrowheads at the intersection of the dimension and extension lines.
    - Conditional Logic: If the distance between extension lines is less than [X] units, flip the arrows to the 'Outside' (pointing inward toward the extension lines). Otherwise, keep them 'Inside' (pointing outward)."
Make the dimension lines/numbers different colors and display each visualize tensors' shape (with color coding) somewhere on the screen.

Review-Round: 1
Owner:
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/matmul.js, triton_viz/static/visualizer.css
Acceptance:
- Dimension markers are replaced with AutoCAD-style dimension lines in tensor views.
- Dimension markers contain all components: extension lines, parallel dimension lines, arrowheads, conditional arrow flipping
- Dimension lines/numbers use distinct colors and tensor shapes are displayed with matching color coding.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan:
- Plan: Overview
  - Replace tensor dimension markers with CAD-style extension/dimension lines, arrowheads, and inside/outside flipping based on a distance threshold; show color-coded dimension numbers and on-screen tensor shapes in matching colors.
- Plan: Mental model
  - Dimension markers are rendered in the front-end tensor views (load/store/matmul) using Three.js plus DOM overlays; the plan is to compute screen-space endpoints from tensor bounds and render a shared CAD-dimension overlay per tensor.
  - A single helper should own geometry math (offsets, parallel lines, arrowhead placement, flip decision) so each view only supplies tensor bounds and axis orientation.
- Plan: Interfaces / contracts
  - Add a small JS helper API (e.g., `createCadDimensionOverlay({container, start, end, axis, offset, threshold, colors})`) returning a handle for cleanup; keep existing op payloads unchanged.
  - Use CSS variables for dimension colors/labels so light/dark themes stay consistent without changing external APIs.
- Plan: Files / functions / data structures
  - `triton_viz/static/load.js`, `triton_viz/static/store.js`, `triton_viz/static/matmul.js`: replace current dimension marker rendering with helper calls and wire in shape legend updates.
  - `triton_viz/static/load_utils.js` or a new `triton_viz/static/dimension_utils.js`: implement CAD dimension overlay builder and arrow-flip logic with a configurable threshold.
  - `triton_viz/static/visualizer.css`: add dimension line/label styles and legend layout/color tokens.
  - Data: define a color map per tensor (global/slice/matrix A/B/C) and a shared threshold constant for arrow flipping.
- Plan: Implementation steps
  - Locate current dimension marker creation in each view and document available tensor bounds/axes to map into a shared API.
  - Implement the CAD dimension helper (extension lines, parallel dimension line, arrowheads, inside/outside flip) and return a cleanup handle for re-rendering.
  - Replace existing markers in load/store/matmul with the helper and ensure arrow flip triggers when the distance falls below the threshold.
  - Add a small legend overlay listing tensor shapes with matching colors; update on view init and window resize.
  - Update CSS for dimension colors/labels and ensure contrast in both themes.
  - Add teardown logic so overlays are removed when re-rendering or switching ops.
- Plan: Logging / observability
  - Log dimension overlay creation/update at info level with op uuid, tensor name, axis, measured distance, and arrow mode (inside/outside).
  - Log resize-triggered overlay recalculations and legend updates to make UI changes traceable.
- Plan: Documentation
  - Update or create root `ARCHITECTURE.md` to describe the front-end annotation layer and how CAD dimension overlays are computed.
  - Add `triton_viz/ARCHITECTURE.md` to document static front-end modules and the new dimension helper.
  - Update `README.md` with the CAD-style dimension lines and color-coded shape legend behavior.
  - Update `MANUAL.md` to describe the new dimension visuals and shape legend (user-facing functionality).
- Plan: Tests
  - Add unit tests for the dimension helper's arrow-flip decision and line endpoint math using deterministic inputs.
  - Add an integration-style test (or minimal harness) that instantiates a tensor view and asserts the legend/overlay elements exist and update on resize.
  - Manual smoke test: open load/store/matmul views and verify arrow flip, color coding, and shape legend in both themes.
- Plan: Risk analysis
  - Parallel conflicts: other viz tasks may touch `load.js`, `store.js`, `matmul.js`, and `visualizer.css`; isolate shared math in a new helper to reduce merge conflicts.
  - Failure modes: incorrect screen-space transforms can misplace arrows/lines; mitigate by validating against known tensor bounds and logging distances.
  - Regressions: existing hover/selection overlays and resize handlers may break if overlays are not cleaned up; ensure cleanup on re-render and window resize.
