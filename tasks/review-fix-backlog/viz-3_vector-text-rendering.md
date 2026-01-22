Context:
Replace rasterized 3js text blitting with text vector rendering.

Review-Round: 0
Owner:
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/matmul.js, triton_viz/static/load_utils.js
Acceptance:
- 3D text labels use vector rendering instead of rasterized sprites.
- Text remains readable at typical zoom levels without pixelation.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan:
- Plan: Overview
  - replace canvas sprite labels with vector text rendering for 3D labels, keeping readability at typical zoom levels.
  - ensure label styling stays consistent across background themes without pixelation.
- Plan: Mental model
  - labels are built in `addLabels` and `createCadDimension`, currently using canvas textures on sprites.
  - move label creation to a vector text pipeline (font loader + TextGeometry or Shapes), so labels are meshes in the scene graph and scale cleanly with camera zoom.
- Plan: Interfaces / contracts
  - preserve `addLabels(scene, globalTensor, sliceTensor, colorOrBg)` and `createCadDimension(scene, start, end, label, axis, color, options)` signatures to avoid call-site churn.
  - add a small shared text factory (e.g., `createVectorText`) that accepts text, color, size, and alignment and returns a `THREE.Object3D`.
- Plan: Files / functions / data structures
  - update `triton_viz/static/load_utils.js` to replace `addLabel` and `createTextSprite` with vector text mesh creation.
  - update `triton_viz/static/dimension_utils.js` to use the shared vector text helper for dimension labels.
  - update `triton_viz/static/load.js`, `triton_viz/static/store.js`, `triton_viz/static/matmul.js` only if import paths or label lifecycles need tweaks.
  - add a new helper module (e.g., `triton_viz/static/text_utils.js`) to host font loading, caching, and mesh creation to reduce conflicts.
- Plan: Implementation steps
  - inspect current label call sites and identify all canvas sprite usage for labels and dimension text.
  - add a vector text helper that loads a font once, caches it, and returns centered text meshes with configurable scale and color.
  - wire `addLabel`/`createCadDimension` to use the helper, keeping previous positioning offsets intact.
  - update label cleanup paths so vector text meshes are removed consistently on refresh.
  - validate label readability at typical zoom levels by tuning font size, depthTest, and render order.
- Plan: Logging / observability
  - add console logs when font load completes and when labels are rebuilt (include tensor name and label count).
  - log failures in font loading or text mesh creation with label text and op uuid if available.
- Plan: Documentation
  - update `ARCHITECTURE.md` with the new vector text helper and label rendering pipeline.
  - update `README.md` with any new asset or font loading requirement.
  - update `MANUAL.md` to note improved label rendering behavior and any user-visible toggles.
  - add a `triton_viz/static/ARCHITECTURE.md` if none exists and document text rendering there.
- Plan: Tests
  - add a lightweight visual regression test or snapshot for label creation if the project supports it.
  - add unit coverage for text helper input handling (text, color, size defaults).
  - ensure any user-facing label controls remain covered in MANUAL.md and tested.
- Plan: Risk analysis
  - risk: font loading latency or failure causing missing labels; mitigate with cached font and fallback behavior.
  - risk: depth sorting issues for text meshes; mitigate by disabling depthTest or setting renderOrder.
  - potential regressions: label refresh, hover behavior, and theme switching, especially in load/store/matmul views.
