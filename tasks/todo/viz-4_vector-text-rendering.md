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
Plan: TBD (low-level plan)
