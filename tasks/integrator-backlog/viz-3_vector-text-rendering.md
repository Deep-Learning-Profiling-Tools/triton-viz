Context:
Replace rasterized 3js text blitting with text vector rendering.

Review-Round: 2
Owner: Gemini
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/matmul.js, triton_viz/static/load_utils.js, triton_viz/static/dimension_utils.js, triton_viz/static/flip_3d.js, triton_viz/static/flip.js
Acceptance:
- 3D text labels use vector rendering instead of rasterized sprites.
- Text remains readable at typical zoom levels without pixelation.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan:
1. Create a centralized `createVectorText` utility in `dimension_utils.js` using `Troika-Three-Text` for MSDF-based sharp vector text.
2. Implement auto-billboarding in `createVectorText` using `onBeforeRender`.
3. Update `load_utils.js` to use `createVectorText` in `addLabel`.
4. Update `dimension_utils.js` to use `createVectorText` in `createCadDimension`.
5. Update `flip.js` and `flip_3d.js` to use `createVectorText` for their labels.
6. Verify that all 3D labels are now sharp and non-pixelated.
