Context:
Consolidate shared functionality between different record (Load/Store/Dot) visualizations like:
- colorbar design (I prefer Dot's implementation)
- hover: highlighting cube outline, loading elements (I prefer Dot's implementation)
- actual 3D tensor visualization

Review-Round: 0
Owner:
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/matmul.js, triton_viz/static/load_utils.js, triton_viz/static/visualizer.css
Acceptance:
- Shared colorbar, hover behavior, and 3D tensor rendering are extracted and reused across Load/Store/Dot views.
- Dot's preferred implementations are used as the unified baseline.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan: TBD (low-level plan)
