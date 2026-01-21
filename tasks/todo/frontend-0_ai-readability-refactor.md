Context:
Refactor the frontend to optimize for AI agent readability: decouple objects/state, consolidate shared utils, make smaller functions. All visualizations (load/store/dot) must use the same shared implementation of:
    - base 3D tensor visualization
    - tensor coloring + colormaps based on value intensity (use Dot viz' current impl)
    - tensor dimension labeling (use Load viz' current impl)
    - sidebar widget addition (use Load viz' current impl)
    - Program ID sliders (use Load viz' current impl)
    - Value histograms (use Dot viz' current impl)
    - hover -> outline cube and show data (use Dot viz' current impl)
    - highlighting selection ranges of all program IDs (note: only applicable to Load/Store)

Review-Round: 0
Owner:
Deps: none
Touches: triton_viz/static/visualization.js, triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/gridblock.js, triton_viz/static/histogram.js, triton_viz/static/visualizer.css, triton_viz/templates/index.html, triton_viz/static/ts/
Acceptance:
- Frontend logic is migrated to TypeScript with explicit types for API payloads and view state.
- State mutations are isolated from rendering, with modularized render functions and smaller helpers.
- Shared utilities are consolidated into common modules to remove duplication.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan: TBD (low-level plan)
