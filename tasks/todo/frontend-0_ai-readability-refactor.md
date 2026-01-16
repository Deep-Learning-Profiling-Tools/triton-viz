Context:
Refactor the frontend to optimize for AI agent readability: decouple objects/state, use TypeScript instead of JavaScript, consolidate shared utils, make smaller functions.

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
