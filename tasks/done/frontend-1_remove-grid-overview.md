# Task: remove grid overview UI, keep active program workflow

Context: remove the click-to-open grid UI; default to opening program (0,0,0) and keep the Tensor/Flow views, control sidebar (Program IDs filters X/Y/Z, op controls, code peek panel), and shape legend unchanged.

Review-Round: 1
Owner: codex
Deps: [frontend-0]
Touches: triton_viz/static/visualization.js, triton_viz/static/gridblock.js, triton_viz/templates/index.html, triton_viz/static/visualizer.css, triton_viz/ARCHITECTURE.md, ARCHITECTURE.md, MANUAL.md, README.md
Acceptance:
- no grid overview canvas or click-to-open grid remains in the UI
- program (0,0,0) opens by default and Program IDs sliders update the active program
- op tabs, Tensor View, Flow View, code peek, and shape legend still work
Notes: Plan section is the low-level plan; use it verbatim without replacement.
Priority: P0

Plan:
- Overview: replace grid overview rendering with a lightweight active program controller that drives the op workspace directly.
- Mental model: the grid canvas and block UI are only for selection; the detail workspace and op visualizers are the real UI to preserve.
- Interfaces / contracts: preserve `visualization_data` key format and existing DOM IDs; keep code panel hooks until refactor.
- Files / functions / data structures: refactor `triton_viz/static/visualization.js` to remove KernelGrid; extract op workspace from `triton_viz/static/gridblock.js` into a new module (e.g., `op_workspace.js`); update `triton_viz/templates/index.html` and `triton_viz/static/visualizer.css` if canvas is removed.
- Implementation steps:
  - move op tab + visualization container logic into a reusable "op workspace" module
  - implement active program selection that defaults to (0,0,0) and updates on Program IDs sliders
  - remove grid drawing/hover/click code paths and any unused canvas DOM
  - remove z-slider wiring (unused) and update sidebar control logic accordingly
- Logging / observability: log program coordinate changes and op tab changes with program coord and op type.
- Documentation: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` to describe "Active Program" and remove grid/block terminology.
- Tests: manual smoke test for default open, slider-driven op changes, tab switching, Tensor/Flow render, and code peek loading.
- Risk analysis: risk of breaking op selection or code panel sync; mitigate by preserving data key parsing and reusing existing op rendering functions.
