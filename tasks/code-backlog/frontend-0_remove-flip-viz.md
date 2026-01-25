# Task: remove flip visualization

Context: "remove the flip viz" while keeping all other visible UI (Tensor/Flow view, control sidebar, Program IDs filters, op controls, code peek panel, shape legend) unchanged.

Review-Round: 0
Owner: codex
Deps: none
Touches: triton_viz/static/flip.js, triton_viz/static/flip_3d.js, triton_viz/ARCHITECTURE.md, ARCHITECTURE.md, MANUAL.md, README.md
Acceptance:
- flip visualization modules are removed and the app loads without errors
- no imports/references to flip visualization remain in static JS or docs
- existing Tensor/Flow views and sidebar controls remain unchanged
Notes: Plan section is the low-level plan; use it verbatim without replacement.
Priority: P1

Plan:
- Overview: remove flip visualization modules and any references; keep remaining UI behavior unchanged.
- Mental model: flip visualizations are standalone 3D overlays; if no modules import them, deletion is safe.
- Interfaces / contracts: none; no public API changes expected.
- Files / functions / data structures: delete `triton_viz/static/flip.js` and `triton_viz/static/flip_3d.js`; scan `triton_viz/static/` for imports; update docs in `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, `triton_viz/ARCHITECTURE.md` if they mention flip.
- Implementation steps:
  - remove flip JS files and any direct imports
  - run a quick scan for "flip" references in static modules and docs
  - verify template/CSS has no flip references
- Logging / observability: no new user actions; ensure no flip-specific logging remains referenced.
- Documentation: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` to remove flip mentions.
- Tests: manual smoke test the UI loads and Tensor/Flow views still open; confirm no 404s for deleted assets.
- Risk analysis: low risk; primary risk is stray imports or doc drift; mitigate via search before deletion.
