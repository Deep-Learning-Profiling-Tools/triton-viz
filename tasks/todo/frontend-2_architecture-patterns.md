# Task: add registry/state/api/disposables and DRY tensor view

Context: make the frontend maintainable by consolidating shared logic, introducing a visualizer registry, a single state store, an API client layer, and cleanup helpers; keep all current UI behavior intact.

Review-Round: 0
Owner:
Deps: [frontend-1]
Touches: triton_viz/static/tensor_view.js, triton_viz/static/visualization.js, triton_viz/static/gridblock.js, triton_viz/static/nki.js, triton_viz/static/histogram.js, triton_viz/static/ui_helpers.js, triton_viz/ARCHITECTURE.md, ARCHITECTURE.md, MANUAL.md, README.md
Acceptance:
- op visualizer dispatch uses a registry with a consistent interface
- all fetch calls are routed through a shared API client
- active program/op/toggles live in a single source of truth state module
Notes: Plan section is the low-level plan; use it verbatim without replacement.
Priority: P1

Plan:
- Overview: introduce core patterns (registry, state store, API client, disposer) and remove duplicated logic without changing UX.
- Mental model: op workspace and visualizers consume shared state and API helpers; cleanup is centralized to prevent leaks.
- Interfaces / contracts: define a visualizer interface `create(container, op, viewState) -> dispose`; keep existing payload shapes intact.
- Files / functions / data structures: add modules like `api.js`, `state.js`, `ops/registry.js`, `utils/dispose.js`; refactor `tensor_view.js` colormap paths into a single mode-based function.
- Implementation steps:
  - build API client module and replace direct `fetch()` usage
  - add registry module and switch op dispatch to registry lookups
  - create state store for active program/op/toggles and wire UI updates
  - add disposal helper and route listeners/intervals through it
  - consolidate colormap loops in `tensor_view.js`
- Logging / observability: add a small logger helper; log all user actions (sliders, toggles, tab clicks, code peek) with identifiers.
- Documentation: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` to describe module boundaries and registry/state flow.
- Tests: add unit tests for registry dispatch, state updates, and API error handling; manual UI smoke test for toggles and op tabs.
- Risk analysis: potential regressions in event wiring and cleanup; mitigate by staging refactors and keeping registry interface backward compatible.
