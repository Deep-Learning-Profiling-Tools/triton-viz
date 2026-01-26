# Task: dev-only overlay for component names

Context: add an optional dev-only overlay to show internal component names without cluttering the UI for normal users.

Review-Round: 0
Owner:
Deps: [frontend-2]
Touches: triton_viz/static/visualization.js, triton_viz/static/visualizer.css, triton_viz/templates/index.html, triton_viz/ARCHITECTURE.md, ARCHITECTURE.md, MANUAL.md, README.md
Acceptance:
- dev overlay toggles via query param or key and is off by default
- overlay labels map to `data-component` attributes on UI roots
- normal UI remains unchanged when overlay is off
Notes: Plan section is the low-level plan; use it verbatim without replacement.
Priority: P2

Plan:
- Overview: add `data-component` markers and a dev-only overlay renderer that reads them.
- Mental model: overlay is a small DOM layer activated by a flag; it must not affect layout or styling.
- Interfaces / contracts: a simple toggle contract (e.g., `?dev=1` or keypress) with a stable set of component names.
- Files / functions / data structures: add `data-component` attributes in template/containers; add overlay module and minimal CSS for badges.

Plan: Overview
- Add dev-only overlay badges that display `data-component` names, defaulting to off.

Plan: Mental model
- Overlay is a floating, pointer-events-none layer that mirrors existing UI roots without changing layout or styling.

Plan: Interfaces / contracts
- Toggle contract: enable with `?dev=1` query param and a keyboard toggle (document the key).
- Stable component names map 1:1 to `data-component` values on root UI containers.

Plan: Files / functions / data structures
- `triton_viz/templates/index.html`: add `data-component` attributes on major UI roots.
- `triton_viz/static/visualization.js`: read toggle state, attach overlay, and refresh on resize/state changes.
- `triton_viz/static/visualizer.css`: badge styles and overlay positioning.
- Docs: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` with dev overlay behavior and toggle contract.

Plan: Implementation steps
- Identify top-level UI containers and assign canonical `data-component` values.
- Implement overlay renderer that scans for `data-component` nodes and draws badges.
- Add query-param and key toggle, storing state without affecting non-dev flows.
- Add lightweight CSS for badges (small caps, high-contrast background, no layout impact).
- Verify overlay does not intercept clicks or change layout when disabled.

Plan: Logging / observability
- Log toggle events with `logAction` for dev overlay enable/disable.

Plan: Documentation
- Document the toggle contract and component name list in the docs.

Plan: Tests
- Manual check: overlay off by default, toggles on/off, no layout shift.

Plan: Risk analysis
- Visual clutter or layout shifts; mitigate with absolute positioning and `pointer-events: none`.
- Implementation steps:
  - tag UI roots with `data-component` values
  - implement overlay toggle and renderer
  - ensure overlay cleanup on toggle off
- Logging / observability: log overlay toggles with mode and source.
- Documentation: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` with dev overlay usage.
- Tests: manual toggle test and a minimal unit test asserting badge presence when flag is enabled.
- Risk analysis: low risk; mitigate accidental production exposure by gating on explicit flag.
