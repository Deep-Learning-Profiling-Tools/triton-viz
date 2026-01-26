# Task: dev-only overlay for component names

Context: add an optional dev-only overlay to show internal component names without cluttering the UI for normal users.

Review-Round: 0
Owner: codex
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
- Implementation steps:
  - tag UI roots with `data-component` values
  - implement overlay toggle and renderer
  - ensure overlay cleanup on toggle off
- Logging / observability: log overlay toggles with mode and source.
- Documentation: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` with dev overlay usage.
- Tests: manual toggle test and a minimal unit test asserting badge presence when flag is enabled.
- Risk analysis: low risk; mitigate accidental production exposure by gating on explicit flag.
