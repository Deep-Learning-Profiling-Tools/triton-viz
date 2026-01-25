# Task: migrate frontend to TypeScript with minimal build

Context: use TypeScript for the frontend so types are enforced and debugging is easier; keep the existing UI intact.

Review-Round: 0
Owner:
Deps: [frontend-2]
Touches: triton_viz/static/*, triton_viz/templates/index.html, package.json, tsconfig.json, triton_viz/ARCHITECTURE.md, ARCHITECTURE.md, MANUAL.md, README.md
Acceptance:
- frontend sources compile from TypeScript to the static bundle
- types exist for op records and API payloads
- UI behavior matches pre-migration behavior
Notes: Plan section is the low-level plan; use it verbatim without replacement.
Priority: P1

Plan:
- Overview: introduce a minimal TS build pipeline and migrate static JS modules to typed TS.
- Mental model: TS sources live in a `src/` tree and compile to `triton_viz/static/` assets loaded by the template.
- Interfaces / contracts: define shared types for `OpRecord`, `TensorPayload`, `ProgramCountsPayload`, `ProgramSubsetsPayload`, `OpCodePayload`, and `SbufTimelinePayload`.
- Files / functions / data structures: add `package.json`, `tsconfig.json`, and build script; convert JS modules to `.ts`; update `index.html` bundle path.
- Implementation steps:
  - add build tooling (esbuild or tsc) and config
  - move/convert existing modules to TS with type definitions
  - update imports/exports and bundle entrypoint
  - validate build output in `triton_viz/static/`
- Logging / observability: keep logger module and type its payloads; ensure all UI actions remain logged.
- Documentation: update `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` with build instructions and type locations.
- Tests: add a minimal JS/TS test runner (if missing) for registry/state/API tests; run a manual UI smoke test.
- Risk analysis: bundling path regressions and CDN import changes; mitigate by keeping external dependency URLs or bundling them consistently.
