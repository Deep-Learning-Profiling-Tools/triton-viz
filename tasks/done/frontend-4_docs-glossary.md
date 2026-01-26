# Task: documentation overhaul with glossary

Context: document the frontend cleanly for AI agents by adding a canonical glossary and rewriting architecture/manual docs to match the UI and data flow.

Review-Round: 0
Owner: codex
Deps: [frontend-3]
Touches: GLOSSARY.md, ARCHITECTURE.md, MANUAL.md, README.md, triton_viz/ARCHITECTURE.md
Acceptance:
- GLOSSARY.md defines canonical UI terms, data types, and events
- ARCHITECTURE.md and MANUAL.md match the current UI and code structure
- README.md includes concise build/run guidance
Notes: Plan section is the low-level plan; use it verbatim without replacement.
Priority: P1

Plan:
- Overview: add a glossary and align all docs to the current UI, naming, and data flow.
- Mental model: ARCHITECTURE explains how it works; MANUAL explains how to use it; GLOSSARY anchors naming and term mapping.
- Interfaces / contracts: document public payload shapes and event names in the glossary and architecture docs.
- Files / functions / data structures: create `GLOSSARY.md`; rewrite `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` to use glossary terms.
- Implementation steps:
  - draft glossary with UI terms, DOM anchors, and type names
  - update ARCHITECTURE to reflect module boundaries and data flow
  - update MANUAL to reflect user-visible features only
  - update README with build/run and location of docs
- Logging / observability: document logged user actions and event payloads in ARCHITECTURE or glossary.
- Documentation: this task is documentation; ensure all listed docs are updated and cross-linked.
- Tests: doc checklist to ensure glossary terms match code identifiers and DOM `data-component` attributes.
- Risk analysis: risk of term drift; mitigate by declaring glossary as canonical and referencing it from other docs.

Plan: Overview
- Restate acceptance: deliver a canonical `GLOSSARY.md`, align `ARCHITECTURE.md` and `MANUAL.md` with current UI/data flow, and add concise build/run guidance in `README.md`.

Plan: Mental model
- Treat `GLOSSARY.md` as the single source of truth for UI terminology, data types, and events; the architecture doc explains system structure while the manual focuses on user workflows.

Plan: Interfaces / contracts
- Enumerate public payload shapes, event names, and DOM `data-component` identifiers in the glossary, and reference them from architecture notes to keep terminology consistent.

Plan: Files / functions / data structures
- Create `GLOSSARY.md`; rewrite `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` to adopt glossary terms and cross-links.
- If any touched subdirectory lacks `ARCHITECTURE.md`, add it and summarize how it fits the glossary and system flow.

Plan: Implementation steps
- Audit current UI and data flow by scanning frontend sources to extract terms, DOM anchors, and event payloads for the glossary.
- Draft `GLOSSARY.md` with categories (ui surfaces, components, data types, events) and explicit mappings to code identifiers.
- Update `ARCHITECTURE.md` to describe module boundaries, data flow, and how glossary terms map to runtime structures.
- Update `MANUAL.md` to describe only user-visible behaviors and controls, referencing glossary terms.
- Update `README.md` with concise build/run steps and links to the glossary, architecture, and manual docs.

Plan: Logging / observability
- Document every user action that emits logs (clicks, keyboard shortcuts, background tasks) and include expected payload keys or identifiers per action.

Plan: Documentation
- Ensure `ARCHITECTURE.md`, `MANUAL.md`, `README.md`, and `triton_viz/ARCHITECTURE.md` are updated and cross-linked, with `GLOSSARY.md` referenced as canonical.

Plan: Tests
- Add a doc checklist that verifies glossary terms match code identifiers and DOM `data-component` values, and confirm every user-facing feature listed in the manual is documented.

Plan: Risk analysis
- Parallel conflicts: other frontend tasks may edit docs or rename terms; mitigate by coordinating glossary ownership and reusing exact term strings.
- Failure modes: inconsistent naming between docs and code or missing event payloads; mitigate with a final glossary-to-code audit pass.
- Regression risk: users rely on MANUAL guidance; ensure removed or renamed UI features are reflected consistently across all docs.
