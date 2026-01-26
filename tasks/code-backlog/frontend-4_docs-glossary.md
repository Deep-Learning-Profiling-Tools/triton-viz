# Task: documentation overhaul with glossary

Context: document the frontend cleanly for AI agents by adding a canonical glossary and rewriting architecture/manual docs to match the UI and data flow.

Review-Round: 0
Owner:
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
