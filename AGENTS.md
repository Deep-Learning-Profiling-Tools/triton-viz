# Repository Guidelines

## Project Structure & Module Organization

Triton-Viz is a Python package with a TypeScript frontend. Core code lives in `triton_viz/`: `core/` handles tracing and client lifecycle, `clients/` contains visualizer/profiler/sanitizer logic, and `frontends/`, `transformers/`, and `visualizer/` cover DSL integration, AST rewrites, and visualizer APIs. Frontend source is in `frontend/`; built assets are copied into `triton_viz/static/` and `triton_viz/templates/`. Tests live in `tests/unit/`, `tests/end_to_end/`, `tests/frontend/`, and `tests/nki/`. Examples are in `examples/`; site docs are in `docs/`.

## Build, Test, and Development Commands

- `uv sync --extra test`: install Python test dependencies.
- `uv sync --extra nki --extra test`: install NKI plus test dependencies; specify all extras in one command.
- `uv run pytest tests/`: run the default Python suite, excluding tests marked `nki`.
- `uv run pytest tests/ -m ""`: run all Python tests, including NKI tests when dependencies are installed.
- `npm install`: install frontend tooling.
- `npm run build:frontend`: compile TypeScript and copy frontend package assets.
- `npm run test:frontend`: build the frontend and run Node-based frontend tests.

## Coding Style & Naming Conventions

Use Python 3.10+ and TypeScript modules. Follow file-local style: 4-space indentation for Python and descriptive snake_case names for functions, variables, and test helpers. Python tests are discovered from `*.py`, classes ending in `Test`, and functions named `test_*`. Ruff is configured in `pyproject.toml` with `E731` ignored; run pre-commit when available. Keep generated package assets in sync when editing `frontend/`.

## Testing Guidelines

Prefer focused tests near the behavior being changed: unit tests in `tests/unit/`, kernel or CLI-visible behavior in `tests/end_to_end/`, and frontend tests in `tests/frontend/*.mjs`. Default pytest skips NKI tests; use `-m nki` for only NKI tests or `-m ""` for the full suite. For sanitizer work, cover records, reports, CLI behavior, and launch lifecycle where applicable.

## Commit & Pull Request Guidelines

Recent commits use concise imperative summaries, often with prefixes such as `[FIX]`, `[FEAT]`, or `[REFACTOR]`, and PR numbers added by GitHub, for example `[FIX] Handle stride-0 expanded tensors in sanitizer (#327)`. Keep commits scoped to one logical change. Pull requests should explain the user-visible change, list test commands run, link related issues, and include screenshots or trace examples when frontend or visualizer output changes.

## Security & Configuration Tips

Do not commit local traces, credentials, or generated debug artifacts. Runtime behavior is controlled by environment variables such as `TRITON_VIZ_VERBOSE`, `TRITON_VIZ_PORT`, `ENABLE_SANITIZER`, `ENABLE_PROFILER`, and `SANITIZER_ENABLE_FAKE_TENSOR`; document any new variables in `README.md`.
