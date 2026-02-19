# Contributing to Triton-Viz

## Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
```

2. Create a virtual environment and install dependencies:

```bash
uv sync --extra test
```

## Running Tests

Use `uv run` to run tests, **not** `python -m`:

```bash
# Run all tests
uv run pytest tests/ -v

# Run only unit tests
uv run pytest tests/unit/ -v

# Run only end-to-end tests
uv run pytest tests/end_to_end/ -v

# Run a specific test file
uv run pytest tests/unit/test_example.py -v
```

## Code Style

This project uses [pre-commit](https://pre-commit.com/) hooks to enforce code style. Install them before committing:

```bash
pre-commit install
```

The hooks include:

- **Ruff** - Python linting and formatting (line length: 120)
- **mypy** - Static type checking
- **clang-format** - C/C++/CUDA formatting
- **codespell** - Spell checking
- **trailing-whitespace**, **end-of-file-fixer**, etc.

You can run all hooks manually:

```bash
pre-commit run -a
```

## Project Structure

```
triton_viz/
  core/         # Core framework (client base, callbacks, data types, config, patch)
  clients/      # Client implementations (Profiler, Sanitizer, Tracer)
  templates/    # HTML templates (for visualizer)
  static/       # Static assets (JS, CSS)
tests/
  unit/         # Unit tests
  end_to_end/   # End-to-end tests
  frontend/     # Frontend tests
  nki/          # NKI-related tests
```

## Submitting Changes

1. Create a new branch from `main`.
2. Make your changes and ensure all tests pass.
3. Run `pre-commit run -a` to verify code style.
4. Submit a pull request.
