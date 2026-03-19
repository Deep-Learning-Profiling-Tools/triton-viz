# Contributing to Triton-Viz

## Quick Start

```bash
# Clone repo
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz

# Setup dev environment and install dependencies
uv sync --extra test  # append "--extra nki" if developing NKI functionality
pre-commit install

# (code up the PR)

# Test
uv run pytest tests/ -m ""      # run all tests
npm run build:frontend           # build JS frontend (if working on the visualizer)
npm run test:frontend            # test JS (if working on the visualizer)
```

## Project Structure

```
frontend/             # TypeScript frontend source code
examples/             # Entry points for new users to try out triton-viz functionality
docs/                 # triton-viz website
triton_viz/
  frontends/          # Where to put DSL-specific code to attach to triton-viz
  transformers/       # AST rewriters
  visualizer/         # Python interface for the visualizer
  utils/              # Miscellaneous utility functions
tests/
  unit/               # Unit tests
  end_to_end/         # End-to-end tests
  frontend/           # Frontend tests
  nki/                # NKI-related tests
```
