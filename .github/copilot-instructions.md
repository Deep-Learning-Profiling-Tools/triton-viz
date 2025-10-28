# GitHub Copilot Instructions for Triton-Viz

## Project Overview

Triton-Viz is a visualization and profiling toolkit designed for deep learning applications. It helps developers visualize and analyze GPU programming with OpenAI's Triton, making GPU programming more intuitive through real-time visualization of tensor operations and memory usage.

## Key Components

- **Core Module** (`triton_viz/core/`): Core tracing and instrumentation functionality
- **Visualizer** (`triton_viz/visualizer/`): Visualization and analysis tools
- **Clients** (`triton_viz/clients/`): Sanitizer and Profiler clients for kernel analysis
- **Wrapper** (`triton_viz/wrapper.py`): Command-line wrappers for `triton-sanitizer` and `triton-profiler`

## Development Setup

### Prerequisites
- Python >= 3.7 (tests run on Python 3.10)
- Triton (version 3.4.0 or later)
- PyTorch (pre-release with CUDA support)

### Installation
```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
pip install -e .
```

### Installing Dependencies
```bash
# Install PyTorch pre-release
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Uninstall pytorch-triton to avoid conflicts
pip uninstall pytorch-triton -y

# Install specific Triton version
pip install triton==3.4.0
```

## Code Style and Standards

### Linting
This project uses **pre-commit** with multiple linters:
- **Ruff**: Python linting and formatting (line length: 120)
- **mypy**: Static type checking (excludes tests/ and docs/)
- **flake8**: Additional Python linting
- **Standard hooks**: Check for large files, merge conflicts, YAML/TOML validity, etc.

### Running Linters
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks (runs on every git commit)
pre-commit install
```

### Code Formatting
- Line length: 120 characters (Ruff configuration)
- Use Ruff for automatic formatting
- Follow PEP 8 conventions

### Type Annotations
- Use type hints where appropriate
- mypy is configured but excludes tests and docs directories

## Testing

### Test Structure
- Tests are located in `tests/` directory
- Test files: `test_core.py`, `test_profiler.py`, `test_sanitizer.py`, `test_wrapper.py`
- Configuration: `pytest.ini`

### Running Tests
```bash
# Run all tests
python -m pytest tests

# Run with verbose output
python -m pytest tests -v

# Run specific test file
python -m pytest tests/test_core.py
```

### Testing Environment Variables
- `TRITON_INTERPRET=1`: Use Triton interpreter mode (required for CI)
- `TRITON_VERSION=3.4.0`: Specify Triton version for testing

## Build and Verification

### Build Process
The project uses setuptools with pyproject.toml configuration:
```bash
pip install -e .
```

### Package Scripts
Two command-line tools are provided:
- `triton-sanitizer`: Apply sanitizer to Triton kernels
- `triton-profiler`: Apply profiler to Triton kernels

### Full CI Workflow
```bash
# 1. Lint
pre-commit run --all-files

# 2. Install dependencies
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip uninstall pytorch-triton -y
pip install triton==3.4.0

# 3. Install package
pip install -e .

# 4. Run tests
python -m pytest tests
```

## Project-Specific Conventions

### Triton Integration
- The project patches `triton.jit` and `triton.autotune` decorators
- Store original functions before patching: `_original_jit`, `_original_autotune`
- Wrappers apply sanitizer or profiler clients to Triton kernels

### Client Pattern
- Sanitizer client: Validates kernel correctness, aborts on errors
- Profiler client: Collects performance metrics
- Both use the `triton_viz.trace()` function with appropriate client instances

### Visualization
- HTML templates in `triton_viz/templates/`
- Static assets in `triton_viz/static/`
- Uses Flask for web-based visualization interface

### File Organization
- Keep templates and static files in their respective directories
- Package data is configured in pyproject.toml to include HTML and JS files

## Common Pitfalls

1. **Triton Version Conflicts**: Always uninstall `pytorch-triton` before installing standalone Triton
2. **Import Errors**: Ensure the package is installed in editable mode (`pip install -e .`)
3. **Pre-commit Issues**: Run `pre-commit install` after cloning to set up hooks
4. **Test Environment**: Some tests may require `TRITON_INTERPRET=1` for CPU-only environments

## Examples

Example files in `examples/` directory demonstrate usage:
- `3dims.py`: Three-dimensional tensor operations
- `ima_example.py`: IMA (Immediate Access) example
- `load_store.py`: Load and store operations

Run examples with:
```bash
cd examples
python <example_file>.py
```

## Dependencies

Core dependencies (from pyproject.toml):
- `triton`: OpenAI's Triton compiler
- `flask`, `flask_cloudflared`: Web interface
- `pyarrow`: Data serialization
- `z3-solver`: Constraint solving
- `anytree`: Tree data structures
- `cairocffi`: Graphics rendering
- `pytest`: Testing framework
- `pre-commit`: Git hooks and linting

## License

MIT License - See LICENSE file for details

## Documentation

- README.md: User-facing documentation
- Publication: SIGCSE TS '25 paper on Triton-Viz
- Related project: [Triton Puzzles](https://github.com/srush/Triton-Puzzles)
