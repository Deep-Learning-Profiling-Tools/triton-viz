<a name="readme-top"></a>
# Triton-Viz: A Visualization Toolkit for programming with Triton
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="docs/logo.png" alt="Logo" width="320" height="320">
</div>
<br/>

Welcome to Triton-Viz, a visualization and profiling toolkit designed for deep learning applications. Built with the intention of making kernel programming in tile-based DSLs like Triton more intuitive.

Visit our [site](https://deep-learning-profiling-tools.github.io/triton-viz/) to see our tool in action!

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-of-triton-viz">Installation of Triton-Viz</a></li>
        <li><a href="#reproducing-the-inf2-nki-cost-model">Reproducing the Inf2 NKI cost model</a></li>
      </ul>
    </li>
    <li>
      <a href="#working-with-examples">Working with examples</a>
    </li>
    <li><a href="#dsl-frontends">DSL frontends</a></li>
    <li><a href="#analysis-clients">Analysis clients</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About

Triton-Viz helps developers inspect Triton kernels with visualization, profiling, and memory-safety analysis tools. It can run many examples through Triton's interpreter, so GPU access is not required for basic debugging workflows.


## Getting Started

### Prerequisites
- Python >= 3.10


### Installation of Triton-Viz

> **Windows Note:** Triton-viz depends on Triton, which can only be installed on Windows Subsystem for Linux (WSL). Once installed, follow below instructions in WSL.

Most users can install directly from PyPI:

```sh
pip install triton-viz
```

If you want to run examples from this repo, contribute, or build the web UI, install from source instead:

```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
uv sync # or "uv sync --extra test" if you're running tests
```

### Web UI Build

The PyPI package ships with prebuilt web UI assets in `triton_viz/static`, so
you do not need npm to run the visualizer. If you want to modify the web UI,
rebuild the TS sources:

```sh
npm install
npm run build:frontend
```

### Optional: Enable NKI Support

For PyPI installs, install with the `nki` extra and AWS Neuron repository:

```sh
pip install triton-viz[nki] --extra-index-url https://pip.repos.neuron.amazonaws.com
```

For source installs:

```sh
uv sync --extra nki # or "uv sync --extra nki --extra test" if also running NKI-related tests
```

Note that you need to specify all features that you want _in one statement_ when using `uv sync`, i.e. if you want both NKI and testing support, you must run `uv sync --extra nki --extra test`. The below statements are wrong and will remove the NKI install when installing test packages:
```
uv sync --extra nki # NKI support but no testing
uv sync --extra test # tests but no NKI support
```

### Reproducing the Inf2 NKI cost model

The paper-oriented cost-model pipeline runs on an AWS Inferentia2 (Inf2) host
with an idle NeuronCore. The reported results were collected with the AWS
Neuron PyTorch 2.9 virtual environment at
`/opt/aws_neuronx_venv_pytorch_2_9`, `neuronx-cc 2.26.6360.0+6f180f47`, and a
separate Tilebench checkout. Compiler behavior is version-sensitive, so record
the output of `python -c "import neuronxcc; print(neuronxcc.__version__)"` when
reproducing or comparing results. The `nki` extra in this repository supplies
the Python-side NKI dependencies, but exact paper-number reproduction requires
the compiler version above.

From the repository root, activate the Neuron environment, put the repository
on `PYTHONPATH`, and run the three top-level stages:

```sh
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PYTHONPATH="$PWD"

python -m triton_viz.tools.nki_cost_model_pipeline collect --root /tmp/nki_cost_model_run \
  --tilebench-dir /path/to/Tilebench/benchmarks/operators
python -m triton_viz.tools.nki_cost_model_pipeline fit --root /tmp/nki_cost_model_run
python -m triton_viz.tools.nki_cost_model_pipeline evaluate --root /tmp/nki_cost_model_run
```

`collect` is the long-running hardware stage. It writes independent
microbenchmark/region controls under `microbench/` and `controls/`, and writes
Tilebench measurements under `holdouts/`. `fit` reads only the control trees;
it must not read `holdouts/`. `evaluate` loads the frozen calibration and only
then replays the holdouts. This directory separation is an intentional
train/holdout boundary, not merely an output convention. Use `--dry-run` on any
stage to inspect its commands without executing them.

The main artifacts under the selected root are:

- `calibration/runtime_overhead.csv`: orthogonally fitted launch/sequencer,
  engine-activation, partition, packet, and synchronization terms.
- `calibration/compute.csv`, `structured_compute.csv`, DMA CSVs, and
  `static_dma.csv`: the remaining control-only calibration surfaces.
- `evaluation/*.csv`: per-case predictions, hardware measurements, and the
  compute-only, compute-plus-DMA, scheduler-overlap, and final ablations.
- `evaluation/report.json`: aggregate holdout MAPE and the worst retained case.

The formal FP32 holdout contains 35 points at `rows=128` across eight operators:
`interleave`, `kl_divergence`, `layernorm`, `mul2`, `relu`, `rmsnorm`,
`sigmoid`, and `softmax`. All eight use free dimensions
`F={128,512,1024,2048}`; `interleave`, `layernorm`, and `rmsnorm` additionally
use `F=4096`, giving 35 points in total. No high-error point is removed, no
Level-B single-instruction constant is tuned on these points, and fitting does
not consume their measurements. With the compiler/environment above, the
mechanism-level runtime model reports 14.171% NC-p50 MAPE on this formal set.

See `microbench/inf2_nki/README.md` for individual microbenchmark commands,
schema details, and lower-level troubleshooting.

### Testing
* To run core Triton-viz tests, run `pytest tests/`.
* (if NKI installed) To run NKI-specific tests, run `pytest tests/ -m nki`.
* To run all tests (Triton + NKI), run `pytest tests/ -m ""`.
* To run visualizer web UI tests, run `npm run test:frontend`.

## Working with Examples

Run an example directly with Python:

```sh
python examples/visualizer/matmul.py
```

Use the decorator API when writing or modifying a Triton kernel:

```py
import triton
import triton.language as tl
import triton_viz


@triton_viz.trace("sanitizer")  # also supports "tracer" and "profiler"
@triton.jit
def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, values)
```

Use the CLI wrappers to run an existing Python script without editing it. These
wrappers patch plain `@triton.jit` kernels, so use them with scripts that do not
already apply `@triton_viz.trace(...)`.

```sh
triton-sanitizer examples/sanitizer/oob_cli.py
triton-profiler examples/profiler/load_store_cli.py
triton-visualizer trace.tvz
```

For visualizer workflows, save a trace and launch the UI from Python:

```py
import triton_viz

triton_viz.save("trace.tvz")
triton_viz.launch()
```

## DSL Frontends

Triton is the default DSL frontend. NKI support is optional and selected with
the `frontend` argument:

```py
triton_viz.trace("tracer")  # Triton
triton_viz.trace("tracer", frontend="nki")  # NKI
triton_viz.trace("tracer", frontend="nki_beta2")  # NKI Beta 2
```

The runtime integration code lives under `triton_viz/core/frontend/`. NKI
simulation runtimes live under `triton_viz/core/simulation/`.

## Analysis Clients

Analyze kernels across visualization, profiling, and sanitization with a single line of code.

- Visualizer: currently supports load, store, and matmul operations for 1/2/3D tensors (more operations and dimensions coming soon).
- Profiler: flags non-unrolled loops, inefficient mask usage, and missing buffer_load optimizations while tracking load/store byte counts with low-overhead sampling.
- Sanitizer: symbolically checks tensor memory accesses for out-of-bounds errors and emits reports with tensor metadata, call stack, and expression trees; optional fake-memory storage avoids real reads.

### Save and load traces

```py
import triton_viz

triton_viz.save("trace.tvz")
triton_viz.load(
    "trace.tvz"
)  # automatically clears out existing records, use kwarg "append=True" to prevent this
triton_viz.launch()
```

CLI: `triton-visualizer trace.tvz`. The archive is a zip file containing `manifest.json` plus `tensors.npz`, and `triton_viz.load(...)` restores the normal trace state for existing consumers.

### Environment variables

Triton-Viz uses a small set of environment variables to configure runtime behavior. Unless noted, boolean flags are enabled only when set to `1`.

- `TRITON_VIZ_VERBOSE` (default: `0`): enable verbose logging and extra debug output.
- `TRITON_VIZ_NUM_SMS` (default: `1`): number of concurrent SMs to emulate for the CPU interpreter (min 1).
- `TRITON_VIZ_PORT` (default: `8000` with `share=True`, `5001` with `share=False`): port for the Flask server.
- `ENABLE_SANITIZER` (default: `1`): enable the sanitizer pipeline that checks memory accesses.
- `ENABLE_PROFILER` (default: `1`): enable the profiler pipeline that collects performance data.
- `ENABLE_TIMING` (default: `0`): collect timing data during execution.
- `REPORT_GRID_EXECUTION_PROGRESS` (default: `0`): report per-program block execution progress in the interpreter.
- `SANITIZER_ENABLE_FAKE_TENSOR` (default: `0`): use fake tensor storage for sanitizer runs to avoid real memory reads.
- `PROFILER_ENABLE_LOAD_STORE_SKIPPING` (default: `1`): skip redundant load/store checks to reduce profiling overhead.
- `PROFILER_ENABLE_BLOCK_SAMPLING` (default: `1`): sample a subset of blocks to reduce profiling overhead.
- `PROFILER_DISABLE_BUFFER_LOAD_CHECK` (default: `0`): disable buffer load checks in the profiler.

## More Puzzles

If you're interested in fun puzzles to work with in Triton, do check out: [Triton Puzzles](https://github.com/srush/Triton-Puzzles)

## License

Triton-Viz is licensed under the MIT License. See the [LICENSE](LICENSE) for details.

## Publication
If you find this repo useful for your research, please cite our paper:

```
@inproceedings{ramesh2025tritonviz,
  author={Ramesh, Tejas and Rush, Alexander and Liu, Xu and Yin, Binqian and Zhou, Keren and Jiao, Shuyin},
  title={Triton-Viz: Visualizing GPU Programming in AI Courses},
  booktitle = {Proceedings of the 56th ACM Technical Symposium on Computer Science Education (SIGCSE TS '25)},
  numpages = {7},
  location = {Pittsburgh, Pennsylvania, United States},
  series = {SIGCSE TS '25}
}

@inproceedings{wu2026tritonsanitizer,
  author    = {Wu, Hao and Zhao, Qidong and Chen, Songqing and Chen, Yang and Hao, Yueming and Liu, Tony C. W. and Chen, Sijia and Aziz, Adnan and Zhou, Keren},
  title     = {Triton-Sanitizer: A Fast and Device-Agnostic Memory Sanitizer for Triton with Rich Diagnostic Context},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Pittsburgh, PA, USA},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
  series    = {ASPLOS '26},
  keywords  = {GPU, Debugging, Symbolic Execution, Memory Safety, Triton, Memory Access Errors}
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
