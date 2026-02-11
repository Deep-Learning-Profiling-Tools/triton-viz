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
    <li>
      <a href="#About">About</a>
    <li>
      <a href="#Getting-Started">Getting Started</a>
      <ul>
        <li><a href="#Prerequisites">Prerequisites</a></li>
        <li><a href="#Installation-of-Triton_Viz">Installation of Triton_Viz</a></li>
      </ul>
    <li>
      <a href="#Working-with-Examples">Working with examples</a>
    <ul>
        <li><a href="#More-Puzzles">More puzzles</a></li>
      </ul>
    </li>
    <li><a href="#Webpage-Notes">Webpage notes</a></li>
    <li><a href="#Analysis-Clients">Analysis clients</a></li>
    <li><a href="#Visualizer-Features">Visualizer features</a></li>
    <li><a href="#License">License</a></li>
  </ol>
</details>

## About

Triton-Viz is a visualization and analysis toolkit specifically designed to complement the development and optimization of applications written in OpenAI's Triton, an open-source programming language aimed at simplifying the task of coding for accelerators such as GPUs.
Triton-Viz offers a suite of features to enhance the debugging, performance analysis, and understanding of Triton code.

Given that Triton allows developers to program at a higher level while still targeting low-level accelerator devices, managing and optimizing resources like memory becomes a crucial aspect of development.
Triton-Viz addresses these challenges by providing real-time visualization of tensor operations and their memory usage.
The best part about this tool is that while it does focus on visualizing GPU operations, users are not required to have GPU resources to run examples on their system.


## Getting Started

### Prerequisites

- Python installed (preferably the latest available version), minimum supported version is 3.10.


### Installation of Triton-Viz

Most users can install directly from PyPI:

```sh
pip install triton-viz
```

If you want to run examples from this repo, contribute, or build the frontend, install from source instead:

```sh
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
uv sync # or "uv sync --extra test" if you're running tests
```

If you want to run tests, run `uv sync --extra test` instead of `uv sync`. Otherwise you're all set!

### Frontend Build

The PyPI package ships with prebuilt frontend assets in `triton_viz/static`, so
you do not need npm to run the visualizer. If you want to modify the frontend,
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

For source installs, if you want to exercise the Neuron Kernel Interface (NKI) interpreter or run the NKI-specific tests:

```sh
uv sync --extra nki # or "uv sync --extra nki --extra test" if also running tests
```

Note that you need to specify all features that you want _in one statement_ when using `uv sync`, i.e. if you want both NKI and testing support, you must run `uv sync --extra nki --extra test`. The below statements are wrong and will remove the NKI install when installing test packages:
```
uv sync --extra nki
uv sync --extra test
```

### Testing
* To run core Triton-viz tests, run `pytest tests/`.
* (if NKI installed) To run NKI-specific tests, run `pytest tests/ -m nki`.
* To run all tests (Triton + NKI), run `pytest tests/ -m ""`.
* To run visualizer frontend tests, run `npm run test:frontend`.

## Working with Examples

Examples live in this repo. Clone it first if you installed via pip.

```sh
cd examples
python <file_name>.py
```

## Webpage Notes

- Triton is best supported today; Amazon NKI DSL support is in active development.
- The web visualizer requires a browser with WebGL/OpenGL enabled (standard in modern browsers).

## Analysis Clients

Analyze kernels across visualization, profiling, and sanitization with a single line of code.

- Visualizer: currently supports load, store, and matmul operations for 1/2/3D tensors (more operations and dimensions coming soon).
- Profiler: flags non-unrolled loops, inefficient mask usage, and missing buffer_load optimizations while tracking load/store byte counts with low-overhead sampling.
- Sanitizer: symbolically checks tensor memory accesses for out-of-bounds errors and emits reports with tensor metadata, call stack, and expression trees; optional fake-memory backend avoids real reads.

## Visualizer Features

- 3D View: inspect tensor layouts and memory access patterns from any perspective.
- Program IDs: examine op inputs/outputs at specific PIDs and see per-program load/store footprints.
- Code Mapping: map visual ops back to source lines for debugging.
- Heatmaps: spot outliers, zeros, or saturation with value color gradients.
- Histograms: review value distributions to guide quantization decisions.

### Environment variables

Triton-Viz uses a small set of environment variables to configure runtime behavior. Unless noted, boolean flags are enabled only when set to `1`.

- `TRITON_VIZ_VERBOSE` (default: `0`): enable verbose logging and extra debug output.
- `TRITON_VIZ_NUM_SMS` (default: `1`): number of concurrent SMs to emulate for the CPU interpreter (min 1).
- `TRITON_VIZ_PORT` (default: `8000` with `share=True`, `5001` with `share=False`): port for the Flask server.
- `ENABLE_SANITIZER` (default: `1`): enable the sanitizer pipeline that checks memory accesses.
- `ENABLE_PROFILER` (default: `1`): enable the profiler pipeline that collects performance data.
- `ENABLE_TIMING` (default: `0`): collect timing data during execution.
- `REPORT_GRID_EXECUTION_PROGRESS` (default: `0`): report per-program block execution progress in the interpreter.
- `SANITIZER_ENABLE_FAKE_TENSOR` (default: `0`): use a fake tensor backend for sanitizer runs to avoid real memory reads.
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
