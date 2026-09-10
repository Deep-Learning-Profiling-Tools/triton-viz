# Shared performance modeling: NKI and Triton/GPU

The prediction contract is **Observe → Expand → Price → Combine**. The experiment
contract is separately **collect → fit → evaluate**. Neither target compilation
nor target machine-code inspection is required for prediction.

## What is shared

`triton_viz.performance` supplies a common `Prediction` result in nanoseconds and
the `predict_latency(source, backend)` entry point. `NkiBackend` delegates to the
existing NKI model, preserving its calibration classes, timeline, and numerical
results. `GpuBackend` consumes interpreted Triton source facts and an explicitly
frozen GPU calibration. Imports of the public prediction API initialize neither
CUDA nor the Triton compiler.

Both backends use the same auditable `GrammarRule`, `GrammarMatch`, and
highest-priority rule selection. The old NKI imports remain compatible. Canonical
provenance digests are shared as well. The new generic calibration module fits
nonnegative resource service costs, validates geometry groups, checks artifact
roles/fingerprints, and reports coverage; NKI's specialized fitters remain intact.

| Stage | NKI implementation | GPU pilot implementation |
| --- | --- | --- |
| Observe | Region IR, partition/free dimensions, storage ranges | CPU interpreter callbacks, program IDs, logical tile shapes, masked sectors, semantic precision |
| Expand | Calibrated multi-engine lowering grammar | Declarative sector, arithmetic-warp, reduction-tree, and dot-work rules |
| Price | Existing Level A/B and transfer calibration surfaces | Nonnegative control-only fit of resource service terms |
| Combine | Existing dependency and engine-queue scheduling | Experimental additive service-time model with a program-wave term |

Sharing the interface does **not** make NKI's partition semantics, DMA queues,
or completion constants GPU models. The GPU wave term is `ceil(programs / SMs)`;
it is a launch-distribution feature, not a claim about actual resident CTAs or
register-limited occupancy. GPU costs are effective service costs, not measured
per-instruction engine latencies.

## GPU pilot scope

The initial suite has 36 primitive controls and 15 composed holdouts: SiLU,
affine/ReLU, softmax, RMSNorm, and batched dot with bias/ReLU. Control input sizes
are 4,096 / 16,384 / 65,536 / 262,144 elements; holdout sizes are interleaved at
8,192 / 32,768 / 131,072. Tile sizes and launch settings are explicit. Elementwise
and reduction inputs are FP32; dots use FP16 inputs and FP32 accumulation.
The cases are a pilot, not a broad application benchmark or a reproduction of
the paper's 254-point NKI result.

The rules do not use operator names. They currently approximate source work;
they have **not** been validated against GPU machine instruction attribution.
In particular, FMA contraction, register spilling, physical layout conversion,
cache misses, asynchronous pipelines, and realistic multi-CTA occupancy are
not modeled. Transpose/join/split conversions and unknown operations are OOD.
New tile shapes, precision sets, launch configurations, or feature values outside
the calibration domain are OOD. `GpuBackend` rejects OOD by default; evaluation
retains annotated OOD cases in the all-case metric instead of dropping them.
Independent ranges are a necessary coverage check, not proof of joint coverage.

This supports one cross-platform architecture and experimental protocol today.
To claim the paper's full lowering contribution on GPU, the next evidence must
come from **control-only** source-to-instruction attribution, lowering-rule
validation, resource/occupancy controls, and broader predeclared application
holdouts. Do not present the pilot as already providing that evidence.

## Run the GPU experiment

Install a CUDA-capable PyTorch build compatible with the GPU and Triton, plus
the optional `gpu` dependencies (`torch`, `nvidia-ml-py`). For example, in an
appropriate environment, `uv sync --extra gpu --extra test` installs project
dependencies; verify that the selected PyTorch wheel actually supports CUDA.
Keep calibration environments isolated on shared machines.

```bash
python -m triton_viz.tools.gpu_cost_model_pipeline collect --root /path/to/new-run --role control
python -m triton_viz.tools.gpu_cost_model_pipeline fit --root /path/to/new-run
python -m triton_viz.tools.gpu_cost_model_pipeline collect --root /path/to/new-run --role holdout
python -m triton_viz.tools.gpu_cost_model_pipeline evaluate --root /path/to/new-run
```

`collect --role all` can collect both splits together, but fitting only opens
the declared `controls/` files. It never opens `holdouts/`. `--resume` reuses
accepted measurements only when the run identity and split definitions match.
`--dry-run` prints the chosen stage without importing a GPU runtime. Select a
physical GPU with `--device`; unset `CUDA_VISIBLE_DEVICES` to avoid ambiguous
mapping between CUDA and NVML device indices.

Each control-size group is held out as a unit during cross-validation. The
predeclared acceptance gate is 20% pooled leave-group-out MAPE, and every fold's
error is retained. Failed candidates are saved but not promoted. A latest-fit
status and content digest prevent evaluation from using a stale or altered
frozen calibration. Hardware identity, driver, package versions, and source
hashes are included in the fingerprint; a changed environment needs a new root.

Artifacts include `manifest.json`, `controls/`, `holdouts/`, all
`measurement_attempts/`, `calibration/candidate.json`, `calibration/frozen.json`,
and `evaluation/report.json`. Reports retain every holdout and OOD reason.

## Timing and shared-machine behavior

The metric is **CUDA-graph steady-cache per-kernel microseconds**, averaged over
1,024 kernel nodes per replay and summarized by the median of 11 replays.
Compilation, capture, and warmup are excluded. This is neither cold-memory
latency nor host launch latency, and must not be labeled as either. Small
working sets can remain in cache. CPU and GPU outputs are checked against a
reference before measurements are accepted.

Direct NVML queries record processes, utilization, SM clocks, temperature, power,
and host load before, during, and after timing. Foreign compute processes,
unapproved graphics processes, monitoring failures, and a sample max/min span
above 15% of the median reject the batch. Up to three complete batches can be
attempted; all samples and rejected attempts remain on disk. Individual slow
samples are never removed. No clocks, power limits, device modes, other users'
processes, or shared software installations are changed.

By default, even existing graphics processes block measurement. On an otherwise
idle desktop, `--allow-idle-graphics` allows only the graphics PIDs seen at the
start, after three low-utilization checks. Such measurements are explicitly
labeled `monitored_shared_desktop`. This does not guarantee exclusive access or
detect all brief activity by an existing desktop process. Publication-quality
results should be repeated on reserved hardware under a declared clock/cache
protocol. Raw clock and temperature telemetry is retained; it is not a model
of DVFS or thermal throttling.

## Predict a fixed kernel on a CPU-only host

```python
from triton_viz.performance import GpuBackend, predict_latency
from triton_viz.performance.triton_observe import observe

# kernel is a fixed @triton.jit function; inputs/output are CPU tensors.
source = observe(kernel, grid, *cpu_args, num_warps=4, num_stages=2)
backend = GpuBackend(calibration, fingerprint=manifest["fingerprint"],
                     sm_count=manifest["identity"]["sm_count"])
result = predict_latency(source, backend)
print(result.latency_ns, result.metric, result.diagnostics)
```

Load `calibration` and `manifest` from the frozen experiment. Source observation
executes stores into supplied CPU tensors and traces all programs; there is no
silent first-block extrapolation. Autotuners must be resolved beforehand. Traces
are concrete for supplied inputs, so data-dependent control flow represents that
execution path, not every possible input. Pointer arithmetic and exposed value
dependencies are retained; the current GPU combiner does not perform memory
hazard scheduling, and the masked-store frontend callback does not expose its
stored value. Do not interpret these traces as a complete memory-dependency DAG.

Existing NKI users can keep their current commands unchanged, or use:

```python
from triton_viz.performance import NkiBackend, predict_latency
result = predict_latency(nki_events, NkiBackend(existing_cost_model))
```
