# NKI Analysis and Cost-Model Tools

This package turns Triton-Viz NKI traces and AWS Neuron Explorer artifacts into
auditable per-engine performance predictions. It is intentionally separate from
the tracing core: core modules record semantic operations and pointer ranges;
these tools perform offline artifact export, source-to-ISA attribution,
calibration, scheduling, and validation.

## Architecture

```text
NKI kernel
├── Triton-Viz trace JSONL
│   ├── memory traffic and address ranges
│   ├── source compute regions
│   └── structured region IR
└── NEFF + NTFF
    └── Neuron Explorer parquet
        ├── instructions/opcodes/engines
        ├── active-time intervals
        ├── flow and semaphore evidence
        └── DMA packets and byte counters

trace + parquet
  -> source-region-to-ISA mapping
  -> Level-B instruction cost + Level-A lowering expansion
  -> control-routed whole-program engine occupancy (interpolated)
  -> dependency/resource scheduler
  -> DMA descriptor-issue floor on DMA resource occupancy
  -> per-engine busy time
  -> one global completion term
  -> predicted NC latency
```

The completion term is deliberately singular. `simulate` ends in

```python
final_ns = max(makespan_only_ns, global_completion_ns)
```

where `global_completion_ns` is
`max(engine_busy) + beta * (sum(engine_busy) - max(engine_busy)) + offset` with
one global `(beta, offset)` pair. No per-structure completion table and no
second overhead floor exists; both classes of mechanism were removed because a
lookup of a same-class program's end-to-end time is not a model, and two
estimators of one physical quantity combined with `max()` bias the result
upward.

## Modules

- `nki_trace_dump.py`: serializes trace records, address/storage ranges, stable
  source region identities, fusion signatures, and structured region IR.
- `nki_region_ir.py`: canonical compiler-relevant grammar features including
  reductions, elementwise arity, DAG edges, dtype/shape, mask/tail,
  free-block count, and partition-broadcast inputs. Its declarative rule
  registry gives every structural family a stable rule ID, rationale, control
  evidence, ambiguity check, and explicit out-of-distribution diagnostics.
- `nki_instruction_source_mapping.py`: maps Explorer instructions to source
  regions through Penguin IDs/source metadata, transfer boundaries, opcode and
  ownership evidence; writes mapping CSV and an audit JSON.
- `nki_cost_model.py`: DMA calibration surfaces, Level-B compute calibration,
  structured Level-A expansion, dependency-aware scheduling, and simulation.
- `nki_fit_structured_controls.py`: exports the operator-independent grammar,
  dtype, shape, engine, instruction-count, effective-count, and fixed-work table.
- `nki_explorer.py`: shared, completeness-checked parquet export that handles
  Explorer's known post-flush process hang without waiting out every timeout.
- `nki_provenance.py`: builds canonical compiler-stack fingerprints and
  classifies environment differences without claiming that a version match or
  mismatch proves lowering stability.
- `nki_fit_structural_static_dma.py`: exports compiler-generated Static DMA
  busy time from controls using structural rule sequence, element width, and
  free dimension; it never keys on operator names.
- `nki_replay_operator_predictions.py`: replays calibrated predictions on saved
  traces and hardware counters without recompilation or holdout fitting.
- `nki_fit_strided_dma.py`: fits compiler-generated strided/Static-DMA packet
  train busy time from independent access-geometry controls.
- `nki_fit_global_completion.py`: fits the single global NC completion model
  (one overlap fraction and one launch/drain constant, no structural key) from
  independent whole-program controls.
- `nki_fit_dma_elapsed.py`: fits the global DMA descriptor-issue interval that
  bounds queue-elapsed time for fragmented transfers.
- `nki_fit_onchip_copy.py`: fits the on-chip PSUM/SBUF copy latency surface
  (`startup_ns + ns_per_free_elem * free`) from the repeat-differenced
  `onchip_copy_disjoint_v2` controls; gate is leave-one-width-out.
- `nki_fit_attention_pipeline.py`: freezes the TensorE busy surface for
  QK-normalize-PV Dot pipelines from independent control grids.
- `nki_fit_tensor_source_geometry.py`: freezes the TensorE busy surface keyed by
  source-visible tiled-Dot geometry.
- `nki_aggregate_unified_replay.py`: aggregates every replayed split into the
  authoritative unique-case NC-p50 headline and per-engine WAPE.
- `nki_cost_model_pipeline.py`: three-stage `collect`, `fit`, and `evaluate`
  entry point for calibration, Tilebench holdouts, MAPE and ablation reports.
- `nki_region_control_experiments.py`: compiles and profiles minimal lowering
  controls, exports parquet, and invokes source mapping.
- `nki_operator_experiments.py`: traces and profiles the same external
  Tilebench kernel source for softmax/rmsnorm/layernorm validation.
- `nki_workload_cases.py`: shared workload-case helpers (`load_cases`,
  `write_csv`, profiling helpers) used by the operator driver and tests.
- `nki_program_context.py`: whole-program source-visible context features
  (region count, DAG join count, masked-event count, total transfer bytes) that
  key the whole-program engine-occupancy surface.
- `nki_evaluate_whole_program_regime.py`: leave-one-entire-free-dimension-out CV
  for that surface. It mirrors production exactly, including the linear
  interpolation in transfer size, so the gate measures what the model does.
- `nki_features.py`: `AccessPattern` and `ComputeRegion`, the shared geometry
  views the DMA and compute paths are keyed on (partition count, free stride,
  active access count, item bytes, layout family).

## Environment

```bash
cd /home/ubuntu/triton-viz
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PYTHONPATH="$PWD"
```

Hardware workflows require an Inf2 instance, `neuronxcc.nki`, and
`neuron-explorer`. Operator validation additionally requires Tilebench. It is an
external dependency rather than copied code:

```bash
export TILEBENCH_OPS_DIR=/path/to/Tilebench/benchmarks/operators
```

The operator driver also accepts `--tilebench-ops-dir`; the environment variable
is only a convenient default.

## Trace and source mapping

Given a case directory containing `trace.jsonl`, `hardware/file.neff`,
`hardware/profile.ntff`, and `hardware/explorer_parquet/`:

```bash
neuron-explorer view \
  -n <case>/hardware/file.neff \
  -s <case>/hardware/profile.ntff \
  --output-format parquet \
  --output-file <case>/hardware/explorer_parquet \
  --disable-ui --ignore-event-trace

python -m triton_viz.tools.nki_instruction_source_mapping <case>
```

The output includes `instruction_mapping.csv` and `audit.json`. Before accepting
a calibration point, verify:

- attributed plus unattributed instruction count equals Explorer count;
- VectorE/ScalarE payload coverage is reported explicitly;
- active time is reconstructed as interval union rather than a sum that double
  counts overlap;
- ambiguous instructions remain `unattributed` instead of being evenly spread.

Runtime semaphore, drain, and notify instructions are preserved in the audit but
are not misrepresented as source arithmetic.

## Two-level compute calibration

Level-B describes one physical instruction. Produce it from independent engine
sweeps, not operator latency:

```bash
python -m microbench.inf2_nki.harness.run_microbench \
  --config microbench/inf2_nki/configs/engine_lowering_sweep.json \
  --profile-export parquet
python -m microbench.inf2_nki.profile_parser.fit_compute_calibration \
  <engine-run>/all_results.csv \
  --output /tmp/compute_calibration_v2.csv
```

Level-A predicts how a structured source region expands onto target engines.
Collect controls and fit the table:

```bash
python -m triton_viz.tools.nki_region_control_experiments \
  --output-dir /tmp/controls_fp32 \
  --dtypes float32 --free-dims 128 512 1024 2048 4096

python -m triton_viz.tools.nki_region_control_experiments \
  --output-dir /tmp/controls_bf16 \
  --kinds elementwise_one elementwise_two \
          two_pass_reduce_multiply two_pass_reduce_affine \
  --dtypes bfloat16 --free-dims 128 512 1024 2048 4096

python -m triton_viz.tools.nki_fit_structured_controls \
  /tmp/controls_fp32 /tmp/controls_bf16 \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --legacy-level-a-csv /tmp/softmax_lowering_calibration_v2.csv \
  --output /tmp/structured_control_lowering.csv
```

The stable model key contains no operator name and no allocation address. Full
source signatures are retained only for debugging. Immediate region context is
included because Explorer demonstrates compiler instruction motion across a
transfer boundary; multi-block fingerprints are separate from single-block
interpolation.

`match_structural_family(region_ir)` returns the family together with its rule
ID, evidence controls, and OOD reasons. Calibration and holdout CSVs preserve
these fields for audit. `structural_family(region_ir)` remains the compatibility
wrapper for consumers that only need the established family string. Unknown
operations are reported as OOD and can be rejected with `strict=True`; they are
never silently treated as validated grammar coverage.

Every new microbenchmark run manifest also contains a structured
`compiler_fingerprint` with package/tool versions, hardware/platform identity,
repository revision, and Region IR schema. A changed fingerprint requires a
canary run; it is provenance evidence, not a substitute for the artifact-level
lowering comparison.

## External operator validation

The Tilebench driver dynamically imports kernels from the supplied directory and
uses identical NumPy inputs for the Triton-Viz trace and NKI hardware compile:

```bash
python -m triton_viz.tools.nki_operator_experiments \
  --tilebench-ops-dir "$TILEBENCH_OPS_DIR" \
  --output-dir /tmp/norm_fp32 \
  --ops rmsnorm layernorm --rows 128 \
  --cols 128 512 1024 2048 4096 --dtype float32 \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --structured-control-csv /tmp/structured_control_lowering.csv
```

Add `--source-mapping` to export Explorer parquet and invoke source mapping for
every hardware case. Both the control and operator drivers support `--resume`:
only successful CSV rows are skipped, and source-mapping mode additionally
requires the mapping artifact to exist before a case counts as complete.

For a strict test, do not add rmsnorm/layernorm mapping rows to the structured
CSV. Leave-one-operator-out experiments may use
`nki_fit_structured_controls --include-case-prefix control_ rmsnorm` (or
`layernorm`) and must label the opposite operator as holdout.

Do not interpret a zero mapped instruction count as observed zero work. The
evaluator reads `mapped_payload_coverage_percent` from each mapping audit and
requires complete attribution before including a region in count or active-time
MAPE. Whole-engine predictions may still be evaluated separately when source
mapping is incomplete.

For DMA evaluation, distinguish source events from compiler-issued transfers.
`--compiler-load-cse` applies only exact same-storage/range/shape load reuse and
records eliminated count/bytes. Directional DMA timing comes exclusively from
the partition/free-byte calibration surfaces. Add
`--structural-static-dma-csv` when comparing against Explorer's total
`dma_active_time`, which includes compiler-generated Static DMA.

## Cost-model CLI

Run a trace through the model with the calibration files appropriate to the
workload:

```bash
python -m triton_viz.tools.nki_cost_model <trace.jsonl> \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --dma-calibration-csv <read-surface.csv> \
  --dma-write-calibration-csv <write-surface.csv> \
  --dma-transpose-calibration-csv <transpose-surface.csv>
```

This CLI is a debugging entry point that exercises the analytical/DMA path
only. The production prediction path is `nki_replay_operator_predictions`,
which is what `nki_cost_model_pipeline evaluate` invokes and what the reported
MAPE comes from; it additionally supplies the structured lowering, whole-program
routing, TensorE, on-chip transfer, DMA descriptor-issue and global completion
calibrations.

Use `python -m <module> --help` to confirm exact arguments for the checked-out
revision. Calibration CSVs carry compiler/profile provenance and should not be
mixed across incompatible compiler fingerprints without an explicit comparison.

## Tests

```bash
# the whole NKI surface
PYTHONPATH=$PWD pytest -q -m "" tests/nki/ tests/unit/test_nki_fit_structured_controls.py

# or the cost-model core only
PYTHONPATH=$PWD pytest -q -m "" \
  tests/nki/test_nki_trace_dump.py \
  tests/nki/test_nki_instruction_source_mapping.py \
  tests/nki/test_nki_region_ir.py \
  tests/nki/test_nki_cost_model.py \
  tests/nki/test_nki_global_completion.py \
  tests/nki/test_nki_dma_elapsed.py \
  tests/nki/test_nki_onchip_transfer.py
```

## Known limitations

- Factory kernels and nested `nl.*` expressions are not always fully visible to
  the simulator AST. Region-control source declarations describe the intended
  grammar, while real Penguin/Explorer output remains the ISA authority.
- Hardware result directories can be large and are intentionally not source
  files. Keep NEFF/NTFF/parquet outside commits unless a review explicitly asks
  for a compact fixture.
- Correct per-engine busy time does not close end-to-end latency by itself.
  Missing load-to-SBUF identities, semaphore dependencies, shared DMA/HBM
  resources, and sequencer overhead affect the critical path.
