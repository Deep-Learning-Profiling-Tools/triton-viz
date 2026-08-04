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
  -> dependency/resource scheduler
  -> per-engine busy time and predicted latency
```

## Modules

- `nki_trace_dump.py`: serializes trace records, address/storage ranges, stable
  source region identities, fusion signatures, and structured region IR.
- `nki_region_ir.py`: canonical compiler-relevant grammar features including
  reductions, elementwise arity, DAG edges, dtype/shape, mask/tail,
  free-block count, and partition-broadcast inputs.
- `nki_instruction_source_mapping.py`: maps Explorer instructions to source
  regions through Penguin IDs/source metadata, transfer boundaries, opcode and
  ownership evidence; writes mapping CSV and an audit JSON.
- `nki_cost_model.py`: DMA calibration surfaces, Level-B compute calibration,
  structured Level-A expansion, dependency-aware scheduling, and simulation.
- `nki_fit_lowering_calibration.py`: mapping-aware signature/fingerprint fitter,
  useful for focused source-mapping studies and softmax seed calibration.
- `nki_fit_structured_controls.py`: exports the operator-independent grammar,
  dtype, shape, engine, instruction-count, effective-count, and fixed-work table.
- `nki_fit_compositional_lowering.py`: experimental additive feature fitter.
- `nki_evaluate_structured_holdout.py`: compares predicted versus mapped ISA
  count and active time without silently fitting holdout operators.
- `nki_region_control_experiments.py`: compiles and profiles minimal lowering
  controls, exports parquet, and invokes source mapping.
- `nki_operator_experiments.py`: traces and profiles the same external
  Tilebench kernel source for softmax/rmsnorm/layernorm validation.
- `nki_model_experiments.py`: validates DMA and scheduler behavior on declarative
  modeled workloads.

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
  --dtypes bfloat16 --free-dims 512 2048

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

For a strict test, do not add rmsnorm/layernorm mapping rows to the structured
CSV. Leave-one-operator-out experiments may use
`nki_fit_structured_controls --include-case-prefix control_ rmsnorm` (or
`layernorm`) and must label the opposite operator as holdout.

Audit mapped holdouts independently:

```bash
python -m triton_viz.tools.nki_evaluate_structured_holdout \
  <mapped-holdout-root> \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --structured-control-csv /tmp/structured_control_lowering.csv \
  --output /tmp/region_count_audit.csv
```

## Cost-model CLI

Run a trace through the model with the calibration files appropriate to the
workload:

```bash
python -m triton_viz.tools.nki_cost_model <trace.jsonl> \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --structured-control-csv /tmp/structured_control_lowering.csv \
  --dma-calibration-csv <read-surface.csv> \
  --dma-write-calibration-csv <write-surface.csv> \
  --dma-transpose-calibration-csv <transpose-surface.csv>
```

Use `python -m <module> --help` to confirm exact arguments for the checked-out
revision. Calibration CSVs carry compiler/profile provenance and should not be
mixed across incompatible compiler fingerprints without an explicit comparison.

## Tests

```bash
PYTHONPATH=$PWD pytest -q -m "" \
  tests/nki/test_nki_trace_dump.py \
  tests/nki/test_nki_instruction_source_mapping.py \
  tests/nki/test_nki_region_ir.py \
  tests/nki/test_nki_cost_model.py \
  tests/nki/test_nki_model_experiments.py
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
