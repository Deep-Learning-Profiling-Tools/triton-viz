# Inf2 NKI Microbenchmark Suite

This directory contains a reproducible NeuronCore-v2 characterization suite for
AWS Inferentia2. It measures NKI-visible latency, directional HBM DMA bandwidth,
DMA transpose behavior, ScalarE/VectorE/TensorE throughput, engine overlap,
program placement, compiler lowering, and the controls used by the Triton-Viz
NKI cost model.

The suite is designed for modeling rather than application-level benchmarking.
Every hardware case records declared work, Neuron Explorer counters, compiler
artifacts, and enough metadata to audit byte counts and instruction ownership.

## Requirements

- An Inf2 host with an idle NeuronCore.
- AWS Neuron SDK, `neuronxcc.nki`, `neuron-explorer`, NumPy, pandas, matplotlib,
  pyarrow, and pytest.
- This repository on `PYTHONPATH`.
- For operator holdouts only, a separate Tilebench checkout. Tilebench is not
  vendored because it is an independent project; pass its
  `benchmarks/operators` directory explicitly.

The commands below assume the AWS Neuron virtual environment used during
development:

```bash
cd /home/ubuntu/triton-viz
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PYTHONPATH="$PWD"
export TILEBENCH_OPS_DIR=/path/to/Tilebench/benchmarks/operators
```

## Directory layout

```text
microbench/inf2_nki/
├── common/           input generation and NKI profiling helpers
├── configs/          declarative JSON sweeps
├── harness/          single-config and run-all orchestration
├── profile_parser/   CSV export, calibration fitting, and plots
└── tests/            top-level NKI kernels grouped by measured mechanism
    ├── latency_pointer_chase/
    ├── bandwidth_dma/
    ├── engine_ops/
    ├── overlap/
    ├── program_mapping/
    ├── static_dma/
    └── region_controls/
```

Adding a benchmark normally requires one top-level kernel factory, one config
entry, and work metadata in `harness/run_microbench.py`. CSV metadata is flattened
automatically, so new parameters do not require a fixed output schema change.

## One-command hardware suite

Run the canonical suite and produce one combined CSV plus DMA plots:

```bash
python -m microbench.inf2_nki.harness.run_all
```

Use `--continue-on-error` when sharing the device with other processes. Run the
small smoke suite with:

```bash
python -m microbench.inf2_nki.harness.run_microbench \
  --config microbench/inf2_nki/configs/quick.json \
  --profile-export summary-json
```

Validate kernel compilation/simulation without running the full sweep:

```bash
python -m microbench.inf2_nki.harness.validate_kernels
PYTHONPATH=$PWD pytest -q -m "" tests/nki/test_inf2_microbench.py
```

Export an existing result tree again without rerunning hardware:

```bash
python -m microbench.inf2_nki.profile_parser.export_csv \
  microbench/inf2_nki/results/<run-directory> \
  --output /tmp/inf2_nki_results.csv
```

## Reproduce DMA characterization

The main DMA profiles separate read/write direction, partition count, free bytes
per partition, transpose layout, and multi-block behavior:

```bash
for config in \
  dma_free_dimension.json \
  dma_partition_surface.json \
  dma_partition_large_free.json \
  dma_write_partition_surface.json \
  dma_transpose_surface.json \
  dma_transpose_pipeline.json; do
  python -m microbench.inf2_nki.harness.run_microbench \
    --config "microbench/inf2_nki/configs/$config" \
    --profile-export parquet
done
```

Generate the plots from the resulting exported CSV files:

```bash
python -m microbench.inf2_nki.profile_parser.plot_dma_free_dimension \
  <dma-free-dimension.csv> --output /tmp/dma_free_dimension.png
python -m microbench.inf2_nki.profile_parser.plot_dma_partition_surface \
  <partition-surface.csv> --output /tmp/dma_partition_surface.png
python -m microbench.inf2_nki.profile_parser.plot_dma_transpose_surface \
  <transpose-surface.csv> --output /tmp/dma_transpose_surface.png
python -m microbench.inf2_nki.profile_parser.plot_dma_copy_vs_transpose \
  <copy.csv> <transpose.csv> --output /tmp/dma_copy_vs_transpose.png
```

Bandwidth is computed from Explorer-observed bytes divided by aggregate DMA
active time. Always check the exported byte-count match columns before using a
point for calibration. Device-level headline bandwidth is not directly
comparable to a single NKI kernel: on Inf2 one kernel program uses one
NeuronCore; concurrent device measurements require separate core-pinned
processes.

## Reproduce Level-B compute calibration

Run instruction-audited one/two-input FP32/BF16 sweeps:

```bash
python -m microbench.inf2_nki.harness.run_microbench \
  --config microbench/inf2_nki/configs/engine_lowering_sweep.json \
  --profile-export parquet

python -m microbench.inf2_nki.profile_parser.fit_compute_calibration \
  <engine-lowering-run>/all_results.csv \
  --output /tmp/compute_calibration_v2.csv
```

When the input is a combined canonical `all_results.csv`, the fitter selects
`run_id=engine_lowering_sweep` by default so same-named diagnostic kernels from
other suites cannot contaminate Level-B. Use repeated `--run-id` only when
combining calibration runs intentionally; selected run IDs are written into the
output CSV provenance.

The fitted Level-B table models the cost of one target-engine instruction as a
startup term plus a free-dimension slope, keyed by engine, dtype, and input
stream count. Do not refit this table on application operators.

## Reproduce structured lowering controls

The region controls isolate elementwise chains, reductions, rsqrt/Newton,
partition broadcast, mask/tail behavior, and single/multi-block two-pass
pipelines. Hardware runs automatically export Explorer parquet and create the
source-region-to-ISA audit.

```bash
python -m triton_viz.tools.nki_region_control_experiments \
  --output-dir /tmp/region_controls_fp32 \
  --dtypes float32 --free-dims 128 512 1024 2048 4096

python -m triton_viz.tools.nki_region_control_experiments \
  --output-dir /tmp/region_controls_bf16 \
  --kinds elementwise_one elementwise_two \
          two_pass_reduce_multiply two_pass_reduce_affine \
  --dtypes bfloat16 --free-dims 512 2048
```

Fit the reusable Level-A grammar table. The softmax CSV must be produced by the
mapping-aware workflow described in `triton_viz/tools/README.md`.

```bash
python -m triton_viz.tools.nki_fit_structured_controls \
  /tmp/region_controls_fp32 /tmp/region_controls_bf16 \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --legacy-level-a-csv /tmp/softmax_lowering_calibration_v2.csv \
  --output /tmp/structured_control_lowering.csv
```

## Tilebench norm holdouts

Tilebench supplies real rmsnorm/layernorm kernels and is deliberately external
to keep calibration controls independent from validation operators. Point the
driver at any compatible checkout:

```bash
python -m triton_viz.tools.nki_operator_experiments \
  --tilebench-ops-dir "$TILEBENCH_OPS_DIR" \
  --output-dir /tmp/norm_holdout_fp32 \
  --ops rmsnorm layernorm --rows 128 \
  --cols 128 512 1024 2048 4096 --dtype float32 \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --structured-control-csv /tmp/structured_control_lowering.csv

python -m triton_viz.tools.nki_operator_experiments \
  --tilebench-ops-dir "$TILEBENCH_OPS_DIR" \
  --output-dir /tmp/norm_holdout_bf16 \
  --ops rmsnorm layernorm --rows 128 --cols 512 2048 --dtype bfloat16 \
  --compute-calibration-csv /tmp/compute_calibration_v2.csv \
  --structured-control-csv /tmp/structured_control_lowering.csv
```

Calibration must use softmax plus independent controls only. Norm active times
are validation data unless a run is explicitly labeled leave-one-operator-out.

## Measurement limitations

- `nki.benchmark` profiles compiled execution; it is not a numerical correctness
  oracle. Use simulator/CPU-reference tests for data-dependent kernels.
- Latency pointer chase reports NKI-visible dependent access including compiler
  and DMA machinery, not bare DRAM cell latency.
- Summary JSON is sufficient for aggregate sweeps; source mapping and packet
  continuity require parquet.
- Explorer may finish writing parquet before its UI process exits. The focused
  drivers accept a timeout only when required tables are complete.
- Per-engine active-time accuracy does not imply accurate NC wall latency.
  Cross-engine dependencies, shared resources, and sequencer overhead are
  separate scheduler-model concerns.
