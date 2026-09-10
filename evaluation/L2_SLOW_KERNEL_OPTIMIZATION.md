# Selected L2 slow-kernel optimization

The 2026-09-07 UTC diagnostic compares nine original L2 configurations
against `ff160f1`, plus two fresh paired full-grid enumeration measurements.
Combined timings use `6f3e9db`; final validated implementation `490250d`
also guards replay snapshot fidelity. The original measured worktree commits
are retained in the integrated history. This is not a formal corpus rerun.

## Implementation

- Ordinary non-atomic accesses use equivalent smaller conflict formulas.
  Literal happens-before results are folded without changing atomic cases.
- The existing 60 s dynamic watchdog interrupts native Z3 checks, retries
  swallowed alarms, and rejects over-budget results. Cleanup also restores
  timers when thread startup fails.
- Structural integer domains remove redundant signed division branches.
  Negative operands and zero divisors preserve the old unbounded-Int
  encoding. Domain caches invalidate when parameters or symbolic mode change.
- Global-memory bounds use captured allocation intervals independently of
  view layout. Full access widths and storage aliases remain visible.
  Incomplete value snapshots are refused. C2/C3 additionally require cloning
  to preserve full allocation, layout, values and alias relationships;
  unavailable replay cannot confirm, refute or produce a content hazard.
- Enumeration shares stable address sorting, caches only interpreter class
  identities between launch-boundary resets, and omits unused taint unions.
  Every original instance, operation, dependency and fence is retained.

## Selected measurements

| Kernel | Earlier result | Combined result |
| --- | --- | --- |
| TileBench BMM | 200 s main timeout | 62.16 s, enum proof |
| TileBench bitonic step | 200 s main timeout | 65.65 s, enum proof |
| TileBench mixed-precision matmul | 191.42 s, abstain | 9.68 s, any-grid static proof |
| Meta tutorial matmul | 195.12 s, abstain | 41.75 s, any-grid static proof |
| Meta persistent matmul | 188.33 s, enum proof | 70.59 s, launch static proof |
| FlagGems classic MM retry | 320 s timeout | 237.35 s, launch static proof |

Classic MM still exceeds the 200 s main budget. General FlagGems MM and
Meta partition-K retain 120 s static solver timeouts. Split-K's exact
any-grid WAW uses an instance outside the captured grid; the launch requery
remains undecided. Its existing broader-grid report is not a new captured
launch bug. Plain FLA loop pairs establish no speedup.

Fresh enum pairs take 62.36 to 53.27 s for `tb_destindex_copy` and 131.39
to 112.85 s for `tb_quantize_kv_transform`. Both preserve all 32768 instances
and all 17 normalized recorder-field hashes. Budgets, inputs and grid sizes
are unchanged. Whole-row comparisons can change proof scope and analysis
path; censored timeouts are not component-speedup denominators.

## Validation and provenance

Final source `490250d` passes **1550 tests with five skipped**, covering all
unit tests and selected detector end-to-end suites. Nineteen new replay
tests include actual false-confirmation and false-refutation counterexamples.
The final guard preserves snapshot eligibility, layout and tensor names for
all nine measured captures, with original input hashes intact. Timings are
still attributed to `6f3e9db`, rather than relabeled as final-source timings.

The complete 144-file archive is
`evaluation/results/l2-slow-kernel-opt-20260907/`, with `SHA256SUMS.json`.
It retains stage probes, main/retry records, input and available TTIR hashes,
failed instrumented FLA samples, enum field comparisons, regression outputs,
and the replay eligibility audit. The paper repository mirrors the compact
record at `baselines/results/l2-slow-kernel-opt-20260907/`. Its results index
and evaluation protocol identify these as diagnostics. Formal paper numbers
and bug counts remain at their original pins pending a new reviewed run.
