# Combined conformance repair validation (2026-09-07)

Hao requested completing the four residual repairs after the paper adopted
the immutable `31c48f5` sequence: dot-C provenance, frontend consistency,
dynamic timeout return, and cuTile token ordering/scoring. The combined
mechanism-validation source is `55adc887dcd5a4f6a5b2399f437fca4d0bfbcaf8`.
The subsequent final integration also contains the concurrently completed
L2 execution-policy change, recorded below.
This is a correctness and protocol validation record, not a declaration
or numerical publication of a new full pinned rerun.

## Integrated changes

- Dot-C production recognition is `e6358a2`; `79034e0` integrates its
  dependency rules with the frontend repair. Exact FLA diagnostics and
  conservative syntax controls are in `TT_DOT_C_PROVENANCE.md`.
- Frontend fixes are `234a8fe`, `52bf83d` and callee location correction
  `edb37fb` (integration `2e970f8`). Source-matched corpus controls and
  remaining full Hadamard symbolic timeouts are in
  `FRONTEND_CONFORMANCE_REPAIRS.md`.
- Final review also found an inactive select arm could create false order
  in both dynamic and enum paths. `b3af535` (integration `55adc88`) fixes
  both, including nested `create_select`/`ternary_op(np.where)` wrappers.
  Four former interpreter false-clean controls now report conflicts;
  enumeration conservatively refuses them. This implements the same
  existing select contract as the static reader.
- Deadline cancellation is `c812712`/`66d0dba`, with fresh-process
  containment at `b10b8f8`. `bb0a88a` (integration `8f4d6ad`) corrects
  source identity's distinction between global reads and attribute names.
  Separate setup, analysis, reap, profiling and RSS clocks are documented
  in `DYNAMIC_DEADLINE.md`.
- Actual cuTile guarded token ordering and content-free proof metadata are
  in `6e1d3eb`; loop-conflict/allocation refinement is `030494c` and its
  associated follow-ups through `ddc4735`. The 369 scoped correctness
  results and all proof qualifications are in `CUTILE_TOKEN_ORDER.md`.

## Final regression and transport admission

The exact combined source passes **1,991 tests**, with **11 existing
skips** and no failures: all `tests/unit` and `tests/end_to_end` under
the repository's default exclusions. The canonical venv CLI is on PATH,
the pinned-test state and compiler cache are isolated, and GPU-visible
warmup cases run. Every tracked Python source hash and the commit agree
before and after. The skipped optional/version-dependent cases and the
exact command/environment are listed in the raw receipts.

The earlier integration `c3dd382` had 1,956 applicable passes and 11
existing skips after rerunning six environment-only failures and the
initially hidden-GPU cases. Those initial logs remain intact. The final
source's single complete run replaces that aggregation for final-code
regression validation. The paper child-observer adapter also passes its
ten independent tests; it changes no archived study script or source guard.

The READY-only transport helper (`8f60079`, integration `7b8393e`) checks
19 corpus kernels and one nested-JIT controlled builder. All twenty pass
at the final source. Every child validates source/configuration/kernel and
logical/full-backing-storage input identities, reaches READY, and is reaped
without GO or analysis. No fallback runs. The selected cases include
aliases, noncontiguous views, scalar tensors, special scalar values,
dtype constexprs, FP8 storage and generated kernel modules. This is a
representative compatibility check, not coverage of every corpus row.

The initial `7b8393e` run's one FLA mismatch remains archived. It was an
invented unused-global dependency from the `exp` attribute in `tl.exp`;
the fix retains actual helper/module/constant checks. All repeated logical
inputs match. One reconstructed aiter `q_pe_ptr` view has different
nonlogical backing-storage bytes between the two independent admissions.
Each child matches its own parent's complete bytes. No cross-admission
full-storage identity, analysis result or timing comparison is claimed.

Raw root: `evaluation/results/conformance-integration-20260907/` in the
canonical detector checkout. `FINAL_REGRESSION_SUMMARY.json` binds the
final regression receipts; `TRANSPORT_ADMISSION_AUDIT.json` checks all
twenty admissions and records the one backing-storage variation. The final
transport receipt `transport-admission-55adc88.json` has SHA-256
`7af7cf265d0292ff03c535aeda8c64dc525e538ca3a5e1fc3c103c5ea5ce070c`.
The enclosing `MANIFEST.json` binds 27 files, all independently verified;
its SHA-256 is
`a21090ece861c1cd12c46acec071d1d8fb193fe085074ad5fc4b04c05a27ccb3`.

## Real-case confirmation and input validity

Eight fresh final-source confirmations preserve kernel source, grid, seed,
scalar arguments and every named logical tensor byte/layout. Seven remain
zero-report results: embedding, weight norm, both reduced S4/D8 Hadamard
interpreter controls, both full 32-instance Hadamard enums and the
eight-instance GDN2 enum. Full Hadamard symbolic completion is still not
established by these controls.

Retention instead reports 13 conflicts (five WAR, eight RAW). A source
and allocation audit explains why this captured launch is not an accuracy
control: its `do` view has shape `(2,4,8,16)`, all-zero strides and only
four backing bytes, but the kernel uses value-tensor strides 128 and 16
to read element offsets 0 through 1,023. Of those offsets, 1,023 are
outside its allocation. The source's output geometry does not validate
the read footprint. `torch.save`/`torch.load` faithfully preserves the
four-byte storage and its view; process relocation can change apparent
cross-buffer intersections for out-of-allocation addresses.

Static in-bounds reasoning is an explicit model boundary, not a
memory-safety certificate for that launch. Neither the earlier zero
reports nor the new 13 supplies accuracy evidence. The original capture
and observations remain immutable. A separately named legal control
materializes only `do`, keeping source, grid, scalars and logical bytes
identical while changing its strides to `(512,128,16,1)` and allocation
to 4,096 bytes. That final-source control completes with zero reports.
Its changed physical layout is explicit and cannot substitute for the
original captured input. No new upstream-bug claim is made.

The same legal control at frozen `31c48f5` completes with 24 same-instance
WAWs, versus zero at `55adc88`. The exact probe source, kernel, grid, seed,
scalar arguments, logical bytes and complete materialized layout/storage
metadata match. The old in-process versus new subprocess harness boundary
is explicit, and the pair supports no timing comparison. This validates
the lane correction on legal geometry independently of the invalid fixture.

The canonical archive is
`evaluation/results/frontend-final-confirmation-55adc887-20260907/`.
Its `MANIFEST.json` binds 51 files and has SHA-256
`d2b1c189cf576ee7b8eb16ab8b048eb93a85d24a8ba7d5ba7b722a3d68d70488`.
It retains the original nonzero expectation summary, the invalid-footprint
witness and review, both legal controls, their exact before/after comparison,
and the final regression receipts. The root audit rehashes every file and
verifies the legal pair's 24-to-zero result and matching input/source fields.

The final combined source also repeats a 60-second context deadline
control: 60.407444575 seconds through child reap, within the unchanged
half-second criterion. Full wrapper/worker times are 66.110283/95.504648
seconds. The golden successful child retains its complete nine-check
profile. `DYNAMIC_DEADLINE.md` records the source-bound archive and the
context TTIR's exact location-filename-only difference from the older IR.

## Integration with the latest L2 execution policy

The required final pull brought in demo `e5d917d`: ordinary L2 now skips
the independent dynamic frontend after a static decision. Integration
`5c4622f271b2ce481f8713d823db45eed1e1dfef` retains that scheduling and
provenance policy while preserving fatal transport failures whenever
dynamic execution is actually entered. Under default `on-demand`, static
decisions leave dynamic `not-run` with `time_s=null`. L1 and explicit L2
`TRITON_VIZ_EVAL_ALL_FRONTENDS=1` comparison runs retain policy `all`.

Two new integration controls fail against the scheduling-only branch and
pass on the merged harness: all-policy static proof plus transport failure,
and on-demand static abstention plus transport failure, must both fail with
`harness-error` and never reach enumeration. The older transport-failure
test explicitly requests all frontends so that its static proof enters
the child boundary. Thirty-five focused policy/transport controls and
the normal merge checks pass. These changes alter scheduling, not the
reader, solver, enumeration or child cancellation mechanisms validated
at `55adc88`; those earlier source-bound receipts remain historical.

The source and policy descriptions in `L2_FRONTEND_POLICY.md` govern the
new execution mode. Independent frontend-complementarity and required
child-profile controls must explicitly request `all`. Merely registering
an observer does not launch a skipped frontend. New operating-cost
measurements must use the declared on-demand policy and cannot be obtained
by subtracting old interpreter times from the earlier measurements.

The complete suite at exact `5c4622f` passes **2,033 tests**, with 11
existing skips and no failures. The commit and every tracked Python
source hash remain identical before and after the run. This is the final
combined scheduling/repair validation; the 1,991-pass `55adc88` receipts
above retain their earlier tested source and are not relabeled.

Two actual `smoke_add_no` launches also verify the merged scheduler:
default L2 skips dynamic execution with null time and no child receipt;
explicit `all` runs a successful, completed `dynamic-spawn-v1` child.
Source, bound input/storage bytes, specialization and TTIR hash match
across the pair. No timing comparison is adopted. The canonical archive
`evaluation/results/scheduler-integration-5c4622f-20260907/` contains
13 manifest-bound files plus `MANIFEST.json`, all rehashed by the root
audit. Manifest SHA-256:
`e648b29510327016cdf79127e86fd5cf1bf6733c2fb4333b1af072eaad248808`.

## Adoption boundary

The old formal inputs, publications, exclusions, timing distributions and
paper numbers retain their original pin. This combined source needs a
new common-pin numerical sequence. Under Hao's current policy that is
full **L1 then L2**, plus the dependent ablation, controlled sensitivity/
scaling and budget studies and cross-language rederivation. L0 is limited
to expressly selected study controls and targeted regressions.

Before dispatch, adapt the archived coordinators/reporters' level set,
review successor copies of source-guarded optimization variants, install
their explicit child observers, and admit only complete child profiles
for complete-stage cost comparisons. Parent-only RSS does not account for
child memory. The paper's existing `pre-submission/pinned-rerun.md` owns
these remaining measurement obligations and the provisional reusable time
estimate. Neither diagnostics nor reduced successful controls replace
full Hadamard symbolic completion or independent FLA `Loaded` replay.
