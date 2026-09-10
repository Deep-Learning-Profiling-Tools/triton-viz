# Reducing repeated construction in FLA static analysis

## Scope

This change follows a rescreen of old FLA slow rows on detector commit
`34a3e683010a4626c4dbdb4183249811a656d189`, which already contains the
snapshot and arithmetic conflict prechecks. The old stopped rerun is used
only to select candidates and verify captured TTIR. Its timings are not
the before measurements below.

The selected current-code profiles expose two remaining construction
costs. GDN2 backward intra spends 31.93 of 42.68 seconds building HB;
KDA variable-length inter-solve spends 81.07 seconds building HB and
64.95 seconds constructing prechecks, out of 157.61 seconds total.
Log-linear attention backward diag spends 235.41 of 249.11 seconds
constructing prechecks, versus 4.99 seconds in all solver checks.

## Exact sparse HB closure

The Floyd-Warshall recurrence is unchanged. Initial edges are simplified
without assumptions. At each intermediate event, snapshot the previous
layer's non-false incoming column and outgoing row and combine only
those entries. Fold Boolean identities and identical expressions while
retaining every conditional path, cycle, and diagonal constraint.
Snapshotting both vectors ensures all right-hand sides use the same
previous layer even for cyclic graphs.

The matrix and row/column scans take O(n^2) work; path construction takes
the sum of incoming-count times outgoing-count over all layers, with the
same O(n^3) dense worst case. Sparse graphs avoid constructing large
numbers of expressions that are identically false. This changes only
the representation of the closure, including for the dynamic solver
that shares the helper.

## Solver-local expression reuse

The pure-Select necessary-condition path caches native simplification
and linear abstraction by immutable, actual Z3 AST. Every cached subtree
retains its Select-applicability flag. The cache belongs to one two-copy
solver and uses post-substitution expressions, so cross-instance and
same-instance conditions remain distinct. It never caches Solver objects
or SAT/UNSAT results. Reusing a fresh abstraction variable for the same
expression is sound: in every original model it can be assigned that
expression's value independently in each new solver query.

Grid bounds, lane ranges and extra assumptions are assembled into one
common conjunction; its same-instance substitution is also reused.
The source tuple objects and frozen copy contexts are retained and checked
by identity. Replacing any source invalidates this common cache; mutable
constraint sequences conservatively rebuild it on every call. Removing
launch pins therefore cannot retain stronger, stale assertions. Pair
activity, byte conflict and lane/different-block conditions stay local.

Mixed-radix abstraction keeps its existing pair-local bound reasoning.
The new expression cache is only used when that reasoning is absent.
Original HB, reads-from, feasibility checks, domains, snapshots, replay,
and the full and optional solver budgets are unchanged. Only UNSAT of
the original necessary-condition relaxation can skip a complete query.

## Selected before/after validation

Measured on 2026-09-06 UTC with before commit `34a3e68` and after commit
`c37835333a0849f3c22b141fdcdbd36954d51228`, both clean. Each configuration
was run once per version in a fresh process, serially, with all existing
prechecks enabled. No component ablation or official rerun was resumed.

| Captured FLA configuration | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: |
| GDN2 `chunk_gdn2_bwd_kernel_intra` | 42.68 | 6.61 | 6.46x |
| KDA varlen `chunk_kda_fwd_kernel_inter_solve_fused` | 157.61 | 19.71 | 8.00x |
| KDA `chunk_kda_fwd_kernel_inter_solve_fused` | 47.69 | 8.76 | 5.44x |
| Delta-rule varlen `merge_16x16_to_64x64_inverse_kernel` | 45.51 | 7.76 | 5.86x |
| Log-linear varlen `chunkwise_bwd_kernel_diag` | 249.11 | 11.79 | 21.14x |
| RWKV7 varlen `chunk_dplr_fwd_kernel_h` | 7.45 | 3.50 | 2.13x |
| MESA varlen `chunk_mesa_net_fwd_kernel_h` | 7.32 | 3.66 | 2.00x |
| Gated Oja varlen `chunk_oja_bwd_kernel_dhu_blockdim64` | 6.40 | 2.92 | 2.19x |

These times cover the complete selected static pipeline, including
capture, parsing, encoding, HB, formula construction, solver setup,
checks, and replay. They exclude process startup, host compilation, and
the separate dynamic and L1 harness tracks. They are not an estimate of
full-corpus FLA performance. The phase wrappers measure exclusive
construction time separately from actual `Solver.check` time.

The main costs declined as expected. GDN2 HB construction fell from
31.93 to 0.60 seconds. KDA varlen HB fell from 81.07 to 0.92 seconds,
and its precheck construction fell from 64.95 to 10.64 seconds.
Log-linear precheck construction fell from 235.41 to 4.61 seconds,
while its actual checks took 4.99 and 4.71 seconds. Its 4096-entry LUT
and original array representation were left intact.

All eight configurations retained their static decisions, proof scopes,
content qualifications and report locations/types. All 5,073 solver
checks retained their order, query context and SAT/UNSAT result;
there were zero unknowns in either version. Some satisfying witness PIDs
changed with the equivalent formulas. In particular, log-linear retained
both same-instance WAR reports (source lines 1421 to 1440 and 1423 to
1439), with the same exact evidence classification. Its and MESA/Oja's
existing `unhandled term Loaded` differential-check limitation remained;
those checks are not an additional correctness oracle for this change.
The unit equivalence and regression tests supply that validation.

The input spec hash, snapshot-sidecar hash, TTIR hash, seed, L2/fence
policy, Python/package versions, probe hash, solver construction metadata,
and all complete result fields were compared. Only result time and the
concrete satisfying PIDs differ. The source/input data and all query
budgets were held fixed.

The local diagnostic archive is
`evaluation/results/fla_tail_opt_20260906/` in the original detector
checkout (untracked): `baseline/`, `baseline-extra/`, `optimized/`,
`comparison.json`, exact probe and serial driver, and stdout/stderr for
every attempt. `profile_fla_static.py` has SHA-256
`4993b6a47b8f0f8d1314211c24c2fa22845112b40357581d68f378db215610f1`.
It is the same probe for all 16 runs. All attempts completed without the
driver's external 360-second safety timeout. The older saved FLA JSONL
is only the TTIR identity and candidate-selection reference; all before
times in the table were newly measured on `34a3e68`.

## Regression validation

The added tests cover sparse/dense closure equivalence, conditional cycles
and diagonals, independent contexts, cache reuse and Select flags,
negative Boolean polarity, independent copies, snapshot fallbacks,
revoked and mutable assumptions, grid/range/context replacement, and
separate solver ownership. The existing mixed-radix and unknown-fallback
regressions also pass. Focused HB/core tests: 78 passed. Focused conflict
and cache tests: 48 passed. The complete unit and end-to-end suite passed:
`pytest -q tests/unit tests/end_to_end`: **1367 passed, 11 skipped** in
119.42 seconds. All commit hooks passed; formatting preserved the tested
Python ASTs. The full-suite log is included in the local archive.
