# Complete-query reuse for remaining L2 timeout rows

## Scope and source pins

The interrupted L2 run at `637f57f` contains 13 outer 200-second timeout
rows across FLA and TritonBench G. Non-varlen log-linear diag received the
previous static optimization. This follow-up rescreens the other 12 rows
and checks log-linear diag and GDN2 inter-solve as large-snapshot regressions.
An outer timeout does not identify a 200-second static phase or Solver query.

The fresh before source is clean commit
`515466bd065fe12c55a7f28e3ee51d9633ea83a6`. The final implementation is
`b52ce582fab709b8c4e68782700a1413f5de668d`. All selected measurements use L2, fence ordering enabled,
seed 0, the original inputs and query budgets, and fresh serial processes.
No official pinned rerun or deferred component ablation was resumed.

## Static rescreen and regression measurements

Before this change, the 12 remaining historical timeout rows already take
0.005–9.424 seconds in the static pipeline. Eleven produce proofs: eight
at this-params-any-grid and three at this-params-this-grid. TB chunk-gate
backward still refuses `load of a non-pointer value`; its short time and
zero Solver checks are not a proof. The 12 pipelines total 26.822 seconds,
with 986 decided checks, no unknowns, and maximum check time 0.399 seconds.

Seven of these cases already had cheap L0 static proofs: HGRN forward and
backward, GLA, Oja, Delta-rule and RWKV6 recurrent forward, and TB matmul.
Their old static times were 0.263–1.033 seconds and their old dynamic
tracks reached roughly 60 seconds. KDA forward and IPLR backward instead
had cheap L0 refusals; the larger L2 fragment now proves them. ABC backward
K was expensive even in L0 (159.73 seconds static). This distinction is
preserved in the raw static-rescreen summary and old-row source references.

The final table includes the two additional snapshot regressions. Static
time includes capture, parsing, encoding, HB, formula construction, Solver
setup/checking, and internal replay/fallback; it excludes process startup,
host compilation, and the separate dynamic and concrete-enumeration tracks.
Each entry is a single selected diagnostic, not corpus overhead.

| Configuration | Fresh baseline static (s) | Final static (s) | Final result |
| --- | ---: | ---: | --- |
| `fla_abc_chunk__chunk_abc_bwd_kernel_K` | 5.985 | 5.783 | proved@T1 |
| `fla_delta_rule_fused_recurrent__fused_recurrent_delta_rule_fwd_kernel` | 1.588 | 1.563 | proved@T1-launch |
| `fla_gated_oja_rule_fused_recurrent__fused_recurrent_oja_fwd_kernel` | 0.443 | 0.387 | proved@T1 |
| `fla_gdn2_chunk_varlen__chunk_gdn2_fwd_kernel_inter_solve_fused` | 19.470 | 18.338 | proved@T1-launch+content |
| `fla_generalized_delta_rule_iplr_fused_recurrent__fused_recurrent_bwd_kernel` | 9.424 | 8.659 | proved@T1 |
| `fla_gla_fused_recurrent__fused_recurrent_fwd_kernel` | 0.747 | 0.717 | proved@T1 |
| `fla_gsa_fused_recurrent__fused_recurrent_bwd_kernel` | 5.258 | 4.755 | proved@T1 |
| `fla_hgrn_fused_recurrent__fused_recurrent_hgrn_bwd_kernel` | 0.475 | 0.426 | proved@T1 |
| `fla_hgrn_fused_recurrent__fused_recurrent_hgrn_fwd_kernel` | 0.244 | 0.229 | proved@T1 |
| `fla_kda_fused_recurrent__fused_recurrent_kda_fwd_kernel` | 0.410 | 0.368 | proved@T1 |
| `fla_log_linear_attn_chunk__chunkwise_bwd_kernel_diag` | 11.892 | 11.621 | races |
| `fla_rwkv6_fused_recurrent__fused_recurrent_rwkv6_fwd_kernel` | 1.202 | 1.117 | proved@T1-launch |
| `tb_chunk_gate_recurrence___bwd_recurrence` | 0.005 | 0.005 | unsupported |
| `tb_matmul_kernel` | 1.038 | 1.032 | proved@T1-launch |

All 14 inputs, TTIR hashes, probe identities and analysis settings match.
Ten complete result records are identical after removing timing and verified
checkout-path differences. Delta forward, GDN2 inter-solve, RWKV6 forward,
and TB matmul choose different valid grid-fragility witness PIDs; every
coordinate change remains in the comparison JSON. Verdicts, report source
locations/types, qualifiers and proof scopes are unchanged. The matmul
source-path alias is accepted only after hashing both actual files.

Total selected static time is 58.184 to 55.002 seconds. All 14 entries are
slightly faster in these samples; this is not a statistical or whole-corpus
speedup claim. Checks decrease from 3027 to 2949, with no unknowns before
or after; the largest final individual check is 0.398 seconds. Log-linear
diag retains the same two exact content-qualified WAR reports and existing
unavailable same-PID replay/Loaded differential-check limitations.

## Diagnosis and implementation

HGRN forward's interpreter capture retains 194 records, producing 388
symbolic events. Its static frontend summarizes the recurrence instead.
A truncated clean-baseline cProfile run reaches the original dynamic
60-second watchdog with 33.747 seconds in Solver construction, 21.096 in
HB, and only 1.513 in actual checking. It builds the base Solver 1074 times,
repeatedly asserting the same constraints and tautological HB diagonals.
These exclusive categories diagnose a truncated instrumented run; they
are not a completed runtime or a speedup denominator.

Four exact changes address that repeated work:

1. Cache the complete base conjunction, checking current source containers,
   elements, HB rows and diagonal by identity before reuse. Include grid,
   ranges, reads-from, coherence, counting, value causality and extras.
   Omit only literal `Not(False)` tautologies. Conditional/true cycles stay
   constrained. Nonstandard containers, coercions and one-shot iterables
   retain the original add path.
2. When `reads_through` has no entry for a pair, synchronization is
   identically false: return its program-order expression directly.
   Existing entries, including a literal false value, retain the original
   scope checks. Dependency, fence and activity conditions remain intact.
3. Normalize the complete base plus cross/same-instance conditions, lane
   identity and the original race expression. Before creating a Solver,
   reuse a directly proved symbolic UNSAT only when that complete AST and
   Z3 context are identical. Retain the AST/context and clear the cache at
   the next `find_races` invocation. SAT models and unknown results are
   never reused; enumeration-only UNSAT never enters this cache. Feasibility
   remains an independent query with live launch premises and extras.
4. Fold ordinary access modes `active AND True` to `active` and
   `active AND False` to false at lowering. Symbolic modes retain their
   condition. Two literal-false writes can skip the conflict formula;
   conditional writes and atomic scope/partial-byte exemptions keep their
   original checks.

The changes preserve every captured access and iteration, all original
formula premises, proof extents, and query/watchdog/enumeration budgets.
There are no kernel-name cases or assumed buffer-disjointness rules.

## Selected dynamic results

The dynamic probe first compiles and runs the ordinary static track, then
times interpreter capture and finalization. Static and compilation times
are outside the dynamic number. Ordinary timing wrappers are enabled on
both sides; cProfile is confined to the separately labeled diagnosis.

| Configuration | Fresh baseline | Final dynamic | Interpretation |
| --- | --- | ---: | --- |
| HGRN recurrent forward | No completion before external termination; separate cProfile run hits 60-second watchdog | 38.365 s | Completes within the original budget |
| HGRN recurrent backward | 60.234 s, watchdog timeout | 111.214 s | Over budget; watchdog exception was ignored, unresolved |
| TB matmul | 11.370 s, completed | 2.501 s | Completed, 4.55x in this selected comparison |

For HGRN forward, all 194 records/388 events remain. The final pass constructs
31525 pair formulas but performs only 1726 full pair Solver decisions,
reusing 29799 identical UNSAT results; feasibility adds one separate check.
There are no reports or unknowns. Solver construction totals 0.364 seconds
and actual checks 2.182 seconds. The uncensored matmul before/after probes
match exactly and preserve their dynamic result and static proof scope.
No exact forward speedup ratio is claimed from a censored or cProfile run.

HGRN backward is explicitly not solved within budget. Its clean final raw
harness record says `ok` after 111.214 seconds with zero reports, but its
log records the 60-second `TimeoutError` being ignored in `AstRef.__del__`
through `Z3_dec_ref`. The comparison annotates it as
`over-budget-watchdog-failed`, not as a budget-valid success. Its earlier
candidate diagnostic still returned an ordinary 60.257-second timeout.
The complete over-budget run spends 37.978 seconds constructing pair
formulas, 19.043 in conflict prechecks, 39.556 in remaining capture/query
bookkeeping and normalization, 7.667 in HB, 0.941 in Solver setup, and 5.809
checking. Its longest single check is 0.00215 seconds.

## Remaining work and watchdog evidence

The next performance target is avoiding repeated pair-formula construction
before the full-query cache lookup, using exact normalized address and
activity/mode conditions. Any such reuse must still include both HB
relations, byte widths, instance/lane constraints and all current premises.
It must not merge iterations merely because their source locations match.
That further optimization is not implemented by this commit.

The watchdog defect is also observed in an ordinary baseline forward run:
the one-shot SIGALRM exception lands inside a Z3 destructor, Python ignores
it, and the diagnostic is stopped externally after 214.8 seconds of total
process elapsed time. That is a censored process duration, not a completed
dynamic measurement. The clean backward run above independently reproduces
the problem. The 200-second diagnostic parent limit remains in force for
final runs. A robust dynamic-stage hard timeout needs supervision that
cannot be lost in a destructor, while retaining static results and the
later harness stages; the current optimization does not change the watchdog.
These observations do not establish the cause of every historical timeout.

The other nine rescreened rows have fresh static measurements but no new
complete dynamic measurements in this follow-up. They must not be counted
as newly resolved dynamic timeouts.

## Validation and raw evidence

`pytest -q tests/unit tests/end_to_end`: 1448 passed, 11 skipped.
The 69 added regressions cover constraint mutation, conditional HB cycles,
query isolation, context identities, complete-formula differences, SAT
models, unknown handling, launch-only enumeration, feasibility, conditional
accesses and atomic scope/byte-overlap cases. They pass again after formatting.
All commit hooks pass, including Ruff and mypy; the committed Python ASTs
match the final checked AST manifest. An earlier full run lacked CLI entries
on PATH (five environment failures); that log remains alongside the corrected
full passing run.

Raw files are untracked in the original detector checkout at
`evaluation/results/remaining_timeout_opt_20260906/`. They include:

- `baseline/` and `snapshot-baseline/`, `optimized-final/`, complete static
  comparison JSON/Markdown, the initial path-normalization failure, and
  exact source-path/hash evidence.
- `dynamic-baseline/`, `dynamic-baseline-tb/`, `dynamic-final/`, and
  `dynamic-comparison.json`, preserving raw and effective timeout statuses.
- All probes/drivers, commands, input/TTIR/source/probe hashes, check logs,
  stdout/stderr, external termination records and test logs.
- Clearly named prototype/candidate and cProfile diagnostics, including the
  initial failed/segfaulting probe. These remain development evidence and
  are not component-ablation results or final after measurements.
- The historical timeout-selection records and `artifact-sha256.json`.

Some dynamic JSON spans inherit the timing helper's default phase label
`static`; those counters were active only inside `_dynamic_track` and do
not belong to the separately saved static result. Source/probe identities
and timing boundaries are recorded explicitly. Paper Section 4 follow-up
and Section 6 deferred ablation remain tracked in the paper TODO; no
manuscript evaluation numbers are changed by this work.
