# Shared snapshot prechecks for remaining L2 static slow rows

## Scope and baseline

The stopped L2 run at `637f57f` has 16 FLA and two TritonBench_G rows
whose recorded static time exceeds 100 seconds. Eight FLA configurations
were validated in the previous optimization work; this follow-up measures
the other eight FLA configurations and both TritonBench configurations.
It also checks non-varlen log-linear diag (an old outer timeout without
phase data) and KV-cache copy (an old 20.86-second static run).

The twelve fresh before measurements use clean detector commit
`30ab953052c9fac9bdc51448649aaed3dea90469`, which already includes the
snapshot, mixed-radix, sparse HB, and expression-cache optimizations.
The final after measurements use clean implementation commit `995a1ccfb2314144f3f1093d0757f5a9d9ed4cec`.
The saved 637f57f times below identify historical candidates; they are
not the before measurements for this new change. In particular, the
original L0 fast times often represent unsupported inputs, not completed
proofs of the larger L2 fragment.

## Diagnosis and change

Non-varlen log-linear diag still takes 193.21 seconds at the fresh
baseline: 177.65 seconds in actual Solver checks and 13.72 seconds in
solver construction. Six full checks return unknown. Its dq/dk/dv
addresses are linear, so the old applicability gate bypasses the
arithmetic conflict precheck. The complete query nevertheless receives
4096 snapshot-array equalities. The corresponding varlen addresses have
Selects and already enter the existing fast path.

Allow the same necessary-condition precheck when the common premises
contain Selects, even if both event addresses are linear. Determine
this property lazily, only when the addresses do not already qualify,
and cache it with the common conjunction. Replaced source tuples and
mutable constraint sequences retain the existing cache-invalidation
rules. No query decisions are cached.

Every full-query model extends to a model of the relaxed conflict
condition by assigning each fresh abstraction variable the value of its
original expression. Only UNSAT can discard a pair. SAT or unknown
falls back to the original complete query. All original assumptions,
byte-overlap conditions, lane and instance constraints, HB, reads-from,
independent feasibility checks, and full-query budgets remain intact.
There is no kernel-name specialization or LUT rewrite.

## Selected static-pipeline measurements

Measured on 2026-09-06 UTC, once per configuration per clean version,
serially in fresh processes. All prechecks are enabled. Timing includes
capture, parsing, encoding, HB, formula construction, solver setup,
actual solving and internal replay/fallback. Process startup, host
compilation, and separate dynamic/L1 harness tracks are excluded.

| Configuration | Saved 637f57f static (s) | Fresh 30ab953 (s) | Final (s) |
| --- | ---: | ---: | ---: |
| `tb_quantize_copy_kv` | 123.27 | 5.16 | 5.16 |
| `fla_comba_chunk_varlen__chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 133.98 | 1.86 | 1.04 |
| `fla_gdn2_chunk_varlen__chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 133.68 | 1.88 | 0.95 |
| `fla_gated_oja_rule_chunk_varlen__chunk_oja_fwd_kernel_h_blockdim64` | 133.61 | 1.56 | 1.02 |
| `fla_gdn2_chunk_varlen__chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 133.34 | 1.60 | 1.61 |
| `fla_delta_rule_chunk_varlen__chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 128.21 | 1.63 | 0.86 |
| `fla_gla_chunk_varlen__chunk_fwd_kernel_h` | 122.37 | 0.98 | 0.64 |
| `fla_gdn2_chunk_varlen__chunk_gdn2_fwd_kernel_inter_solve_fused` | 122.03 | 19.23 | 19.54 |
| `fla_delta_rule_chunk_varlen__chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 108.62 | 1.52 | 1.51 |
| `fla_log_linear_attn_chunk__chunkwise_bwd_kernel_diag` | phase unavailable | 193.21 | 11.79 |
| `tb_chunk_retention__chunk_retention_bwd_kernel_dqkv` | 153.05 | 3.03 | 3.00 |
| `tb_kv_cache_copy` | 20.86 | 1.55 | 1.59 |

The new log-linear change reduces static time to 11.79 seconds
(16.39x), with zero unknowns. Actual Solver checks take
4.12 seconds and solver construction takes
1.32 seconds. The common feature scan is
performed only when needed. An earlier clean candidate, `66c1a05`,
performed it eagerly and increased KV-copy precheck construction by
1.50 seconds; that measured regression motivated the lazy scan and its
dedicated regression test. All intermediate attempts are retained.

Most hundred-second rows had already become fast under the earlier
generic optimizations. Their saved-to-final differences must not be
attributed entirely to this small follow-up. These selected timings do
not establish a full-corpus speedup or a component ablation result.

## Decisions and evidence

Eleven configurations retain every static decision, proof extent,
qualifier, diagnostic field, and report location/type. GDN2 inter-solve
can choose different satisfying PIDs for the same grid-fragility
reports; those alternate witnesses are preserved in the comparison.
All final runs have zero unknowns.

Log-linear improves from solver abstention to two exact, content-qualified
same-instance WAR reports: dq source lines 1421 to 1440 and dv lines
1423 to 1439. These locations/types match the earlier varlen result.
The witnesses come from the original complete solver, including launch
requery. They are not replay-confirmed: the current C2 replay returns
unavailable for same-PID reports, and C3 retains its existing
`unhandled term Loaded` differential-check limitation. These reports
are not a claim of two new independently confirmed real-world bugs.

Both clean versions have identical input-spec and snapshot-sidecar
hashes, freshly compiled TTIR, seed, L2/fence-order configuration,
Python/package versions, profiling code and enabled precheck settings.
The saved timeout row lacks TTIR metadata; its fresh before/after TTIR
hashes match each other, without claiming identity to missing old data.

## Verification and archive

The twelve new regressions cover linear cross/intra disjointness, real
and partial-byte overlap, duplicate lanes, tuple/list invalidation,
immutable feature reuse, infeasible snapshots including array congruence,
unknown fallback, disabled prechecks, and avoiding redundant common
scans for already eligible indirect addresses.

`pytest -q tests/unit tests/end_to_end`: 1379 passed, 11 skipped.
All commit hooks pass. Formatter changes preserve the tested Python ASTs.

Raw diagnostics remain untracked in the original detector checkout at
`evaluation/results/static_growth_opt_20260906/`: baseline and final
profiles, intermediate attempts, exact phase probe, serial driver,
comparison code/JSON, source and input hashes, all checks, stdout/stderr,
and full-suite logs. The final clean profiles are in `optimized-final/`.
The phase probe has SHA-256
`364aa7318e3d69cad5d95457f5faf087f363b8bcb9afb5bc8dfc250bbeafffce`.

No official pinned rerun or deferred ablation was resumed. Paper
Section 4 documentation and Section 6 ablation remain in the existing
paper TODO; no manuscript performance claims are updated here.
