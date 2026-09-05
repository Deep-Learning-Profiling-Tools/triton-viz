# Change-surface run at L1: the 492 pinned-abstain real-code rows

Date 2026-09-04. Detector commit 5ba8b6a (branch `route1-concrete-enumeration`), ladder level L1, per-row budget 200 s, jobs=1, seed 0; triton 3.6.0, torch 2.10.0+cu128, z3 4.15.3. Dataset: `evaluation/results/change_surface_L1.jsonl` (gitignored; header stamps level, budget and commit). Rows: every real-code row the pinned L0 run (`PINNED_fb91fc0.jsonl`) left as `abstain` (492 of 1062; the 6 timeouts and 5 capture failures are outside the L1 rung's reach and were not rerun).

This is the change-surface diff the design (paper repo `design-route1-concrete-enumeration.md`, section 7 step 4) requires before any paper use of L1: it says what the rung decides, what it refuses and why, and where the remaining abstentions come from. It is NOT a pinned rerun: L0-decided rows were not rerun here (the selective-pricing check is the pinned rerun's job).

## Headline

| outcome | rows | share of 1062 |
|---|---|---|
| proved@enum | 391 | 36.8% |
| race@enum | 16 | 1.5% |
| proved@T1 (decided by commits since the pin, not by the rung) | 1 | |
| still undecided | 84 | 7.9% (was 492 = 46.3%) |

The rung decides 407 of the 492 (82.7%). Every decision is at the analyzed-launch extent with `content_fragile=True`: these scalar arguments, this grid, THESE tensor contents.

## Residual by refusal kind (84 rows)

| kind | rows | what it is |
|---|---|---|
| interpreter-error | 29 | the Triton interpreter itself cannot run the kernel (reproduced with the plain C2 replay recorder, no taint patches): 9 `'int' object has no attribute 'to'`, 7 `_semantic` helper-call failures, 2 tuple-unpack, 2 `None` to tensor, 2 `float + None`, 7 singletons (inline asm, `tl.assume` on rebuilt inputs, ...) |
| cutile-no-interpreter | 23 | cuda.tile rows: no interpreter exists, the rung cannot run (the design's fixed floor) |
| row-crash | 12 | the harness subprocess died without writing a row (see the crash section) |
| atomic-return | 7 | an atomic return value reaches a host branch (5: masked_scatter/masked_select part-sum, mm_streamk first_wave, spinning_lock_reduction, la_persistent_paged) or a footprint position through memory (2: nll_loss fwd/bwd): footprints are not per-instance determined |
| projected-cost | 6 | 10240-instance chunked/paged prefill kernels at 96 to 111 ms per instance (projected 17 to 19 min) and two 8192-instance template-attention kernels at 302 to 306 ms (projected 41 min); refused 5 s in |
| instance-ceiling | 4 | 131072 to 2031616 instances, over ENUM_MAX_INSTANCES = 65536; refused before executing |
| row-timeout | 3 | the whole subprocess exceeded 200 s (rope_fwd_3d, gdn2 fused_recurrent, iplr fused_recurrent bwd; see the crash section) |

Residual by corpus: fla 17, tritonbench_meta 11, tritonbench_g 11, aiter_ops 10, flaggems 10, tilebench_cutile 23 (all cuTile), torchao 1, tilebench 1, liger 1; flagattn and tutorials 0.

## By the pinned static-refusal family

| static family (pinned) | rows | proved@enum | race@enum | residual |
|---|---|---|---|---|
| indirect-address | 229 | 189 | 8 | 32 |
| control-flow | 84 | 68 | 1 | 15 |
| other | 76 | 62 | 6 | 8 |
| nested-loop | 51 | 32 | 0 | 19 |
| data-dependent-bound | 40 | 34 | 1 | 5 |
| spin-shape | 9 | 5 | 0 | 4 |
| solver | 3 | 1 | 0 | 2 |

## Per corpus

| corpus | rows | proved@enum | race@enum | residual |
|---|---|---|---|---|
| fla | 226 | 206 | 3 | 17 |
| aiter_ops | 62 | 52 | 0 | 10 |
| tritonbench_g | 56 | 37 | 8 | 11 |
| flaggems | 36 | 23 | 3 | 10 |
| torchao | 36 | 34 | 1 | 1 |
| tilebench_cutile | 23 | 0 | 0 | 23 |
| tritonbench_meta | 20 | 9 | 0 | 11 |
| flagattn | 17 | 17 | 0 | 0 |
| tilebench | 11 | 9 | 1 | 1 |
| liger | 4 | 3 | 0 | 1 |
| tutorials | 1 | 1 | 0 | 0 |

## Cost

- enum run time over the 453 rows that executed: median 0.17 s, p90 3.7 s, p95 11.7 s, max 185.9 s.
- per-instance interpreter time: median 10.5 ms, p90 90 ms, max 880 ms (not constant across instances: data-dependent trip counts, pid branches, triangular workloads).
- row wall time (compile + both symbolic tracks + the rung): median 3.6 s, p95 63.1 s, max 200.2 s; the whole run took 1.42 h at jobs=1. Before the projected-cost refusal the first 52-row stretch averaged 22.6 s per row (five rows burning the full budget); with it, 10.1 s.

## The 16 race@enum rows: triage

None of these is a new finding; the Leads-30 counting discipline holds (none counted). Grouped by what the witness actually says about the CAPTURED contents:

1. Capture-rebuild artifacts (11): the captured launch rebuilds tensors above the 8192-element value-snapshot cap from their descriptors (`randint` for integer tensors), so index tensors carry contents the real call never passes. destindex_copy, destindex_copy_kv1, destindex_copy_kv2, quantize_kv_transform (randint destinations with replacement: the Leads-30 reading, same as casebook A6); kv_cache_filling fwd/quant (all-zero captured BlockOffsets: two instances fill one block); context_attn_llama (B_Start_Loc rebuilt all-zero: every batch row writes Out[0]); moe_jagged_rowwise (randint jagged offsets: duplicate lanes in one store); masked_select write_back (`part_sums` is a 9-element value snapshot of the REAL mask's prefix sums while the 32768-element mask itself is rebuilt at random, so block 2 writes [4123, 6159) and block 3 starts at 6128: a 31-row overlap the real inputs cannot produce); radix_sort (`global_ones` = 499384 is a snapshot, the rebuilt input has 499185 zeros: the zero/one partitions overlap); unique_large (the `idx` tensor is rebuilt as random int64 in the range 4e6 to 3.9e10 and used as addresses). These rows say: the rung reads contents, so it is the first rung to expose capture fidelity; the fix is in the corpus capture (snapshot the index tensors or rebuild them with the real semantics, e.g. `randperm`), not in the rung.
2. Out-of-bounds-induced (2, the casebook A8 class, excluded by the paper's in-bounds premise): iplr fused_recurrent_varlen bwd (the known A8 shape: instance (0,2,0) indexes past the 8192-element state into the neighbouring allocation); chunk_gla_fwd A intra_sub_intra_merge (A captured with 4096 elements while the kernel indexes it as NK x n_bh x T x BC = 32768: the reads run into the adjacent clone).
3. Model races with a benign effect, worth a casebook note (3): unique_dup simple_unique_flat (line 45 `tl.store(data_out + cumsum, a, mask)`: duplicate sorted values share a cumsum slot, so two lanes of ONE store write the same address with the SAME value; the model's duplicate-position query reports it, the A1 shape); ttt layer_norm_bwd chunk / fused_chunk (line 439: each program owns BS = 2 rows but stores a BT = 32-row `dx` tile, so neighbouring programs overwrite 30 shared rows with identical values; the captured constexprs are the real launch's). Both were among the Leads-30 candidates the external tools also flagged.

The design's section 7.3 expectation that the three permutation-scatter Leads-30 rows come out proved@enum did NOT hold (masked_select and radix_sort are race@enum, nonzero crashed): the rung is right about the rebuilt contents, which are internally inconsistent; the expectation assumed the captured inputs were the real permutation.

## Drift against the first stretch

The first 52 rows (aiter_ops) were also run under the pre-fix semantics (spin pre-gate on the reader's `spin-shape` kind, same-instance writes counted against the premise, no projected-cost refusal, 150 s cap). 44 rows unchanged; 7 abstentions became proved@enum (2 mis-gated carried-value `scf.while` rows, 4 same-instance in-place updates, 1 budget-edge row); 1 row went the other way, rope_fwd_3d (11840 instances at 6.8 ms, 81.9 s in the first stretch) hit the 200 s row budget in the full run: a budget-edge row whose wall time depends on machine load (see the crash section).

## Crashes and timeouts

All 15 rows were re-run through the harness at L0 and at L1 with signal capture (`repro_crash.py`, 2026-09-04).

**row-crash (12): deterministic, all inside the L1 rung, the out-of-bounds class.** Every one of the 12 abstains cleanly at L0 in about 3 s and dies at L1 within 3 to 7 s: 8 with SIGSEGV, 4 with SIGABRT from glibc's heap checks (`corrupted size vs. prev_size`, `free(): invalid size`). The rung executes the kernel's memory operations on raw host pointers, so an out-of-bounds store on the rebuilt inputs corrupts the process heap; the plain C2 replay recorder would do the same. Two rows produced output before dying (nonzero emitted `race@enum` and then aborted at teardown; chunk_gla_fwd split raised a nonsensical AttributeError on the recorder object, the signature of a corrupted heap), so a verdict from a kernel that writes out of bounds is not trustworthy even when the process survives. The subprocess isolation contained every crash (no other row was affected), but the paper's in-bounds premise, which the symbolic frontends enforce by fail-stop, is NOT enforced by the rung today. Recommended fix (a semantic change, not landed): check every access's active-lane address range against the cloned tensors' spans in the before-callback and refuse by name (`out-of-bounds`) before the interpreter dereferences; that turns the 12 crashes into named abstentions and also converts the two OOB-induced `race@enum` rows (iplr varlen bwd, chunk_gla merge) into honest refusals. The affected rows: fla iplr fused_recurrent_varlen fwd (the A8 fwd twin), flaggems cross_entropy_loss bwd x2 and nonzero, tritonbench_g chunk_gla_fwd split, fused_rotary_embedding (the Leads-30 row whose OOB claim was "refuted on verify"; it corrupts the heap here), rotary_emb_nopad v2, softmax_reducev, token_attn llama2 / mistral / reduceV, tritonbench_meta grouped_gemm.

**row-timeout (3).** Two are not the rung's cost: fla gdn2 fused_recurrent and iplr fused_recurrent bwd sit on the dynamic track's 60 s watchdog already at L0 (pinned wall 64 s, `dynamic.status = timeout`); in reproduction the L0 row itself ran to the 200 s budget (the SIGALRM watchdog did not interrupt the interpreter), while at L1 both rows decided `proved@enum` in 66 to 67 s with the rung taking 2 to 3 s (8 and 4 instances). They are budget-edge rows of the SYMBOLIC tracks under load. The third, aiter_ops rope_fwd_3d (11840 instances at 6.8 ms, decided in 81.9 s in the first stretch), exceeded 200 s in the full run and 260 s in reproduction: a regression of the memory-taint patch, not of the rung's execution. Diagnosis (standalone, 60 s watchdog): the run phase is unchanged at 6.86 ms per instance (8553 of 11840 in 60 s); the premise check had become quadratic (a full scan of the interval buffer per value-source load, 35520 of them over the 106560 operations' intervals), and it ran OUTSIDE the watchdog, so the row blew its budget instead of refusing by name. Fixed (bisection over the op-sorted buffer; the analysis phase now runs under the remaining budget): the row decides `proved@enum` through the harness in 157 s (84.7 s execution, 68.3 s analysis). The remaining 68 s is the cross-instance sweep over 97,593,600 per-lane intervals: the kernel's accesses are strided, so no lanes coalesce (916 intervals per operation, 2.3 GB of interval columns). That is the rung's real scalability limit for strided kernels on large grids (a 65536-instance row of this shape would need about 12 GB) and is recorded as an open item: represent an operation's footprint as a bounding box plus a uniform-stride run and sweep boxes, materializing lanes only where boxes of distinct instances overlap.

## Addendum 2026-09-05: the in-bounds premise enforced in the rung

Hao's decision after the crash analysis above: the rung now checks every access's active lanes against the tensor arguments' storages (the cloned allocations) BEFORE the interpreter dereferences, and refuses by name (`out-of-bounds`, naming the access, the instance and the offending byte). Masked-off lanes may point anywhere. Measured cost: about 4 microseconds per access (a min/max over the lanes and one bisection), one to four percent of the rung's end-to-end time.

The 14 affected rows re-run through the harness at L1 (same commit lineage, 200 s budget): all 12 former crashes and both out-of-bounds `race@enum` rows (iplr fused_recurrent_varlen bwd, chunk_gla_fwd A intra_sub_intra_merge) now end as `out-of-bounds` refusals in 2.6 to 6.5 s, exit code 0, no signal. Cross-validation on the 51 interpreter-decided benchmark rows is unchanged (35 agree, 16 disqualified by name, 0 disagree). Restated headline for the 492 rows under the enforced premise: 391 proved@enum, 14 race@enum, 1 proved@T1, 86 residual (8.1% of 1062), of which 14 `out-of-bounds`, 29 interpreter-error, 23 cuTile, 7 atomic-return, 6 projected-cost, 4 instance-ceiling, 3 row-timeout (rope_fwd_3d now decides, see the timeout section; the two fused_recurrent rows remain symbolic-track budget-edge rows). The 14 race@enum rows: 11 capture-rebuild artifacts and 3 benign-effect model races; none counted.
