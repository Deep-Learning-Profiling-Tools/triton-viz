# Sweep Report — Triton Race Detector Evaluation

**Date**: 2026-07-12 · **Detector**: `race-detector-z3-demo` @ `a364ebb` · **Env**: triton 3.6.0, torch 2.10.0+cu128, z3 4.15.3, numpy 2.4.2, Python 3.12 · **Capture GPU**: RTX 4090 (sm89); sweeps are GPU-independent (TTIR host-compiled at sm80) · **Seed**: 0 · sweeps run at `--jobs 8` (definitive paper runs to be re-done at `jobs=1`)

---

## 1. Corpora & provenance

| Corpus | Rows | Source pin | Kind |
|---|---|---|---|
| tritonracebench (+golden_smoke/rmw_sync/await_sync) | 56 (+7+9+9) | in-repo, hand-labeled yes/no pairs | labeled micro-benchmark |
| tutorials | 9 | triton 3.6 tutorials, vendored | real code, race-free labels |
| liger | 23 | liger-kernel 0.8.0 (PyPI pin, upstream `c4b16d4`) | real code |
| tritonbench_g | 202 | thunlp/TritonBench `603e28a`, vendored | real code (GitHub-crawled) |
| fla | 378 | fla-core 0.5.1 (PyPI pin, upstream v0.5.1 `2e38c1f`) | real code |
| flagattn | 28 | flag_attn git-pin `41fc31d` (no PyPI) | real code |
| flaggems | 82 | flag_gems git-pin `1051e56c` (PyPI stale) | real code, atomic-heavy |

All real-code rows carry heuristic `race-free` labels (production code); the micro-benchmark carries ground-truth yes/no labels with planted witness lines. Captured launches rebuild deterministically: int/bool tensors ≤8192 elements are value-exact snapshots; every results header pins package versions + upstream commits.

## 2. Ground-truth scorecard (tritonracebench, 56 rows)

**precision = recall = 1.0 · witness-matched 25/25 · ladder audit zero (ladder-unsound=0, replay-unsound=0) · mutation sensitivity: all applicable proofs flip under at least one mutant.**

Terminals: race-confirmed 12, races-unclassified 13, race@interp 7, race-unconfirmed 1, proved@T0 7, proved@T1 8, proved@T1+assumes-termination 4, proved@interp 4. Companion micro-suites: golden_smoke 7 (3 race-confirmed / 3 proofs / 1 abstain), rmw_sync 9, await_sync 9 (3 conditional proofs + 6 detected races).

## 3. Real-code corpora (722 rows)

| Corpus | Rows | Decided-clean | — static (T0/T1) | — interp | Abstain | Races-unclassified¹ | race@interp | Other² |
|---|---|---|---|---|---|---|---|---|
| tutorials | 9 | 5 (56%) | 3/2 | 0 | 4 | 0 | 0 | 0 |
| liger | 23 | 17 (74%) | 0/17 | 0 | 5 | 0 | 0 | 1 |
| tritonbench_g | 202 | 116 (57%) | 30/69 | 17 | 57 | 22 | 3 | 4 |
| fla | 378 | 134 (35%) | 15/107 | 12 | 227 | 9 | 1 | 7 |
| flagattn | 28 | 1 (4%) | 0/0 | 1 | 17 | 10 | 0 | 0 |
| flaggems | 82 | 42 (51%) | 11/22 | 9 | 36 | 1 | 2 | 1 |
| **Total** | **722** | **315 (44%)** | 59/217 | 39 | 346 | 42 | 6 | 13 |

¹ static-track SAT verdicts whose witnesses lie OUTSIDE the launch grid (T1 any-grid semantics vs wrapper-coupled launches) — every instance checked has out-of-extent witness pids; resolved by the queued launch-scoped verdict tier (TODO §3c).
² compile-error / timeout / crash.

Ladder audits: **PASS on every corpus** (ladder-unsound = replay-unsound = 0 everywhere).

## 4. Genuine races found: 3 (all triaged, all fixed upstream)

| # | Row | Mechanism | Scope | Upstream fix |
|---|---|---|---|---|
| 1 | `tb_nested_loops_processing` | kernel never reads `program_id`; grid=(2,) → both programs write identical `out_ptr` tiles (44 WAW witnesses, value-benign) | global, inter-CTA | [TritonBench#10](https://github.com/thunlp/TritonBench/pull/10) |
| 2 | `tb_quantize_kv_copy` | scatter through `Dest_loc` with real duplicate destinations (snapshot-faithful; witness pids match duplicate positions) | global, inter-CTA, data-dependent | [TritonBench#11](https://github.com/thunlp/TritonBench/pull/11) |
| 3 | `fla_based_fused_chunk` fwd | `z` store address omits the `i_v` grid axis → NV programs write identical values unsynchronized; bwd twin guards with `if i_v == 0`, fwd omits it | global, inter-CTA, same-value WAW | [fla#1018](https://github.com/fla-org/flash-linear-attention/pull/1018) |

All three: machine-generated witnesses first (detector-found), seed-independent, triage only adjudicated the heuristic labels. FlagAttention and FlagGems: zero genuine races on every decidable row — notably the atomic-heavy FlagGems families (bincount/histc/scatter_reduce/index_reduce with duplicate indices) all PROVE clean, `vdot`'s atomic accumulate at T0.

## 5. Triage ledger — every surviving race report accounted

| Row | Verdict | Mechanism class |
|---|---|---|
| tb_nested_loops / tb_quantize_kv / fla_based | **genuine** ×3 | see §4 |
| tb_masked_select | interpreter-artifact | Python `and` on block tensors (interpreter truthiness drops mask terms) |
| flaggems_weight_norm | interpreter-artifact | same `and`-truthiness class, 3rd instance |
| tb_triton_argmax (crash row) | interpreter-artifact | same class inside C3 differential replay → OOB native load, SIGSEGV |
| tb_cache_transform | detector bug — **fixed** | reduce folded over one symbolic lane fabricated nondeterministic WARs; reduce family now gated out of event addresses |
| flaggems_embedding_dup | detector bug — queued | two-copy lane model lacks same-axis arange coupling → phantom intra-instance WAW |
| tb_token_softmax_bloom/llama | retired | randint-rebuild infidelity; value snapshots flipped both to proved@interp |

## 6. Detector defects surfaced by this evaluation round

1. **Reduce single-lane fold** (fabricates races in address position) — **FIXED**: reduce family gated in `_VALUE_DEPENDENT_ADDRESS_OPS`, pinned by test; affected row now abstains deterministically.
2. **`and`-truthiness interpreter divergence** — 3 instances across 2 corpora (fabricated WAW ×2, replay SIGSEGV ×1). Queued: pre-trace AST scan for BoolOp over tensors → mark interp-divergence-suspect, refuse replay (TODO §3f).
3. **Two-copy lane-model coupling** — same-axis arange vars must be equal per copy (TODO §3h; interim fail-closed gate proposed).
4. Philox/math-patch interp gap (`Patching math ops not yet supported`, flagattn dropout bwd) — small, queued.

## 7. Abstention taxonomy → queued lifts

| Class | Rows (attributed) | Lift |
|---|---|---|
| indirect-address (loaded values in addresses; varlen `cu_seqlens`/`chunk_indices`, `block_tables`) | fla 147 + flaggems 12 + TB + liger | §3d snapshot-select extension to the COMPILED track |
| pid-affine loop bounds (`(pid+1)*BLOCK`-style, flash-attention causal loops) | flagattn 14 + flaggems 12 | §3g lift — bounds affine in pid enter the iteration-existence premise |
| wrapper-coupled any-grid (races-unclassified) | 42 rows across 4 corpora | §3c launch-scoped verdict tier (advisor decision) |
| nested loops | fla 20 + flaggems 6 + TB 4 | §3e reader support (interp already rescues some) |
| data-dependent loop bounds (paged attention `context_lens` etc.) | fla 19 + flagattn 1 + flaggems 1 | §3e snapshot-lifted loop bounds |
| unstructured control flow (`cf.cond_br`) | flagattn 2 + flaggems 3 + TB 2 | §3e path-condition encoding |
| carried-value spin (`mm_streamk` stream-K spinlock) | flaggems 1 | S6 await-abstraction extension — first production instance |
| runtime-codegen kernels (FlagGems pointwise_dynamic) | 3 filtered at capture | source-embedding capture scheme (backlog) |

## 8. Reproduction

```
uv run python -m evaluation.runner --corpus <name> [--jobs 8]   # per-corpus sweep
uv run python -m evaluation.report                              # regenerate RESULTS.md
uv run python -m evaluation.<corpus>_capture                    # GPU re-capture (one-time)
```

Corpus packages: `liger-kernel==0.8.0`, `fla-core==0.5.1`, `flag_attn @ git+FlagOpen/FlagAttention@41fc31d`, `flag_gems @ git+flagos-ai/FlagGems@1051e56c` (`--no-deps` + `sqlalchemy`). Detailed per-row tables: `evaluation/results/RESULTS.md`; raw rows with serialized witnesses: `evaluation/results/*.jsonl`.
