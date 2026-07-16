# Sweep Report — Triton Race Detector Evaluation

**Date**: 2026-07-13 · **Detector**: `race-detector-z3-demo` @ `c848c2b` + torchao/tritonbench corpus patches (committed with this report) · **Env**: triton 3.6.0, torch 2.10.0+cu128, z3 4.15.3, numpy 2.4.2, Python 3.12 · **Capture GPU**: RTX 4090 (sm89); sweeps TTIR-host-compile at the device capability when present, sm80 fallback (fp8 kernels need ≥89) · **Seed**: 0 · sweeps run at `--jobs 8` (definitive paper runs to be re-done at `jobs=1`)

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
| torchao | 67 | torchao git-pin `bfbc842` (`USE_CPP=0`, pure-Python Triton) | real code, fp8-quant + atomics |
| tritonbench_meta | 41 | meta-pytorch/tritonbench git-pin `1edaf3e` (harness-driven capture) | real code, benchmark ops |
| tilebench | 56 | Deep-Learning-Profiling-Tools/Tilebench local checkout `224ec81` (harness-driven capture) | real code, cuTile-twin benchmark |
| tilebench_cutile | 61 | same checkout — the cuTile (cuda.tile) twins, captured CuTile IR | real code, FIRST non-Triton corpus |
| aiter_originals | 2 | ROCm/aiter#3091 pre-fix kernel, vendored | RQ4 known-race reproduction |

All real-code rows carry heuristic `race-free` labels (production code); the micro-benchmark carries ground-truth yes/no labels with planted witness lines. Captured launches rebuild deterministically: int/bool tensors ≤8192 elements are value-exact snapshots; non-contiguous (column-major / broadcast-expanded) args rebuild from recorded strides; `tl.dtype`/`torch.dtype` constexpr objects round-trip as tagged JSON; every results header pins package versions + upstream commits.

torchao coverage note: 44/44 capture cases succeeded (67 kernel specializations). Structurally out of reach on this rig, recorded in `torchao_capture.py`: fp8_sdpa_inference (torch-2.11 package init), nvfp4 + mxfp8-CUDA + mx dim0/dim1 (sm100 gates), distributed comms kernels, one dead-code kernel, and the fp8 path of the common matmul (upstream KeyError as installed).

tritonbench_meta coverage note: capture DRIVES the suite's own `BenchmarkOperator` harness (`--only <impl> --num-inputs 1 --input-id 0 --test-only --force`) rather than a case table, with `module_prefix="tritonbench."` keeping only the suite's own kernels (its liger/inductor/vendor backends are excluded — liger is already a corpus, inductor is codegen). 43 cases → 41 specializations. Removed with a verified structural reason (recorded in `tritonbench_meta_capture.py`): sm90/sm100-only tlx/gluon/autows/TMA-persistent attention + gemm families, stream-k's host-side TensorDescriptor args (M4 track, 13-min autotune), and impls needing uninstalled deps (xformers/cutlass-ck/fbgemm/mslk/generative_recommenders). This is meta-pytorch/tritonbench (Meta's benchmark suite), distinct from thunlp/TritonBench = the `tritonbench_g` corpus. Its ~102-of-repo own kernels are hand-written (not the rumored 2000+, which counts only inductor codegen).

tilebench coverage note: the group's own multi-backend tile-DSL benchmark; every operator ships structurally-equivalent Triton AND cuTile implementations, so this corpus doubles as the Triton-side baseline for the planned cuTile frontend (same-operator cross-DSL differential). First local-checkout corpus (no packaging metadata): `TILEBENCH_ROOT` on sys.path, checkout HEAD commit as the pin (capture refuses tracked-dirty trees; `build_captured_corpus(installed_version=)` reuses the shared drift guard). Harness-driven capture through the suite's `core.engine` with `case_indices=[0]` and `report_benchmark` stubbed out — the only launch recorded is the engine's plain-stream verification run; `autotune` stays False so every impl fires its raw @triton.jit kernel once with its `_DEFAULT_CONFIG`. 45/45 operators captured (56 specializations), zero failures/skips.

tilebench_cutile coverage note — the cuTile front-end: rows carry CuTile IR TEXT compiled at capture (`compile_tile(return_final_ir=True)`, pure-Python — rebuild needs neither cuda-tile nor a GPU), consumed by the new reader (`clients/common/cutile_ir_reader.py`) which emits the SAME AccessGraph/Term algebra as the TTIR reader — the encoder, two-copy solver, tier selector and §3c launch-scoped rung run UNCHANGED. Semantic mapping: tile-space `tile_load/store(view, index)` lowers to `index*tile_shape + arange` affine terms with the implicit OOB-clip materialized as ordinary mask terms; `pointer_offset + tile_atomic_rmw / load_pointer / store_pointer` are exactly the TTIR raw-pointer shapes; python floor-division lowers to `c_mod` + a boolean-xor sign-fix the reader models exactly ((a∧¬b)∨(¬a∧b)); integer xor (bitonic partner indexing) and while-form `loop`/`if` blocks abstain honestly. Capture drove all 45 operators (385 specializations, zero failures); the corpus keeps ≤2 specializations per (case, kernel) with the drop count in provenance (bitonic-network operators bake one ct.Constant per host-loop step). v1 has NO confirmation channel (cuda.tile ships no interpreter) — race SATs would terminate at races-unclassified; none did.

## 2. Ground-truth scorecard (tritonracebench, 56 rows)

**precision = recall = 1.0 · witness-matched 25/25 · ladder audit zero (ladder-unsound=0, replay-unsound=0) · mutation sensitivity: all applicable proofs flip under at least one mutant.**

Terminals: race-confirmed 12, races-unclassified 13, race@interp 7, race-unconfirmed 1, proved@T0 7, proved@T1 8, proved@T1+assumes-termination 4, proved@interp 4. Companion micro-suites: golden_smoke 7 (3 race-confirmed / 3 proofs / 1 abstain), rmw_sync 9, await_sync 9 (3 conditional proofs + 6 detected races).

Launch-scoped-tier invariance (re-sweep at the §3-tier code state): the distribution above is IDENTICAL before and after the tier landed, and **zero** ground-truth rows carry the grid-fragile attribute — all 13 races-unclassified rows are in-extent SAT (their pinned re-queries stay SAT), so no planted race was proof-inflated away. The empirical separation holds through the machinery: every genuine race's witness is realizable at the launch extent; every wrapper-coupled artifact's is not.

## 3. Real-code corpora (886 rows)

Counting discipline (§3c guardrail 2): decided-clean is split BY SCOPE.
"Any-grid" proofs (T0: any params + any grid along read axes; T1: this
launch's params, any grid) are the unconditional column. "Launch-scoped"
proofs hold for the analyzed launch — `proved@T1-launch` (the §3c rung:
any-grid SAT, launch-extent UNSAT) and `proved@interp` (always
per-launch). Grid-fragile is its OWN column: rows whose launch-scoped
proof coexists with out-of-extent any-grid evidence (the wrapper's grid
contract is load-bearing). It enters neither the race counts nor the
unconditional-clean count, and the genuine-finding count (§4) stays 3.

| Corpus | Rows | Any-grid clean (T0/T1) | Launch-scoped clean (T1-launch/interp) | Grid-fragile | Abstain | Races-unclassified¹ | race@interp | Other² |
|---|---|---|---|---|---|---|---|---|
| tutorials | 9 | 5 (3/2) | 1 (0/1) | 0 | 3 | 0 | 0 | 0 |
| liger | 23 | 17 (0/17) | 1 (0/1) | 0 | 4 | 0 | 0 | 1 |
| tritonbench_g | 202 | 99 (30/69) | 40 (23/17) | 23 | 57 | 0 | 3 | 3 |
| fla | 378 | 123 (15/108) | 21 (9/12) | 9 | 228 | 0 | 1 | 5 |
| flagattn | 28 | 0 | 11 (10/1) | 10 | 17 | 0 | 0 | 0 |
| flaggems | 82 | 33 (11/22) | 10 (1/9) | 1 | 36 | 0 | 2 | 1 |
| torchao | 67 | 14 (5/9) | 16 (7/9) | 7 | 36 | 1 | 0 | 0 |
| tritonbench_meta | 41 | 13 (5/8) | 8 (1/7) | 1 | 19 | 0 | 0 | 1 |
| tilebench | 56 | 36 (21/15) | 6 (1/5) | 1 | 11 | 0 | 0 | 3 |
| tilebench_cutile | 61 | 36 (17/19) | 2 (2/0) | 2 | 23 | 0 | 0 | 0 |
| **Total** | **947** | **376 (40%)** | **116 (54/62)** | **54** | 434 | 1 | 6 | 14 |

Decided-clean across both scopes: 492/947 = 52% (each scope stated
separately above; the two are not interchangeable claims).

### 3b. Cross-DSL differential (TileBench twins: same operator, two DSLs)

45 operators ship structurally-equivalent Triton AND cuTile
implementations; verdict classes AGREE on 30/45 — including identical
abstention kinds where both are data-dependent (destindex's duplicate-
destination scatter, histogramming's value-indexed atomic, matmul_int8's
nested loops). The 15 divergences all attribute cleanly:

- **cuTile ahead (4)**: `batched_matmul` (Triton TIMED OUT on swizzled
  pointer arithmetic; cuTile's structured tile indices prove @T1),
  `matmul_fp32_fp16_fp8` (Triton Z3-undecided; cuTile proves @T1),
  `rope` and `flash_decode` (the cuTile twins avoid the loop shapes the
  Triton twins abstain on). Structured tile addressing is genuinely
  EASIER for Z3 than flat-pointer arithmetic on the matmul family.
- **cuTile behind (10)**: 7 nested-loop abstentions (the cuTile twins
  are multi-pass loops where Triton twins are single-pass or rescued by
  proved@interp — a channel cuTile lacks entirely, no interpreter),
  plus cross_entropy (interp-rescued on the Triton side only) and
  linear_self_attention ×1 case + block_sparse (while-form `loop`
  constructs, v1 unmodeled).
- **scope split (1)**: `top_k_selection` — Triton proves @T1 (any-grid);
  the cuTile twin's per-step launches prove only @T1-launch with the
  grid-fragile attribute (witness pid (2,0,0) outside grid [2,1,1]) —
  the §3c rung working unchanged through the new front-end.

¹ was: static any-grid SAT with every checked witness OUTSIDE the launch
extent (52 rows across 7 corpora). The §3c launch-scoped tier resolved
51 of them to `proved@T1-launch` + grid-fragile (three prior borderline
timeout/abstain rows also joined; net 52 launch-scoped static proofs).
The 1 remaining row (torchao common split-k matmul) is any-grid SAT
with a launch-scoped query Z3 cannot decide even at 120s (nonlinear
split-k scheduler arithmetic) — the terminal now precisely means
"any-grid SAT + launch-scoped undecidable". (An IN-extent SAT with a
genuine cross-block conflict is `race-confirmed`, not this bucket — see
the aiter_originals row and §6.8.)
² compile-error / timeout / crash.

Ladder audits: **PASS on every corpus** (ladder-unsound = replay-unsound = 0 everywhere).

## 4. Genuine races found: 3 (all triaged, all fixed upstream)

| # | Row | Mechanism | Scope | Upstream fix |
|---|---|---|---|---|
| 1 | `tb_nested_loops_processing` | kernel never reads `program_id`; grid=(2,) → both programs write identical `out_ptr` tiles (44 WAW witnesses, value-benign) | global, inter-CTA | [TritonBench#10](https://github.com/thunlp/TritonBench/pull/10) |
| 2 | `tb_quantize_kv_copy` | scatter through `Dest_loc` with real duplicate destinations (snapshot-faithful; witness pids match duplicate positions) | global, inter-CTA, data-dependent | [TritonBench#11](https://github.com/thunlp/TritonBench/pull/11) |
| 3 | `fla_based_fused_chunk` fwd | `z` store address omits the `i_v` grid axis → NV programs write identical values unsynchronized; bwd twin guards with `if i_v == 0`, fwd omits it | global, inter-CTA, same-value WAW | [fla#1018](https://github.com/fla-org/flash-linear-attention/pull/1018) |

All three: machine-generated witnesses first (detector-found), seed-independent, triage only adjudicated the heuristic labels. FlagAttention, FlagGems, torchao, tritonbench_meta, and tilebench: zero genuine races on every decidable row — notably the atomic-heavy FlagGems families (bincount/histc/scatter_reduce/index_reduce with duplicate indices) all PROVE clean, `vdot`'s atomic accumulate at T0; torchao's float8nocompile scale/cast kernels prove at T0, 7 of its 8 any-grid SAT rows land proved@T1-launch (+grid-fragile) with the split-k matmul the sole launch-undecidable holdout; tritonbench_meta's gdpa atomics and layer_norm/softmax/rms_norm backward lock-reductions all decide clean, its flash-TMA SAT row now a launch-scoped proof as well.

Separately, the **`aiter_originals`** RQ4 corpus (ROCm/aiter#3091, the MoE-routing `_sum_bitmatrix_rows_fused` at its pre-fix state) is `race-confirmed`: every program writes the full histogram with no pid partitioning — an in-extent cross-block WAW the detector reports and the interpreter reproduces. This is a real, previously-reported race (issue closed COMPLETED with upstream barrier fix), the paper's "detector flags the bug at the pre-discovery code state" data point. Its confirmation was restored this round (§6.8) and is unchanged under the §3c tier (in-extent SAT keeps the race path; the pinned re-query only sharpens its witnesses).

Counting discipline: the 52 grid-fragile rows are NOT findings — they are launch-scoped proofs whose safety depends on the wrapper's grid contract, reported as an attribute. The genuine-race count stays 3 (+ the aiter reproduction).

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
| wrapper-coupled any-grid class ×52 (7 corpora) | **resolved → proved@T1-launch + grid-fragile** (§3c tier) | launch-extent UNSAT on every one (e.g. tilebench `_kv_kernel`: witness pid (0,32,0) outside grid [32,32], axis-1 overflow wrapping into the next row); the any-grid evidence is carried as the grid-fragile attribute, not a race report |
| torchao common split-k matmul | races-unclassified (the 1 §3c holdout) | any-grid SAT; the launch-pinned query is Z3-undecidable even at 120s (nonlinear split-k scheduler arithmetic) — fail-closed, no launch-scoped claim made |

## 6. Detector defects surfaced by this evaluation round

1. **Reduce single-lane fold** (fabricates races in address position) — **FIXED**: reduce family gated in `_VALUE_DEPENDENT_ADDRESS_OPS`, pinned by test; affected row now abstains deterministically.
2. **`and`-truthiness interpreter divergence** — 3 instances across 2 corpora (fabricated WAW ×2, replay SIGSEGV ×1). Queued: pre-trace AST scan for BoolOp over tensors → mark interp-divergence-suspect, refuse replay (TODO §3f).
3. **Two-copy lane-model coupling** — same-axis arange vars must be equal per copy (TODO §3h; interim fail-closed gate proposed).
4. Philox/math-patch interp gap (`Patching math ops not yet supported`, flagattn dropout bwd) — small, queued.
5. **fp8 element width missing in the shared TTIR reader** (`_DTYPE_BITS` had bare `f8` but not MLIR's `f8E4M3FN`-family spellings) — **FIXED** this round; 15 torchao rows were pseudo-abstaining with `elem_bits=0`, 11 of them now decide (proved@T0/T1) or classify.
6. **TTIR host-compile target hardcoded to sm80** (`evaluation/harness.py`) — every fp8-arg kernel false-failed with `fp8e4nv not supported in this architecture`; **FIXED**: target the real device capability, sm80 fallback.
7. **Scalar-pointer atomic_rmw shape gap** — `tl.atomic_max/min` on a single-element global scalar (the fp8 global-amax idiom) abstains with `atomic_rmw of a non-pointer value`; 2 torchao rows (f8nc `_amax_atomic`, moe `_..._transpose_scales_rhs`). Queued reader extension.
8. **Confirmation gate over-declined exact races at unrolled same-line stores** — **FIXED** this round. The C2 ambiguous-site gate (which stops a dropped-mask WIDENED report from riding an unrelated same-line access's overlap) also skipped EXACT reports whose store is unrolled by `tl.static_range` onto one source line (`count>1` ⇒ ambiguous). The aiter#3091 kernel is exactly that shape, so its genuine in-extent WAW landed on `races-unclassified` instead of `race-confirmed`. Fix: gate WIDENED reports only — an exact report is a definite SAT witness whose access is live by construction, so the same-line bucket is its own real footprint. Pinned by `test_c2_confirms_exact_waw_at_unrolled_ambiguous_site`; ground-truth scorecard and all out-of-extent §3-¹ artifacts unchanged.
9. **Interpreter-track `tl.cumsum` overrider signature mismatch** — **FIXED**. The tl-module patch intercepts BEFORE triton binds `tl.cumsum`'s own defaults, so a bare `tl.cumsum(x)` (tilebench radix_sort) reached `_op_cumsum_overrider` as one positional arg while the overrider required `axis` — aborting the dynamic track with a TypeError. Fix: the overrider now mirrors `tl.cumsum(input, axis=0, reverse=False, dtype=None)`; every other tl-level patched op already mirrored its defaults. Pinned by `test_cumsum_overrider_defaults_axis_like_tl_cumsum`; the radix_sort row's dynamic track now lands on a clean `unsupported` (cumsum has no Z3 lowering) instead of a crash.

## 7. Abstention taxonomy → queued lifts

| Class | Rows (attributed) | Lift |
|---|---|---|
| indirect-address (loaded values in addresses; varlen `cu_seqlens`/`chunk_indices`, `block_tables`) | fla 147 + flaggems 12 + torchao 6 + tilebench 3 + tilebench_cutile 8 (incl. integer-xor bitonic partner indexing) + TB + liger | §3d snapshot-select extension to the COMPILED track |
| pid-affine loop bounds (`(pid+1)*BLOCK`-style, flash-attention causal loops) | flagattn 14 + flaggems 12 | §3g lift — bounds affine in pid enter the iteration-existence premise |
| runtime-scalar loop bounds (bound is a non-constexpr scalar arg; T1 wants launch-concrete) | torchao 8 + tilebench 1 | launch-scoped scalar binding, rides the §3c tier |
| wrapper-coupled any-grid | **LANDED**: §3c launch-scoped tier — 51/52 rows → proved@T1-launch + grid-fragile; 1 holdout (split-k, launch query Z3-undecidable) stays races-unclassified | done 2026-07-15 |
| nested loops | fla 20 + flaggems 6 + torchao 4 + TB 4 + tilebench 1 + tilebench_cutile 9 | §3e reader support (interp already rescues some); the cuTile 9 include multi-pass loops the single-loop slot rejects |
| data-dependent loop bounds (paged attention `context_lens`, jagged group offsets) | fla 19 + flagattn 1 + flaggems 1 + torchao 3 + tilebench 2 | §3e snapshot-lifted loop bounds |
| unstructured control flow (`cf.cond_br`; cuTile while-form `loop`/`if` blocks) | flagattn 2 + flaggems 3 + TB 2 + tilebench 1 + tilebench_cutile 6 | §3e path-condition encoding; cuTile if/while block support |
| carried-value `scf.while` (spin: `mm_streamk`, tilebench streamk `first_wave`; plain iteration: torchao mx swizzles) | flaggems 1 + torchao 2 + tilebench 1 | S6 await-abstraction extension; the torchao pair shows the gate also catches NON-spin carried whiles |
| non-contiguous tensor args (in-bounds premise needs dense layout; column-major quant outputs) | torchao 11 | strided-layout in-bounds premise (new; unlocked by the strides-capture extension) |
| scalar-pointer atomic_rmw (fp8 global-amax idiom) | torchao 2 | reader shape extension (§6.7) |
| runtime-codegen kernels (FlagGems pointwise_dynamic) | 3 filtered at capture | source-embedding capture scheme (backlog) |

## 8. Reproduction

```
uv run python -m evaluation.runner --corpus <name> [--jobs 8]   # per-corpus sweep
uv run python -m evaluation.report                              # regenerate RESULTS.md
uv run python -m evaluation.<corpus>_capture                    # GPU re-capture (one-time)
```

Corpus packages: `liger-kernel==0.8.0`, `fla-core==0.5.1`, `flag_attn @ git+FlagOpen/FlagAttention@41fc31d`, `flag_gems @ git+flagos-ai/FlagGems@1051e56c` (`--no-deps` + `sqlalchemy`), `torchao @ git+pytorch/ao@bfbc842` (`USE_CPP=0` + `--no-build-isolation`), `tritonbench @ git+meta-pytorch/tritonbench@1edaf3e` (+ `pynvml`, `transformers`). Detailed per-row tables: `evaluation/results/RESULTS.md`; raw rows with serialized witnesses: `evaluation/results/*.jsonl`.
