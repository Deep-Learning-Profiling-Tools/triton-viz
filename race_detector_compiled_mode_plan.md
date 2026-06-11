# Compiled-Mode Race Detector: Implementation Plan

**Target**: shared-memory (and later tensor-memory) data races, detected statically from
TritonGPU IR (TTGIR) via an SMT encoding — the "compile mode" counterpart to the existing
interpreter-driven dynamic mode (global memory).

Every load-bearing claim below was verified empirically on this machine
(triton 3.6.0 wheel, z3-solver 4.15.3, host-only compilation with
`GPUTarget("cuda", 80/90, 32)`); probe scripts and golden TTGIR dumps live in `/tmp`
(`dump_ttgir.py`, `probe_ir_bindings.py`, `ll_probe*.py`, `ttgir_pipeline.py`,
`matmul_s{1,3}_sm{80,90}.ttgir`).

---

## 1. Scope

### v1 goals
- **Memory space**: shared memory (`#ttg.shared_memory` memdescs).
- **Concurrency granularity**: intra-CTA — thread vs. thread, and thread vs. the
  *async proxy* (the cp.async engine; later TMA / WGMMA agents).
- **Bug surface**: the software-pipelining machinery — exactly where real compiler and
  hand-written-pipeline bugs live:
  - missing / miscounted `ttg.async_wait {num=N}`,
  - multibuffer rotation errors (stage dim too small, wrong index init/wrap),
  - buffer reuse after `ttg.local_dealloc` (allocation-level aliasing: the
    `convert_layout` scratch demonstrably reuses dot buffers — `metadata.shared`
    confirms),
  - sm90: `ttng.warp_group_dot_wait {pendings=N}` miscounts, missing
    `ttng.fence_async_shared`.
- **Guarantee**: per kernel *specialization* (constexprs and divisibility fixed — that is
  what compilation means), but **for all inputs, all grids, symbolic loop trip counts**.
  UNSAT = a proof; SAT = a witness (thread pair / iteration distance / stage / byte)
  mapped back to a source line via MLIR locs.

### Non-goals for v1 (explicit, each with a reason)
- **Generic-proxy ordering** (`local_store` vs `local_load` without async involvement):
  at TTGIR these are *deliberately unordered*; the backend **Membar pass** inserts
  `bar.sync` during lowering (verified: zero barrier ops in any TTGIR dump; 5 `bar.sync`
  in the PTX). Reporting them at TTGIR would be pure false positives. v1 treats
  generic↔generic pairs as Membar-ordered and only checks pairs with at least one
  async-proxy access — the contract Membar does *not* enforce. (v2 may re-implement
  Membar's aliasing analysis to also verify that contract.)
- TMA / mbarrier / `ttg.warp_specialize` / tensor memory (Blackwell `tcgen05`): not in
  the v1 op vocabulary; M4.
- Multi-CTA CGA layouts (`CTAsPerCGA > 1`), non-power-of-two shapes: assert-unsupported
  (mirrors the dynamic mode's unsupported-not-race policy).
- Global-memory static checking: a later bonus (§8), not v1.

---

## 2. Architecture

```
JITFunction.warmup (REAL compile, runs BEFORE interpreter patching)
        │  post_warmup_callback(jit_fn, CompiledKernel)        [hook exists today]
        ▼
CompiledKernel.asm["ttgir"]  ──write temp file──►  ir.parse_mlir_module + module.walk
        │                                            + one regex pass over the same text
        ▼
                      EventGraph (per specialization, cached)
   events: {kind, memdesc, stage-index expr, layout attrs, elem bits, R/W, proxy, loc}
   sync:   {token def-use chains, commit-group counters, wait counts, fences}
   loops:  {scf.for bounds, induction var, loop-carried index recurrences}
        ▼
                      SMT encoder (z3py, QF_BV)
   addr(tid, reg, k, stage) from layout closed forms      [§4]
   HB from token/counting semantics + rotation closed form [§5]
   two-copy query over (agent_a ≠ agent_b, k_a, k_b)       [§6]
        ▼
   SAT  → RaceReport(loc_a, loc_b, witness) → launch.records / last_reports
   UNSAT→ last_status="ok" (proof)   |   unmodeled op → last_status="unsupported"
   (side artifact: SMT-LIB2 dump per query — the paper's "SMT-IR")
```

New code lives entirely inside a new client package (narrow-hooks rule: no triton-specific
types leak into `core/`):

```
triton_viz/clients/race_detector/compiled/
    client.py        # CompiledRaceDetector(Client): warmup hook, spec cache, finalize
    ttgir_reader.py  # binding walk + text layer → EventGraph
    layouts.py       # blocked/swizzled/nvmma/padded closed forms → BV terms
    hb.py            # tokens, counting waits, rotation recurrences → HB relation
    smt_encoder.py   # two-copy QF_BV query builder + SMT-LIB2 export
    reports.py       # model → RaceReport (reuses race_detector/data.py types)
tests/unit/test_compiled_layouts.py
tests/unit/test_compiled_hb.py
tests/end_to_end/test_compiled_race_detector.py
tests/golden/ttgir/*.ttgir          # checked-in dumps + mutants
```

### Integration points (all verified to exist)
- **IR acquisition**: `pre_warmup_callback → True` forces the real warmup;
  `post_warmup_callback(jit_fn, ret)` receives the `CompiledKernel`
  (`ret.asm` keys: `source/ttir/ttgir/llir/ptx/cubin`). The profiler already uses this
  exact path (`clients/profiler/profiler.py:104-119`). Warmup runs *before* `patch_run`,
  so compilation sees unpatched `tl` ops.
- **Crucial correctness rule**: analyze the TTGIR produced by the *runtime's own
  specialization* (warmup path), never a hand-built `ASTSource` — without the runtime's
  `tt.divisibility=16` attrs and compile-time-1 inner strides the pipeliner silently
  disables and `num_stages=1` vs `3` yield byte-identical IR (verified). Driverless CI
  fallback: `triton.compile(ASTSource(...), target=GPUTarget(...))` with attrs supplied
  explicitly.
- **Specialization cache**: key on the kernel hash triton itself computes
  (`compute_cache_key`, jit.py:563-584) so each TTGIR is analyzed once; pattern precedent
  in `sanitizer.py:137-151`.
- **Reports**: return `list[RaceReport]` from `finalize()`, maintain
  `last_reports/last_status/unsupported_reason` — tests, `launches[-1].records`,
  visualizer and `trace_io` all work unchanged.
- **Mode composition**: the public `RaceDetector(...)` factory selects the backend via a
  `compile` keyword — `RaceDetector(compile=True)` dispatches `__new__` to
  `CompiledRaceDetector` (flag on); the default / `compile=False` stays the dynamic
  `SymbolicRaceDetector`; flag off is `NullRaceDetector` either way. Caveat (verified):
  `ClientManager.pre_run_callback` combines with `all()` — a compiled-mode client must
  return `True` from `pre_run_callback` (let other clients decide) and simply do nothing
  per block; it must NOT return `False` or it suppresses co-registered clients' blocks.
  Standalone compiled mode still pays one interpreted pass in v1 (acceptable); a
  `skip_interpretation` flag in `TritonTrace.run` is an optional later optimization.
- **NKI**: `NKITrace` never calls `patch_warmup` → compiled mode is Triton-only by
  construction, no frontend changes needed.

---

## 3. IR reading: hybrid binding walk + text layer

Verified capabilities of `triton._C.libtriton.ir`:
- `parse_mlir_module(path, ctx)` + `module.walk(cb)`: op names, operand/result SSA graph
  via `value.id()`, full type strings (including layout attrs), region nesting, module
  attrs (`ttg.num-warps`, `ttg.threads-per-warp`), locs on **result values**
  (`loc("acc"("file.py":29:23))`).
- `GluonOpBuilder.get_gluon_layout_from_tensor/memdesc(value)` returns fully-populated
  layout dataclasses (`BlockedLayout`, `SwizzledSharedLayout(vec=8, per_phase=2,
  max_phase=4, order=[0,1])`, …) straight from walked values — **no attr-text parsing
  needed for layouts**.

Known gaps and their workaround (the *text layer*):
- generic attributes are opaque (`arith.constant` values, `tt.make_range` bounds,
  `async_wait {num}`, `warp_group_dot_wait {pendings}`, atomic kind/sem/scope), and
  zero-result ops (`tt.store`, `scf.yield`) expose no loc. One regex pass over the SAME
  `asm["ttgir"]` string recovers these; lines correlate to walked ops by per-block
  program order (walk order verified deterministic on 3.6.0). Golden-file tests pin the
  printer format per triton version.
- `parse_mlir_module` takes a file path → write `asm["ttgir"]` to a temp file; keep the
  `ir.context` referenced (GC segfault hazard, verified pattern from triton's own code).
- **Never** call `to_linear_layout` on shared encodings: SIGABRT in the 3.6.0 wheel
  (binding wraps results in `LinearEncodingAttr`, which asserts on the `offset` in-dim).

The v1 op vocabulary (complete catalogue from the dumps):

| Role | Ops |
|---|---|
| smem write (async proxy) | `ttg.async_copy_global_to_local` |
| smem write (generic) | `ttg.local_alloc` (with operand), `ttg.local_store` |
| smem read (generic) | `ttg.local_load` (carries optional token operand) |
| smem read (async, sm90) | `ttng.warp_group_dot` (consumes memdescs directly) |
| buffer selection | `ttg.memdesc_index` (3.6 uses this, **not** `memdesc_subview`) |
| lifetime | `ttg.local_alloc` (mutable, no operand), `ttg.local_dealloc` |
| sync | `ttg.async_commit_group`, `ttg.async_wait {num}`, `ttng.warp_group_dot_wait {pendings}`, `ttng.fence_async_shared` |
| structure | `scf.for` (iter_args carry tokens + rotation indices), `arith.*`, `tt.*` pointer math |

Anything outside the vocabulary that touches a memdesc → `last_status="unsupported"`
with the op name (never silently wrong — same policy as dynamic mode).

---

## 4. Address function: layouts → QF_BV

`addr(tid, reg, k, stage) : BV` per event, built from layout attributes alone (all
formulas below were verified exhaustively against the C++ `LinearLayout` ground truth):

- **Distributed (blocked etc.)** — which element does thread `tid`'s register `reg`
  touch: decompose `lane = tid[4:0]`, `warp = tid >> 5`; per dim
  `coord[d] = rep[d]·tile[d] + warpIdx[d]·tpw[d]·spt[d] + laneIdx[d]·spt[d] + regInTile[d]`
  with `tile[d] = spt·tpw·wpc`, repetition bits order-fastest. For mma/dot_op layouts,
  don't re-derive by hand: read bases from python `to_linear_layout` (works for all
  distributed layouts) and encode the generic XOR-linear form
  `out = ⊕_i ite(x[i], b_i, 0)` — one `bvxor`+`ite` per input bit (≤ ~20 bits).
- **`#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>`**:
  `phase = (row / perPhase) % maxPhase`;
  `elemOff = row·numCols + (((col/vec) XOR phase)·vec) % numCols + col % vec`;
  `byteOff = elemOff·elemBytes + allocBase`. Pure shift/extract/bvxor/concat.
- **`#ttg.nvmma_shared<{swizzlingByteWidth W, elementBitWidth E}>`**: same XOR scheme on
  an 8×(8·max(16,W)/E) core tile with `vec = 128/E`, `perPhase = 128/W`,
  `maxPhase = W/16` (W=0 → no swizzle), tiled to the full shape.
- **`#ttg.padded_shared`**: XOR-linear part (bases literal in the attr) plus
  `Σ_i (off >> log2(interval_i)) << log2(padding_i)` — shift/add, still QF_BV.
- **Multibuffering**: stage is a leading dim on the memdesc
  (`!ttg.memdesc<2x64x32xf16, …, mutable>`); `ttg.memdesc_index %buf[%idx]` adds
  `idx · stageBytes`. Measured: sm80 depth = `num_stages−1`, sm90 depth = `num_stages`.
- **Allocation aliasing**: `local_dealloc` + later `local_alloc`/scratch may reuse bytes
  (verified via `metadata.shared`). v1 models each allocation's base as a symbolic BV
  constant with non-overlap constraints *only between live ranges that overlap in program
  order*; dealloc'd-then-reused pairs keep overlapping bases possible, so misordered
  access to a recycled buffer is reportable rather than assumed-disjoint.

**Broadcast caveat (FP guard)**: layouts with zero bases make several threads own the
*same* element (verified). Same-address writes whose addr functions are literally
identical modulo `tid` and write the same value-source SSA node are whitelisted as
intentional broadcast, not WAW.

Differential testing: the python `LinearLayout` API (`from_bases/apply`) is the oracle —
unit tests enumerate every (tid, reg) for small shapes and compare against the closed
forms, for randomized pow2 configs plus the five real configs from the dumps.

---

## 5. Happens-before at TTGIR (no barriers!)

TTGIR has **no CTA barrier ops** — ordering is carried by:

1. **Program order** within one agent.
2. **Token chains**: `async_copy → token`, `async_commit_group(tokens) → group token`,
   `local_load(…, token)` / `async_wait(tokens) {num=N}`. SSA def-use gives the edges
   directly from the binding walk.
3. **Counting semantics**: `async_wait {num=N}` at a point P orders *all but the last N
   committed groups* before P. With `g` groups committed per iteration (g=2 in the
   matmul: one per input), a copy issued at iteration `j` is HB-before a wait at
   iteration `k` iff `g·(k−j) > N` (after accounting for prologue peels). Same shape for
   `warp_group_dot_wait {pendings=N}` on the WGMMA agent.
4. **Rotation recurrences**: insert/extract indices are loop-carried
   `addi/cmpi/select` chains — *not* affine, but they are pure functions of the
   iteration: `idx(k) = (k + c) mod N`. The reader derives the closed form by abstract
   interpretation of the select chain; soundness is then **checked, not trusted**, with
   a per-kernel induction lemma in Z3 (`idx₀ = c ∧ (idxₖ = f(idxₖ₋₁) ⇒ idxₖ₊₁ = f(idxₖ))`
   against the closed form). If the chain doesn't match a rotation pattern →
   unsupported.
5. **Fences**: `ttng.fence_async_shared` orders generic-proxy smem writes before
   async-proxy reads at that program point (sm90 s1 pattern).

**Decidability without unrolling — the pipeline window**: addresses depend on `k` only
through `k mod N`, and counting-HB depends only on the distance `d = k_b − k_a`. For
`d > ⌈(N_wait + g)/g⌉` every pair is HB-ordered by the wait counting; so the query
quantifies over symbolic `k_a` (bounded by the symbolic trip count) and a *finite* set
of distances `d ∈ [0, depth + 1]`. No loop unrolling, trip count stays symbolic.

---

## 6. The query: two-copy over agents

Direct transplant of the dynamic mode's solver skeleton (alpha-renaming two agents, HB
closure from `hb_common.build_transitive_hb`, conflict predicate including the
scope/width-aware mutual-atomicity rule we just landed):

- Agents: `(tid_a, proxy_a)` vs `(tid_b, proxy_b)` with
  `(tid_a, proxy_a) ≠ (tid_b, proxy_b)`; `tid < num_warps·32` from module attrs; the
  async-copy "writer" is the async proxy (its `tid` is the issuing thread but the write
  lands asynchronously — modeled as a distinct agent whose HB edges are exactly the
  token/counting edges).
- Per cross-agent event pair: `SAT?(active_a ∧ active_b ∧ byte_overlap ∧
  conflicting ∧ ¬HB(a,b) ∧ ¬HB(b,a))` — same shape as
  `TwoCopySymbolicHBSolver._race_expr`, with QF_BV addresses instead of integers.
- Witness extraction: model gives `tid`s, `d`, stage, byte offset; locs come from result
  values (access ops all produce results; `tt.store` is global-side and out of scope) →
  `RaceReport` with both source lines, same dataclass as dynamic mode.

What changes vs. the dynamic solver: capture layer (IR reader instead of interpreter
overriders), HB edge generators (tokens/counting instead of CAS rf + acq/rel), address
sort (BV instead of Int). What is reused verbatim: two-copy alpha-renaming discipline,
HB transitive closure, conflict predicate, report plumbing, unsupported-not-race policy.

---

## 7. Milestones

**M0 — skeleton + IR capture (≈1 week)**
Client with warmup hook, spec cache, golden TTGIR check-ins (matmul s1/s3 × sm80/sm90 +
elementwise), `ttgir_reader` producing the EventGraph with locs.
*Exit*: structured dump of the matmul EventGraph matches a hand-checked YAML; reader
marks an unknown-op kernel unsupported.

**M1 — layouts → BV (≈1 week)**
`layouts.py` closed forms + generic XOR-linear encoder; differential tests vs. python
`LinearLayout` (distributed) and vs. the transcribed bases construction (shared).
*Exit*: oracle parity, exhaustive on small shapes, on ≥20 random pow2 configs + the 5
real configs; broadcast whitelist behavior covered.

**M2 — HB + solver, sm80 cp.async (≈2 weeks)** ← the heart
Token/counting HB, rotation closed-form + induction lemma, window theorem, two-copy BV
query, RaceReport mapping, SMT-LIB2 export.
*Exit*: stock matmul s2/s3/s4 → UNSAT (proof). **Mutation suite** (hand-edited golden
TTGIR) each → SAT with the right witness: (a) `async_wait num` too large, (b) wait
deleted, (c) stage dim shrunk (`2x…` → `1x…`), (d) rotation init off-by-one,
(e) commit-group dropped. Plus a `tl.static_range` hand-pipelined kernel written at the
source level both correctly and buggy.

**M3 — productization (≈1 week)**
`cfg.race_detector_mode`, factory wiring, `both` mode composition (verified callback
rules), CLI wrapper, docs; perf budget: ≤ a few seconds per specialization (events are
few; queries are per-pair like dynamic mode).
*Exit*: e2e tests through `triton_viz.trace`; dynamic suite untouched.

**M4 — sm90/Hopper (≈2 weeks)**
`warp_group_dot_wait {pendings}` agent, `fence_async_shared`, nvmma layouts (formula
already verified); then TMA descriptors + mbarrier phase/arrive-count modeling +
`ttg.warp_specialize` (producer/consumer warps are natural two-copy agents). Needs fresh
dumps from descriptor-based kernels (recon confirmed block-ptr kernels get rewritten to
plain pointers — must use `tl.make_tensor_descriptor` style sources).

**M5 — paper artifacts**
SMT-IR story: per-query SMT-LIB2 emission with a small metadata header (event ids, locs)
as the interchange format; optional later MLIR `smt`-dialect emitter (requires a
from-source triton/MLIR build — none of the needed bits ship in the wheel, verified).
Evaluation sweep: triton tutorials × `num_stages ∈ {1..4}` × {sm80, sm90}: proofs,
solve times, mutation-detection matrix; case studies from historical pipeliner bugs.

---

## 8. Later extensions
- **Static global-memory mode**: same encoder over `tt.load/tt.store` with grid-symbolic
  pids when no indirect loads exist; falls back to dynamic mode on indirection — the
  clean hybrid story (torch eager/compile analogy).
- **Membar verification (v2)**: re-implement the Membar aliasing analysis as constraints
  and check generic-proxy pairs too — turns the v1 assumption into a checked theorem.
- **Gluon kernels**: Gluon IR uses the same ttg dialect with explicit layouts — the
  reader should work nearly unchanged; valuable because Gluon authors hand-write the
  pipelining that the compiler normally gets right.

## 9. Risks

| Risk | Mitigation |
|---|---|
| TTGIR printer drift across triton versions | golden-file tests pinned per version; binding walk (stable API) carries structure, text layer only carries literals |
| `to_linear_layout` SIGABRT on shared attrs | never call it on shared; closed forms + oracle tests (already verified) |
| analyzing different IR than what runs (divisibility/specialization) | acquire IR only via the runtime warmup path; assert `metadata.shared` consistency |
| rotation chains beyond `(k+c) mod N` | induction lemma must pass or → unsupported (never trust the pattern match) |
| broadcast layouts → same-address multi-writer FPs | whitelist identical-addr-function writes of the same SSA value |
| autotuner: TTGIR is whatever config ran last | analyze per config via `compute_cache_key` (options are part of the key) |
| mbarrier phase parity (M4) | start with structural arrive/wait matching; data-dependent phases → unsupported |
| driverless CI | direct `triton.compile(ASTSource, target=…)` fallback (verified working host-only) |
