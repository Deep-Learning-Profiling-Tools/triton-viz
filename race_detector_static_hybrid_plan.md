# Race Detector: Static & Hybrid Modes — Implementation Plan

**Scope of this document**: the static (compiled-mode) race-detection tracks and the
hybrid dispatch layer that unifies them with the existing dynamic (interpreter-driven)
mode.

- **Part I** — the conceptual skeleton: one solver, one claim ladder, hybrid as a
  concretization policy. Read this first; Parts II/III are instances of it.
- **Part II** — Track 1: shared-memory races over TTGIR. **Shipped** (PR #476 and
  follow-ups); kept as reference, including the model boundary of the shipped v1.
- **Part III** — Track 2: global-memory races over TTIR + the tier selector.
  **Planned** (~5 weeks).

This file supersedes `race_detector_compiled_mode_plan.md` (renamed): Track 1 content is
carried over near-verbatim; its "later extensions" bullet on global memory graduates to
Track 2 here.

---

# Part I — Conceptual skeleton: one solver, one claim ladder

## I.1 Hybrid is a concretization policy, not a fallback arrow

The system has **one solver** (`TwoCopySymbolicHBSolver`) and multiple **capture
front-ends** (interpreter-driven, IR-driven). "Hybrid" is not a static box and a dynamic
box joined by a failure→fallback arrow; it is a per-kernel — and per-term — policy:

> **Choose the least concretization that makes the query decidable, and report the
> strongest claim that survives.**

Proof rungs (UNSAT side, ∀-claims):

- **T0 — everything symbolic** (scalar params, grid, pid, loop iterations, trip
  counts): *"no race for any input, any grid"* (per kernel specialization). Track 1
  already delivers T0 for its domain — its shipped guarantee is exactly "all inputs,
  all grids, symbolic trip counts". For global memory, T0 is opportunistic (see the
  nonlinearity gate, §I.3).
- **T1 — params concrete, threads symbolic** (scalar params taken from a real launch;
  pid, grid, loop iterations symbolic): *"no race for this input shape, for any grid
  and any pair of program instances."* Strictly stronger than what the dynamic mode
  alone claims today (concrete grid, executed path only).

A SAT result is **not** a rung: T0/T1 are universal claims, a SAT is an existential one
("this specific witness races"), and it gets its own **confirmation channel** (witness
replay, §I.4-C2). Earlier drafts numbered the replay "T2" — retired; it is not a proof
tier. Every query therefore terminates in exactly one of **five states**:

| terminal state | meaning |
|---|---|
| `proved@T0` | no race for any input, any grid (per specialization) |
| `proved@T1` | no race for this input, for any grid / pid pair |
| `race-confirmed` | SAT witness reproduced concretely by the interpreter |
| `race-unconfirmed` | SAT, but replay did not reproduce it — potential over-approximation FP, reported as *potential* |
| `unsupported` | outside every front-end's decidable region; reason recorded (unsupported-not-race policy, as always) |

These five states are the report vocabulary, the provenance labels, and the columns of
the evaluation tables (§III S5).

## I.2 Front-ends have reachable regions

The two capture front-ends differ in **what they are able to concretize**, and the
boundary is principled, not an implementation accident:

| | IR front-end (TTIR reader) | interpreter front-end (dynamic mode) |
|---|---|---|
| scalar params | symbolic **or** concrete | concrete (from the launch) |
| pid / grid | symbolic | pid symbolic (SymbolicExpr, alpha-renamed in the solver), grid concrete |
| control-flow paths | **both branches encoded** with path conditions | forced concrete — one executed path (why pid-dependent branches are its largest unsupported source) |
| memory contents (indirect indexing) | **unreachable** | concrete; loaded values modeled as Z3 arrays over concrete address tables (`race_detector.py:298-300`) |

The IR front-end can never concretize memory contents: doing so means executing load
semantics, which *is* the interpreter. Conversely the interpreter cannot avoid
concretizing paths. Hence at T1 **neither front-end dominates**: the IR front-end is
strictly stronger on path coverage, the interpreter strictly stronger on memory
dependence. That asymmetry — not "static failed" — is why both front-ends exist, and it
makes the dispatcher's job a one-liner: *within each front-end's reachable region, pick
the point with the least concretization whose query is decidable.*

The paper's core figure is the resulting 2-D map — axis 1: what is concretized
(nothing / scalar params / memory contents / paths); axis 2: what stays symbolic (pid,
grid, loop iteration, trip count). Every component is a point on the map; the dispatcher
is a policy that walks it; every benchmark kernel lands on a point (§III S5).

## I.3 The tier selector (per kernel, per term)

1. **Linearity gate for T0.** After encoding, a cheap syntactic scan of the term tree
   for symbolic×symbolic products (`pid × sym_stride`, `sym_param × sym_param`) decides
   whether a T0 attempt is worth a solver call at all; a short Z3 timeout is the
   backstop. Nonlinear → skip straight to T1, where params are concrete and every
   query is linear.
2. **DataDep placement rule** (per term, not per kernel):
   - loaded value in an **address** chain → a free address makes the query meaningless
     (nearly always SAT) → route the kernel to the interpreter front-end;
   - loaded value in a **mask** chain only → stay on the IR front-end and encode it as
     a **free variable**. The over-approximation is sound for the proof direction
     (UNSAT under an unconstrained mask is a real proof); a spurious SAT is caught by
     the confirmation channel. This converts a chunk of "indirect → unsupported" into
     `proved@T1` or `race-unconfirmed`.
3. **Any SAT → confirmation channel** (C2).

## I.4 The three information channels (the "strong hybrid")

- **C1 — concrete injection (dynamic → static).** The T1 rung itself: launch-captured
  scalar args populate the `LaunchContext` of the symbolic query while pid/grid stay
  symbolic — concolic *within a single SMT query*, not between two tools. Already free:
  args flow through the existing arg/grid callbacks.
- **C2 — witness replay (static → dynamic).** A SAT model (pid pair, loop iterations,
  params) is replayed under the interpreter: run with the witness grid dims and the
  captured args, executing only the two witness program ids (the designated-block
  capture slot, `race_detector.py:288-292`, pointed at them), then intersect the two
  concrete footprints. Role: **the soundness patch for over-approximated free
  variables** (the DataDep-in-mask rule) and a detector for encoding bugs — load-
  bearing, not a DART/CUTE homage. v1 replays T1 witnesses (params already real); T0
  witnesses would require materializing tensors of witness shapes — stretch.
- **C3 — differential cross-check (both directions).** Instantiate the static symbolic
  footprint at the dynamic launch's concrete params; it must match the dynamic records
  one-to-one. Each side is the other's oracle: divergence exposes either a compiler
  lowering the IR reader misread or an interpreter semantics deviation. Precondition:
  align the masked-lane convention (whether masked-off lanes appear in records) before
  comparing, or the diff is pure noise.

---

# Part II — Track 1 (shipped): shared-memory races over TTGIR

**Target**: shared-memory (and later tensor-memory) data races, detected statically from
TritonGPU IR (TTGIR) via an SMT encoding — the "compile mode" counterpart to the
interpreter-driven dynamic mode. In the ladder of Part I this track proves at **T0** for
its domain (intra-CTA shared memory, per specialization).

Every load-bearing claim below was verified empirically on this machine
(triton 3.6.0 wheel, z3-solver 4.15.3, host-only compilation with
`GPUTarget("cuda", 80/90, 32)`); probe scripts and golden TTGIR dumps live in `/tmp`
(`dump_ttgir.py`, `probe_ir_bindings.py`, `ll_probe*.py`, `ttgir_pipeline.py`,
`matmul_s{1,3}_sm{80,90}.ttgir`).

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
  UNSAT = a proof; SAT = a witness (iteration distance / stage / slot) mapped back to a
  source line via MLIR locs; the witness `byte_offset` is a representative byte derived
  from the layout closed forms *after* the solve, not part of the solved query.

> **v1 as shipped — model boundary (read before trusting an `ok`).** The implemented
> query is a **wait-coverage** check: for each (copy, load) on a slot, does the load's
> guarding `async_wait` cover the copy's commit group (counting + per-allocation operand
> gating)? UNSAT is a proof *of that property*, not a full byte-level data-race proof. It
> models the whole-tile cp.async shape (same slot ⇒ full byte overlap) and the RAW
> direction only, under the lockstep/Membar-barrier assumption. It does **not** yet
> implement: copy/load active masks, sub-tile byte-overlap or per-thread/register
> footprint in the solver (sound for the whole-tile shape, see `smt_encoder.py`), the
> sm90 `ttng.*` path (M4 — flagged unsupported), or **allocation aliasing**: a
> `local_alloc` after a `local_dealloc` (buffer reuse) is flagged **unsupported**, not
> proven — see the "Allocation aliasing" bullet below.

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
- Global-memory static checking: **Track 2 — Part III of this document**, not this
  track.

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
- **Allocation aliasing** (deferred — **not implemented in v1 as shipped**): `local_dealloc`
  + later `local_alloc`/scratch may reuse bytes (verified via `metadata.shared`). The
  intended model gives each allocation a symbolic BV base with non-overlap constraints
  only between live ranges that overlap in program order. v1 does **not** track allocation
  bases or live ranges and the encoder only forms same-allocation (copy, load) pairs, so to
  stay sound the reader flags any `local_alloc` that follows a `local_dealloc` as
  **unsupported** rather than silently assuming disjointness. A *terminal* dealloc (stock
  epilogue cleanup, no later alloc) is harmless and ignored. Cross-allocation aliasing is a
  v2 item alongside Membar verification.

**Broadcast caveat (FP guard)**: layouts with zero bases make several threads own the
*same* element (verified). Same-address writes whose addr functions are literally
identical modulo `tid` and write the same value-source SSA node are whitelisted as
intentional broadcast, not WAW.

Differential testing: the python `LinearLayout` API (`from_bases/apply`) is the oracle —
unit tests enumerate every (tid, reg) for small shapes and compare against the closed
forms, for randomized pow2 configs plus the five real configs from the dumps.

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
  **(v1 as shipped):** the implemented `_check_pair` solves a reduced form of this —
  same-slot equality stands in for `byte_overlap` (sound for whole-tile copy/load, where
  same slot ⇒ full overlap), and `active_a ∧ active_b` (masks) and the per-thread/register
  footprint are dropped because they do not change which commit group a wait covers. The
  full `active ∧ byte_overlap ∧ thread-footprint` formula is the v2 generalization to
  sub-tile / partially-masked shapes.
- Witness extraction: model gives `tid`s, `d`, stage, byte offset; locs come from result
  values (access ops all produce results; `tt.store` is global-side and out of scope) →
  `RaceReport` with both source lines, same dataclass as dynamic mode.

What changes vs. the dynamic solver: capture layer (IR reader instead of interpreter
overriders), HB edge generators (tokens/counting instead of CAS rf + acq/rel), address
sort (BV instead of Int). What is reused verbatim: two-copy alpha-renaming discipline,
HB transitive closure, conflict predicate, report plumbing, unsupported-not-race policy.

## 7. Milestones — status

**M0–M3 landed** (skeleton + IR capture, layouts → BV, HB + solver for sm80 cp.async,
productization — shipped via PR #476 and follow-up commits). Outstanding:

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

## 8. Later extensions (Track 1)
- **Membar verification (v2)**: re-implement the Membar aliasing analysis as constraints
  and check generic-proxy pairs too — turns the v1 assumption into a checked theorem.
- **Gluon kernels**: Gluon IR uses the same ttg dialect with explicit layouts — the
  reader should work nearly unchanged; valuable because Gluon authors hand-write the
  pipelining that the compiler normally gets right.

## 9. Risks (Track 1)

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

---

# Part III — Track 2 (planned): global-memory races over TTIR + tier selector

## III.0 Design decisions

- **D1 — IR layer: TTIR, not TTGIR.** Global addresses at TTIR are complete
  `tt.addptr`/`arith` chains with no layout attributes to interpret; the vocabulary is
  far smaller; and software pipelining does not change the *set* of global accesses, so
  nothing is gained by waiting for TTGIR. (Track 1 stays on TTGIR because shared-memory
  ops exist only there — two readers on two IR levels is deliberate; the provenance
  labels carry the track dimension so merged reports stay distinguishable.)
- **D2 — solver: reuse `TwoCopySymbolicHBSolver`, skeleton unchanged.** It already
  consumes records of Z3 address expressions + constraints, with pid alpha-renaming,
  mutual atomicity, intra-lane queries and report plumbing. "Same encoder" concretely
  means: one solver, two capture front-ends — one driven by the interpreter, one by the
  IR.
- **D3 — primary target is T1; T0 is opportunistic.** Any 2-D kernel has
  `pid × stride`; with symbolic strides that product is nonlinear and Z3 `unknown`
  becomes the norm, not the exception. With params concrete: strides are constants →
  every query is linear; loop bounds concretize through the existing `_loop_bounds`;
  the symbolic-trip-count work is deferred to the T0 stretch (S5). The T1 claim
  already strictly dominates the dynamic mode's per-launch claim, so the paper
  narrative stands without T0.

## III.1 Existing assets (verified in-tree — do not rebuild these)

- **`triton_viz/clients/common/ttir_reader.py`** (~760 lines; promoted from
  `clients/sanitizer/compiled/` in S1, old path kept as a re-export shim): `parse_ttir`
  → `AccessGraph` with per-access Term chains (offset, mask), source locs, `DataDep`
  markers for loaded values (the indirect-indexing signal, ready-made), and `guarded`
  flags for scf.if regions. The vocabulary covers `tt.load/store`,
  `tt.atomic_rmw/atomic_cas` (added in S1; float atomic max/min's sign-trick lowering
  stays fail-closed), `tt.addptr/splat/broadcast/expand_dims/make_range/get_program_id`,
  `arith.*`, `scf.for/if/yield`. Nested/multiple loops → `UnsupportedTTIR`
  (`ttir_reader.py:399`).
- **`triton_viz/clients/sanitizer/compiled/oob.py`**: `_eval` Term→Z3 evaluator;
  `LoopVar` is already a free variable over `[lower, upper)` — **no unrolling is the
  status quo**, not a work item; `_loop_bounds` concretizes bounds at launch (raises
  on non-constants — exactly the T1 behaviour).
- **TTIR acquisition + parse cache**: sanitizer `client.py:87-90` (`asm["ttir"]` from
  `post_warmup_callback`) and `client.py:149-158` (`_graph_cache` keyed on the TTIR
  text hash).
- **Solver channels, all present**: `copy_local_vars` (per-copy loop variables),
  `local_constraints`/`premises` (per-record constraints — path conditions ride here),
  arange substitution (`_make_arange_subs_and_constraints`), `_exact_atomic_addr` +
  the scope/width-aware mutual-atomicity rule. Grid concreteness lives in exactly one
  place: the `int(d)` cast in `_normalize_grid`
  (`two_copy_symbolic_hb_solver.py:404`), and the grid constraints are already written
  in the shape `0 ≤ pid_x[i] < grid[i]`.
- **Dynamic front-end facts the channels rely on**: records are captured from one
  designated block's symbolic execution and alpha-renamed in the solver
  (`race_detector.py:288-292`); loaded values are modeled per-launch as Z3 arrays over
  concrete address tables (`race_detector.py:298-300`).

Net effect: "write a TTIR reader" and "write a symbolic evaluator" collapse into
"promote, extend, generalize". The only component with no existing code is scf.if
condition modeling (S2).

## III.2 Steps

### S1 — reader promotion + atomic semantics (≈3–4 days)

- Promote `ttir_reader` to a shared module (shared code stays mechanism-only, each
  client owns its policy — the same narrow-hooks rule as Track 1); the sanitizer path
  keeps a re-export for compatibility. **Done** → `triton_viz/clients/common/ttir_reader.py`.
- The race-detector compiled client captures `asm["ttir"]` alongside its existing TTGIR
  capture; parse cache copied from the sanitizer pattern. **Done** —
  `_consume_pending_ttir` in `race_detector/compiled/client.py`; parse failures are
  recorded per kernel (`last_ttir_unsupported`) and never touch the TTGIR verdict.
- Atomics get race semantics. **Done, with two corrections to the original premise**:
  (a) `tt.atomic_*` was not "recorded as a plain access" before — it FAILED CLOSED in
  the reader; (b) rather than emitting two reader events, an atomic parses into a
  single `AccessEvent` with `kind="atomic_rmw"|"atomic_cas"`,
  `AtomicInfo(rmw_op, sem, scope)` and `is_read`/`is_write` both true — the expansion
  into separate read+write solver events belongs to the S3 record builder (two reader
  events would double-report sanitizer OOB). Float `tl.atomic_max/min` lower to a
  sign-trick dance (pointer `tt.bitcast`, masks derived from the loaded value) and
  correctly stay fail-closed — golden `atomic_fmax_*.ttir` pins this. The solver side
  (`_exact_atomic_addr`, mutual atomicity) is untouched, as planned.
- The single-loop limitation stays and is **written into the support matrix**, so S5's
  numbers aren't a surprise.

*Exit*: **met** — the race-detector client parses stock TTIR into `AccessGraph`s with
atomic RMW events (`tests/unit/test_compiled_race_detector_ttir.py`,
`tests/unit/test_ttir_reader_atomics.py`; goldens `atomic`/`atomic_fmax`/`cas`);
sanitizer OOB now also checks atomics (side benefit) and its suite stays green.

### S2 — scf.if condition modeling + per-term DataDep policy (≈1 week — the core new work)

- Capture the branch condition's Term chain; every access in the region carries a path
  condition (conjunction across nested ifs); **both branches are encoded**. **Done** —
  `AccessEvent.path` (new `Not` term for else-regions), `_IfFrame` walker state;
  `guarded` now means only "condition unmodelable (loaded data)" and keeps the pre-S2
  pessimism.
- scf.if results upgrade from `DataDep` to ite Terms when the condition is modelable.
  **Done, with a scope discovery**: triton canonicalizes pure-scalar if/else into
  `arith.select` before TTIR (already in the vocabulary), so the value-yielding scf.if
  that actually survives is the side-effect kind whose results are loaded data —
  correctly left as `DataDep`. The Select upgrade covers the remaining single-result
  case; yields are resolved at the yield line because then/else regions legally reuse
  SSA names. Multi-result ifs stay fail-closed.
- Per-term DataDep policy (feeds the selector, §I.3). **Done** —
  (a) mask chain: the mask is dropped (widened to free) and the access flagged
  `mask_dropped`; UNSAT stays a sound proof (dropping constraints only widens the
  footprint), SAT follows the same never-a-witness discipline as `guarded`
  (sanitizer: abstain with kind `data-dependent-mask`; race records at S3 consume the
  flag, C2 replay confirms SATs at S4). Refinement deferred: splitting a
  `modelable ∧ DataDep` mask to keep the modelable conjunct — precision, not
  soundness.
  (b) address chain: stays whole-kernel unsupported (a free address makes the query
  meaningless), now classified — `UnsupportedTTIR.kind` carries a stable taxonomy
  (`indirect-address` / `data-dependent-bound` / `nested-loop` / `out-of-vocabulary` /
  `control-flow` / `block-pointer` / `unmodelable-condition` / `data-dependent-mask` /
  `other`) that the tier selector routes on and S5 buckets; the race client's
  `last_ttir_unsupported` reasons are prefixed with it. Classification precision:
  only a `DataDep` whose provenance is MEMORY CONTENTS counts as
  `indirect-address`/`data-dependent-bound` (the interpreter-route family);
  modeling gaps (loop accumulators, unmodeled ops, unresolved SSA) stay `other`
  so the buckets don't overstate permanent indirection.
- Side benefit: **done in the same change** (not separately) — `check_access` adds the
  path constraint, so sanitizer accesses under modelable conditions are proved
  precisely and a SAT under the path is a real, reachable witness.

*Exit (headline acceptance)*: **met — S2 complete.** The `pid_branch` golden (a store
only block 0 executes, the dynamic mode's classic unsupported case) gets a
path-precise proof and, for the too-small-tensor variant, a witness pinned to pid 0
(`tests/unit/test_ttir_reader_scf_if.py`); data-dependent-mask kernels that
previously died at parse now prove or abstain
(`tests/unit/test_ttir_reader_datadep_policy.py`).

### S3 — T1 evaluation + solver hookup — **done**

- **Record builder** (`race_detector/compiled/global_records.py`): lowers the shared
  `AccessGraph` under one launch's concrete params into the exact record shape the
  dynamic mode produces — byte addresses on the launch's `data_ptr` bases, pids as
  the shared `SymbolicExpr.PID0/1/2` consts, one interned arange summary var per
  (make_range, dim) in an `ARANGE_DICT`-shaped registry, the scf.for iteration as
  ONE symbolic index in `copy_local_vars` with its range in `premises`,
  `mask ∧ path` in `active`. Launch capture lives in `pre_warmup_callback` (the only
  hook that sees real args on the warmup-only path). New client surface:
  `last_global_status` / `last_global_reason` / `last_global_reports`, independent
  of the TTGIR shared-memory verdict.
- **Solver**: `_normalize_grid` accepts Z3 dims (symbolic dims get `≥ 1` in the grid
  constraints); nothing else needed changing — the audit found no concrete-grid
  short-circuits in code (the vacuous-unsat "shortcut" was semantics, not code).
- **Two modeling discoveries** (both surfaced by the pipeline itself on stock
  goldens):
  1. *Unused grid axes are pinned to 1.* Under a fully symbolic grid every 1-D
     kernel "races" on a 2-D grid it never reads (two blocks differing only in an
     ignored axis compute identical addresses). The honest T1 claim is "race-free
     for every grid **along the axes the kernel reads**" — where "reads" is the
     PARSE-time set of `tt.get_program_id` axes (`AccessGraph.pid_axes`), never
     the axes that survive into modeled terms: the adversarial round produced
     three false-proof families from eval-time collection (pid in a stored
     VALUE, pid inside a dropped mask, pid inside an unmodeled condition — all
     distinguish block behavior without entering address math). Two more holes
     from that round, also fixed with their repros as regression tests: a
     zero-trip loop was modeled as one phantom iteration (spurious definite
     reports; in-loop accesses are now skipped when the launch's trip count is
     zero, and the iteration premise attaches only to in-loop records), and a
     non-contiguous tensor's `numel` understates its strided extent (the
     in-bounds premise would deactivate legal accesses — non-contiguous now
     fails closed, like the sanitizer).
  2. *The in-bounds premise.* Unbounded symbolic pids let offsets stray
     arithmetically into other tensors' address ranges, fabricating cross-tensor
     races no launch produces. Every record carries its allocation bounds
     (`base ≤ addr < base + numel·elem`); real aliasing still surfaces (the bounds
     are the launch's actual intervals). Composition: the compiled sanitizer's OOB
     verdict proves exactly the premise the race verdict assumes.
- **Atomics**: RMW = one record (`is_atomic`, `atomic_kind="rmw"`; the solver's own
  lowering makes it read∧write); mutual atomicity/scope/width reused verbatim —
  including atomics inside loops, which the dynamic mode marks unsupported. CAS →
  classified unsupported (`cas-synchronization`) → interpreter route, as planned.
- **Uncertainty discipline honored** (the S2 invariant): reports touching a
  `mask_dropped`/`guarded` record are never definite races; only-widened SATs make
  the launch `unsupported ("possible race under over-approximation")`.

*Exit*: **met** — `proved@T1` on stock add (1 symbolic grid axis), masked 2-D tile2d
(2 symbolic axes), and the matmul K-loop (full loop machinery); pid-stride mutation
→ definite WAW with a cross-block witness (`tests/unit/test_t1_global_races.py`,
15 cases). The system is usable end-to-end; evaluation can start.

### S4 — tier selector + the three channels (≈1 week)

- Selector per §I.3: linearity gate for T0, DataDep placement rule, every SAT → C2.
  **Selector done** (`_solve_one_graph` / `_try_t0` in the compiled client;
  `t0_linearity_gate` + `encode_graph_t0` in `global_records.py`):
  - T0 = scalar params as SHARED symbolic Ints (not copy-local — one launch, two
    blocks) behind the syntactic linearity gate (no symbolic×symbolic product, no
    symbolic divisor; iter-arg deltas count). Z3 timeout as backstop.
  - T0 has no launch, so the non-aliasing premise is realized by PARTITIONING: one
    solver run per base pointer, addresses are byte offsets from that base;
    read-only groups skipped. Loop bounds referencing a param fail to concretize →
    automatic T1 fallback.
  - **Any T0 SAT falls through to T1** — a T0 witness carries parameter values that
    need not match this launch (this also subsumes the widened-record discipline at
    T0: only UNSAT matters there).
  - **The T0 proof only stands in for a launch that PROVES the non-aliasing
    premise** (adversarial round): the partition assumes distinct args are
    distinct allocations, but an in-place launch (same tensor twice) really
    races — and a bare T0 accept short-circuited exactly the T1 run whose real
    bases would report it. The selector now requires captured, contiguous
    metadata for every accessed base and pairwise-disjoint
    `[data_ptr, data_ptr+numel·elem)` intervals before accepting T0; aliased or
    unverifiable captures fall to T1 (report or fail closed). The gate walk is
    also exception-guarded (deep-but-legal term chains must degrade to
    unsupported, not crash the launch teardown).
  - Provenance surfaced: `last_global_provenance` = `proved@T0` ("race-free for ANY
    scalar params — this specialization, non-aliased args — on every grid along
    the read axes"; the claim neither the dynamic mode nor T1 can make) or
    `proved@T1`. The stock add kernel (folded-constant stride) proves at T0;
    param-stride kernels (tile2d, matmul) gate to T1
    (`tests/unit/test_t1_global_races.py`, 26 cases).
- **C1** is already free (launch args → `LaunchContext`).
- **C2 — done** (`compiled/replay.py` + client wiring, `confirm_races=True` by
  default): SAT witnesses replay under the INTERPRETER via a lightweight
  `FootprintRecorder` client — a fresh nested trace that executes only the two
  witness pids on PRE-launch tensor clones (finalize runs after the real kernel
  already mutated the originals, so the snapshot happens at `pre_warmup`, capped
  at 256 MB). Proofs never engage the interpreter (the in-process
  interpreter-then-real-compile leak hazard documented in trace.py stays dormant
  on the clean path). Footprints intersect at element granularity with mutual
  atomicity honored; the aggregate lands in `last_global_confirmation`
  (`confirmed`/`unconfirmed`/`partial`). Two upgrades over the plan:
  a CONFIRMED widened (dropped-mask) report **graduates to a definite race** —
  the S2 abstention becomes a verdict on this launch's data — and an
  unreproduced widened SAT becomes the explicit `race-unconfirmed` terminal
  state. All five §I.1 terminal states are now materialized.
  **Second adversarial round — three fabrication holes fixed** (each with an
  end-to-end repro pinned in `test_replay_channels.py`):
  1. *Replay runs at the LAUNCH grid*, never a synthetic `max(pid)+1` grid:
     `tl.num_programs` is an unmodeled op (DataDep → widened, NOT unsupported),
     and a grid-observing dropped mask flips value under a different grid —
     fabricating a confirmed race for a launch that stores nothing. Also kills
     the replay-cost DoS (solver-chosen witness pids no longer size the grid;
     `REPLAY_MAX_BLOCKS` caps pathological launch grids).
  2. *Witness pids must exist on the launch grid* — a grid=(1,) launch cannot
     have its widened abstention graduated by a fabricated second block (the
     grid-generic T1 "races" claim for exact reports is unchanged; only the
     "on this launch's data" graduation demands launch-grid witnesses).
  3. *Focus buckets must be unambiguous*: footprints key on (tensor, kind), so
     two same-kind access SITES on one tensor share a bucket and an unrelated
     exact overlap would confirm a dead widened report. Reports in ambiguous
     buckets classify `unavailable` (per-site keying, e.g. by source line, is
     the noted refinement). rmw∩rmw pairs are `unavailable` too (scope/width
     live outside a footprint — the misleading "unconfirmed" label is gone),
     the `race-unconfirmed` claim is made only when EVERY widened report was
     actually replayed (cap honesty), and widened reports replay before exact
     ones (they are what the channel exists to classify).
- **C3 — done** (`compiled/differential.py` + `cross_check` in `replay.py`;
  opt-in `differential_check=True` → `last_differential`): the static side is a
  numpy-only CONCRETE enumerator of the AccessGraph (deliberately independent of
  the Z3 encoding — the diff compares two implementations that share only the
  kernel), the dynamic side is the same interpreter footprint capture. At
  element-start byte granularity masked-off lanes are naturally absent from
  BOTH sides, so no masked-lane convention alignment was needed after all.
  Over-approximated accesses are excluded (no exact static footprint) and
  reported as skipped — SYMMETRICALLY: the widened access's exact same-bucket
  siblings leave the diff scope on both sides (one-sided deletion fabricated a
  static-only divergence for the common exact+widened-same-tensor pattern;
  second adversarial round). Both sides speak the snapshot clones' addresses.
- Provenance on every report and status: terminal state (five states, §I.1) × track
  (global/TTIR vs shared/TTGIR).
- Mutation suite: wrong pid stride, dropped mask term, atomic → plain store — each must
  go SAT with the correct witness **and** come back `race-confirmed` through C2.

### S5 — evaluation + T0 stretch (≈1.5–2 weeks, overlapping from S3)

Protocol modeled on DataRaceBench (Liao et al., SC'17) and the LLOV evaluation
(three-outcome scoring for static tools), with three deliberate departures noted
below.

**Harness design decisions:**

- *Driverless throughout.* TTIR via host-only `triton.compile(ASTSource,
  GPUTarget("cuda", 80, 32))`; the client is driven synthetically
  (`pre_warmup_callback(jit_fn, *args, grid=…)` → `post_warmup_callback(asm)` →
  `finalize()`) with CPU tensors; C2 replay and the dynamic-mode comparison run on
  the CPU interpreter. CI-runnable and reproducible; the cost is one hand-written
  **LaunchSpec** per kernel (`kernel_fn, signature, constexprs, make_args(seed),
  grid, params`, plus the ground-truth fields below) — which is also where the
  reproducibility comes from. Autotuned kernels: take `.fn`, pin one config
  (methodology note).
- *One subprocess per kernel.* Hard wall-clock timeout (timeout is a recorded
  outcome, not an accident); fixed compile-before-interpret ordering sidesteps the
  interpreter-patching hazard documented in trace.py; a crash cannot take down the
  sweep.
- *The dynamic-mode comparison is a first-class column*, not an afterthought: each
  subprocess runs the static track (`CompiledRaceDetector(confirm_races=True,
  differential_check=True)`) AND the dynamic `RaceDetector()` on the same
  kernel+launch — the "rescued from dynamic-unsupported" headline is a per-row diff.

**Corpus (three phases):**

1. *Phase A — labeled micro pairs ("TritonRaceBench", a publishable artifact in its
   own right: no labeled Triton race corpus exists).* DRB-style yes/no PAIRS: each
   race pattern contributes a racy variant and a fixed variant, ground truth by
   construction, named with the label (e.g. `trb007_pid_branch_store_yes/_no`).
   Patterns: pid-stride misalignment, missing mask term, atomic→plain store, pid
   branch, data-dependent mask, loop-carried overlap, aliased in-place, CAS lock,
   gather, nested loop, … (~15 pairs; several distilled from existing tests).
   Input-parameterized kernels included (same kernel, n=0 race-free vs n=5 racy —
   the T1 claim made concrete), one row per parameter set.
2. *Phase B — triton tutorials* (vendored for triton 3.6, hand-written LaunchSpecs,
   ~10–12 kernels): the "standard corpus" column.
3. *Phase C — a real library* (liger-kernel or a TritonBench subset, 20+ kernels).
   Expect `unsupported` to dominate (nested loops); that IS the data — the kind
   distribution tells the paper where the next modeling investment pays.

**Scoring (per row: kernel × launch-params):**

- Terminal state (five states) + `provenance` + `confirmation` + unsupported
  `kind`/reason; tier-selector detail (`t0_gate`, T0 attempted/result — so the T0
  stretch shows up as a re-run delta); per-phase wall-clock; C3 result
  (agree/mismatch/skipped) as the harness's built-in correctness oracle — any
  mismatch is a red flag to investigate by hand.
- DRB/LLOV-style table: TP/FP/TN/FN + precision/recall + **coverage**, with
  abstentions as their OWN outcomes (SV-COMP-style), split into
  `race-unconfirmed` (reported but not certified) vs `unsupported` (never entered
  the pipeline). Mapping: proved@T0/T1 → "no"; race-confirmed → "yes".
- Per-pattern breakdown (which bug classes the detector is strong/weak on) and the
  static-vs-dynamic per-kernel matrix.
- *Cross-row ladder audit* (departure 4): per kernel, derive "∃ racy launch"
  from the yes-labels within each specialization and T0-premise scope; any
  proved@T0 against it → `ladder-unsound` (its own severity class, above FP);
  any race-confirmed on a no-labeled launch → `replay-unsound`. Both zero by
  construction or the release is blocked.
- *Mutation sensitivity mode*: for every kernel that PROVES, auto-mutate the TTIR
  (pid-stride constant) and assert the verdict flips to races — proofs are not
  vacuous; one broken constant is caught.

**Four deliberate departures from DRB (paper differentiators):**

1. *Witness-level scoring, not file-level binary.* DRB scores "reported a race
   y/n" per file (its acknowledged weakness — any report counts, even the wrong
   one). Our reports carry source-line pairs and witness pids; micro-pair ground
   truth labels the RACING ACCESS PAIR (`race_pair` in the LaunchSpec), and
   scoring distinguishes "found the planted race" from "found some race".
2. *Deterministic protocol.* DRB needs N runs × M configs for nondeterministic
   dynamic tools; our static side is deterministic and the replay is
   deterministic given the seed — one run per row is the protocol, stated as
   such.
3. *Proof-strength dimension.* DRB's "no" is "did not report"; ours carries the
   provenance ladder (proved@T0 = any input vs proved@T1 = this input, any grid)
   — §I.1's claim ladder appears directly in the evaluation tables.
4. *Scoped ground truth — the corpus audits the claim ladder itself.* DRB's
   truth is a flat per-file yes/no; ours labels per (kernel, launch-params), so
   a kernel with several differently-labeled launches acquires a DERIVED
   kernel-level truth: "∃ input that races". If the detector answers proved@T0
   (race-free for ANY input) on such a kernel, that is not an ordinary FP — it
   is a LADDER-SOUNDNESS violation (a universal claim contradicted by a labeled
   counterexample input), scored as its own severity class above FP.
   Precision matters here or the auditor itself lies: the derivation only
   counts yes-launches WITHIN the T0 claim's premises — same specialization
   (constexpr set) and non-aliased, in-bounds launches. An aliased in-place
   yes-label does NOT contradict a non-aliased T0 proof (the claim excludes
   aliasing by stated premise); a yes under a different BLOCK constexpr is a
   different specialization. Analogously, `race-confirmed` on a no-labeled
   launch is a replay-soundness violation (worse than FP). Parameterized pairs
   thus do double duty — detector benchmark AND provenance-hierarchy auditor —
   the operational face of departure 3.

**Outputs**: `evaluation/results/<corpus>.jsonl` (schema above, versions/seeds in
the header) + generated `RESULTS.md` (five-state distribution by corpus, kind
buckets, headline numbers, DRB-style table, per-pattern table, timing
percentiles). The 2-D concretization map (§I.2) exports from the JSONL (each
row's terminal state + front-end determines its point); the figure script stays
out of the harness proper.

**Build order**: (1) LaunchSpec + harness/runner skeleton, smoke on golden
kernels (~½ day); (2) Phase A pairs + first report (~1 day); (3) Phase B
tutorials (~1 day, mostly LaunchSpec handwork); (4) mutation mode + Phase C
(~1–2 days). First full RESULTS.md ≈ 3–4 days; the five-state distribution is
visible after Phase A (~1.5 days in).

- **T0 stretch, off the critical path**: symbolic loop bounds (`lower ≤ i < upper` plus
  step-divisibility constraint), accept nonlinear `unknown` → the kernel simply lands
  on T1 per the ladder; whatever reaches T0 becomes the paper's "upper bound" section.
  The harness records the tier-selector fields from day one, so the stretch's impact
  is a re-run diff.

## III.3 Timeline & risks

Total ≈4.5–5.5 weeks; end-to-end capability lands at S3 (~2.5 weeks) so evaluation and
implementation overlap rather than serialize.

| Risk | Mitigation |
|---|---|
| T0 nonlinearity (`pid × sym_stride`) → Z3 `unknown` | linearity gate skips hopeless T0 attempts; the T1 primary target is all-linear; the ladder guarantees every kernel lands on some rung |
| S2 edits the shared reader the sanitizer depends on | sanitizer suite in CI must stay green; reader stays mechanism-only, policy differences live in the clients |
| Z3 `unknown`/timeout at any rung | unsupported-not-race policy — never report an unsat that wasn't proven |
| C3 diff noise from masked lanes | align the record convention before enabling the check |
| C2 replay of a symbolic-grid witness | replay uses the witness grid dims + captured args, executing only the two witness pids; T0-witness replay (materializing witness-shaped tensors) is stretch |
| free-variable masks flood reports with spurious races | every SAT passes through C2; `race-unconfirmed` is a distinct terminal state, reported as *potential*, never as confirmed |
