# Race Detector — Remaining Work

Companion to `race_detector_static_hybrid_plan.md`. LANDED (all
independently verified 2026-07-09/10, latest at b2d279c): Part III
S1–S6 — shared TTIR reader, scf.if path conditions, per-term DataDep
policy, the T1 global-memory track, the tier selector, the C2/C3
channels, RMW-return modeling with the guarded counting axiom, and
the await abstraction — plus the whole S5 evaluation program:
harness, TritonRaceBench (42 rows / 19 patterns, precision = recall
= 1.0, witness 17/17, ladder audit zero, terminal-identical
back-to-back runs), tutorials and liger corpora, mutation mode
(35/37 proofs flip), RQ2 headline aggregation, RQ3 scaling sweeps
(all five predicted shapes confirmed), RQ5 ablations (7/25
attributable flips), verdict-attribute emission (whose tests exposed
and closed the atomic grid-pinning soundness gap), and T0 symbolic
loop bounds (iteration-existence premise; trb019 proves for every
trip count). Post-b2d279c: the unread-pid-axis grid pinning for
non-atomic graphs was found unsound by running the ORIGINAL
aiter#3091 kernel (`--corpus aiter_originals`): the distillation's
phase-2 pid read had masked the class, and a no-pid broadcast store
at grid (4,) was falsely proved while the interpreter reported the
WAW. symbolic_grid now ENFORCES the launch contract instead of
assuming it (unread axes floor at the real launch extent, T0 and
T1; atomic T0 stays symbolic), the ladder audit gained
grid-geometry premise compatibility, and the flipped regression
test documents the new semantics. Zero collateral: benchmark,
tutorials, and liger terminals are unchanged line for line; 228
race-detector tests pass. The checked-item histories live in this
file's git log and the commit messages.

What remains, ordered by paper impact:

## 1. M5 — shared-track evaluation (the only item still blocking paper placeholders)

DESCOPED 2026-07-10 per the advisor: sell the idea with z3py; the
per-query SMT-LIB2 emission / interchange-format deliverable is
dropped (z3's native to_smt2 covers any future need). Remaining:

- [x] Evaluation sweep, sm80 half — landed
      (`evaluation/shared_track.py`, writes `results/SHARED_TRACK.md`):
      tutorial matmul (inner strides folded to 1, mirroring real JIT
      specialization — a runtime inner stride defeats the contiguity
      proof and the pipeliner never emits cp.async) and the persistent
      softmax, × `num_stages ∈ {1..4}` at sm80. Matmul proves at every
      stage count (4/6/8 async copies at 2/3/4; ~10 ms analyze);
      softmax abstains honestly (conditional region inside the
      pipelined tl.range loop — the documented Track 1 boundary);
      stage 1 is the no-pipeline trivial row. Mutation-detection
      matrix: weaken-wait, delete-wait, single-buffer — every
      applicable cell DETECTED (single-buffer n/a at stages=2, where
      the rotation is already depth 1).
- [x] Evaluation sweep, sm90 half — landed with M4 tranche 1
      (2026-07-10): matmul proves at stages 2..4 (RAW via async_wait
      counting AND the new WAR via warp_group_dot_wait pendings
      counting, both UNSAT); stages=1 abstains honestly (generic
      local_alloc store feeding a wgmma read crosses the generic→async
      proxy boundary — the documented model gate); softmax rows
      unchanged. Matrix gains weaken_pendings + delete_dot_wait
      columns: every applicable sm90 cell DETECTED, single_buffer at
      sm90 stages=2 now applicable (depth = num_stages) and caught as
      WAR. CS3 case study: pendings+1 leaves the previous iteration's
      wgmma read pending on exactly the slot the next cp.async
      overwrites — a WAR the sm80 model cannot express.
- [x] Case studies — both captured from the matrix with solver
      witnesses: CS1 missing `async_wait` (matmul @2: 4 RAW reports;
      prologue prefetch vs k_load=0, slot 0) and CS2 insufficient
      buffering (matmul @3 single-buffered under unchanged prefetch
      distance: 4 RAW reports — the producer's cp.async targets the
      slot the consumer still reads). Narratives in SHARED_TRACK.md
      feed the paper's §7 pipeline placeholders.

## 2. Benchmark corpus growth (feeds the paper's rq1 tag)

- [x] Four litmus variants — landed as trb020–023 (TritonRaceBench now
      52 rows / 24 patterns, precision = recall = 1.0, witness 19/19,
      ladder audit PASS):
      trb020 partially overlapping masks (same kernel, labels flip with
      the k1/k2 scalars; single-writer pid==0/pid==1 branches — a parity
      split would put two same-branch blocks on one range for any grid
      ≥ 3 under the every-grid claim, a corpus-design bug the solver's
      own witness caught); trb021 release-only / acquire-only guarded
      P/C (both racy in the dynamic column, acq_rel control clean;
      static abstains honestly with cas-synchronization);
      trb022 acquire-on-failure positive (consumer CAS with cmp=7 never
      succeeds, yet its acquire READ of the released value synchronizes
      — dynamic proves clean, relaxed twin races; e2e pair pinned in
      test_race_detector.py); trb023 oversized (2048 > 1024 cap) flag —
      deliberately UNLABELED: rf-init cap exceeded → rf_unknown (no sw)
      → conservative race report on a race-free program, the
      monotonicity-lemma over-report demo (labeling it would score the
      designed behavior as an FP).
- [x] cta-scope atomic-pair litmus — trb024: cross-CTA cta-scoped adds
      at one cell report (STATIC-track verdict, races-unclassified);
      the gpu-scoped twin proves at T1 (mutually atomic).

## 3. Moral-strength conflict refinement (feeds the paper's memory-model tag)

- [x] AUDIT RESULT: the implementation already matches Tile IR moral
      strength — `hb_common.conflicting_access_modes` exempts an
      atomic pair only under inclusive scopes (both non-cta for the
      cross-CTA queries), same width, and the exact same address; the
      TODO's "exempts ALL atomic pairs" described the PAPER's Def.
      conflict, not the code. Semantics now pinned by
      tests/unit/test_moral_strength_scopes.py (9 tests: gpu/sys
      inclusive-exemption cells ×3, cta-mismatch raciness ×4,
      width/address-torn raciness ×2) plus the trb024 corpus pair.
      The paper can update Def. conflict and drop the divergence
      caveat, citing these tests as the implemented-semantics record.

## 4. M4 — sm90/Hopper (UNGATED 2026-07-10; tranche 1 landed)

- [x] Tranche 1 — the wgmma agent: `ttng.warp_group_dot` smem operands
      are async reads (they join the RAW machinery as pseudo-loads
      guarded by the cp.async wait; a memdesc operand that does not
      resolve to a local_alloc fails closed), and
      `ttng.warp_group_dot_wait {pendings=N}` is a per-agent counting
      wait that opens the WAR direction — a copy must not overwrite a
      slot while a wgmma read of it can still be pending (all waits in
      effect at the copy constrain; sm80's lockstep argument does not
      retire the async MMA agent, so WAR is genuinely new here).
      `fence_async_shared` is vocabulary-accepted (only ADDS ordering
      the model never relies on; the generic-store-into-async-read
      shape it orders is gated unsupported). nvmma_shared layouts
      landed in layouts.py from the recon closed form (8×(8W/E) core
      tile, vec=128/E, perPhase=128/W, maxPhase=W/16, inner-first tile
      repetition); the LinearLayout oracle still aborts on shared
      encodings in the 3.7.1 wheel, so the differential test
      cross-checks closed form vs the independent basis construction
      (bijectivity + inverse consistency, 7 cases incl. transposed,
      col-repetition, W=0). Stock sm90 golden dump: proved race-free
      (was unsupported); pendings=2 already races (stock is exactly
      tight at 1). Mutation pins: off-by-one/weakened/deleted dot-wait
      → WAR; weakened async_wait → RAW naming the wgmma reader.
- [ ] Tranche 2 — TMA descriptors + mbarrier phase/arrive-count
      modeling + `ttg.warp_specialize`. Needs fresh golden dumps
      from descriptor-based kernels (`tl.make_tensor_descriptor`
      sources — block-ptr kernels get rewritten to plain pointers);
      ttng TMA/mbarrier ops outside the tranche-1 subset still
      degrade to honest unsupported (pinned by test).

## 5. Results landing figure — script landed (paper inclusion still an
## advisor call)

- [x] `evaluation/concretization_map.py` (separate from the harness)
      exports the plan §I.2 map from the results JSONLs: terminal
      state → (concretized, stays-symbolic) point; proofs /
      conditional proofs / static reports / confirmed / unconfirmed
      classes; abstentions in a residual table. Artifacts:
      CONCRETIZATION_MAP.{md,csv,svg} — the SVG is dependency-free
      (no matplotlib in the env), the CSV is pgfplots-ready, and the
      unreachable memory-without-paths column is hatched with the
      §I.2 asymmetry note. Current 109 rows: 11 at T0, 31+7 at T1,
      18 static reports, 17 confirmed + 2 unconfirmed on the
      interpreter point, 18 residual. Whether it enters the paper is
      pending the next advisor alignment; demoted from "core figure"
      2026-07-09 per the contribution-triad feedback.

## 6. S6 stretch items (require B + C1 together; none block the paper)

- [ ] Ticket lock: needs the bounded reads-through chain OVER
      unmodeled grid instances beyond the counting axiom's
      single-record guard (two RMW records — next_ticket and
      now_serving — interact).
- [ ] Looped work-queue fetch: RMW inside scf.for needs
      per-iteration observation symbols (one var per iteration, or
      an uninterpreted function of the loop index) before the
      counting axiom can extend.
- [ ] pingpong_phase (await nested in scf.for with expected =
      f(LoopVar)): parses and encodes today, but the awaited atomic
      keeps old_value=None inside loops (no rf), so it lands on
      reports, not proofs.

## 7. Small refinements (non-blocking)

- [x] C2 per-site footprint keying — landed. Replay footprints and
      report foci key by (base, kind, USER SOURCE LINE); the recorder
      resolves the INNERMOST user frame (capture_current_source_location
      resolves the OUTERMOST — the launch call site — and keyed every
      access to one constant line), which matches the reports' TTIR loc
      lines exactly. The ambiguity gate narrows to same-line/no-loc
      collisions only; missing lines classify unavailable (sound).
      Recovery demonstrated both ways on the same-tensor two-site
      kernel: dead widened site → classified unconfirmed (partial)
      instead of declined; LIVE widened site → graduates to a
      replay-confirmed second report (previously unclassifiable). C3
      keeps (tensor, kind) granularity by aggregating over sites — line
      attribution noise must not read as a lowering divergence.
- [x] Interpreter × numpy 2.x — landed as a shim over triton's
      interpreter patch (upstream's `_patch_lang_tensor` installs
      `__index__ = int(handle.data)`, which numpy 2 rejects for the
      shape-(1,) wrappers of scalar args): both patch paths (the triton
      frontend's patch_lang and the gluon simulation) re-install a
      size-1-safe `__index__` AFTER triton's. Recovered coverage:
      scalar-bound loop kernels' C2/C3 came back alive —
      trb008/trb019 racy rows upgraded races-unclassified →
      race-confirmed, C3 'agree' where it was unavailable, and the
      gluon scalar-range test passed. Two more gluon version-skew fixes
      rode along (tcgen05_commit pred optional for 3.6; the TMA example
      falls back when tensor_descriptor.nbytes_per_cta is absent).

## 8. Repo hygiene: pre-existing test-isolation bugs — ALL RESOLVED

- [x] TRITON_INTERPRET at module level in test_multithreading —
      REMOVED outright: the trace machinery constructs
      InterpretedFunction itself (trace.py), so the env var was
      redundant; verified by import-order probe (later modules keep
      JITFunction kernels) and the module's own 10 tests. This was
      also the true root of the local "compiled sanitizer/detector
      environment family": those real-compile tests were being fed
      poisoned kernels at collection.
- [x] draw.py sys.setrecursionlimit(100000) at import — moved into
      collect_grid() (both public entries route through it), so the
      process-wide bump no longer defeats recursion-exhaustion tests.
- [x] Wrapper/CLI failures — NOT REPRODUCIBLE here: console scripts
      present (uv sync installs the project), 5/5 pass sequential and
      xdist. The failures were another environment's missing project
      install; nothing to fix in-repo.

      Net effect of §7+§8 together: the FULL local suite is green for
      the first time — 763 passed, 0 failed, sequential AND -n auto
      (down from 34 baseline failures at the branch's start).

## Corpus & experiment backlog (the paper's extension placeholders)

Each item pairs a paper placeholder with the implementation work it
needs; none blocks submission.

- [ ] Pre-fix aiter scan (paper RQ2/RQ4): vendor the MoE-routing
      kernel family at the repository state BEFORE the #3091 fix and
      run the corpus protocol over it — the lowest-cost path to a
      "previously undetected race" data point (the detector flagging
      the bug class at the pre-discovery code state, plus any
      neighbors). New corpus module per the aiter_originals pattern.
- [ ] TorchInductor corpus (paper RQ2): dump kernels from a
      torchbench sweep, author LaunchSpecs, run coverage — generated
      code nobody hand-reviews is the second-best discovery ground.
- [ ] vLLM / unsloth / flash-attention corpus modules (paper RQ2
      scale; import-or-vendor per the liger/tutorials patterns).
- [ ] Witness pretty-printer (paper RQ6 / case studies): format a
      report (line pair, instances, byte, type, evidence,
      qualifiers) from the JSONL/report objects; the case-study set
      should include one conservative-flagged (trb023) and one
      termination-conditional (any await row) witness. Tiny; mostly
      unblocks writing.
- [ ] External-baseline adapters (paper RQ5): GPU-GATED. Two of the
      planned baselines are already covered by the ablation switches
      (no-hb = the overlap checker, no-load-values = the concrete
      replayer); the external ones (compute-sanitizer racecheck,
      thread-level tools) need real hardware and an applicability
      pass first (racecheck covers shared memory; our litmus corpus
      is mostly global).
- [ ] Address-position lifting (paper §4 placeholder + the three
      doubly-undecided benchmark rows): select terms in ADDRESS
      position with the read-only flow check extended to index
      tensors and the witness side conditions revalidated — the one
      remaining large feature; scatter litmus pair + benchmark row
      flips + RQ5 complementarity refresh follow. Post-submission
      unless prioritized.

## Decision points (not tasks)

- PR layout: `race-detector-z3-demo` now carries the plan-doc
  restructure plus S1–S6, the evaluation phases (A–C, mutation, RQ
  instrumentation, T0 stretch), and the docs commits; decide
  whether to merge as one PR, split per step, or split
  detector-core / evaluation-harness before opening against main.
- Next advisor alignment carries: Q5 (M4/sm90 submission scope),
  the landing-figure question, and confirmation of the executed
  contribution-triad reframing.
