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
- [x] Category 8a — communication kernels, single-GPU half (Keren
      2026-07-11): comm/comp SM-partition semaphore, DeepSeek-V3
      style. Kernel shape: pid range split into a comm role and a
      comp role; the comm side publishes through a global-memory
      payload + semaphore (atomic release add / store), the comp
      side polls the semaphore (await) before reading the payload.
      Expressible TODAY with the shipped B+C1 machinery: this is the
      guarded producer/consumer family with a role split on pid
      instead of pid parity. Racy twins: drop the acquire on the
      poll, poll the wrong counter value, or skip the poll on one
      branch of the role split. Reference shapes: upstream gsan's
      `_single_cta_atomic_sync_kernel` / `_single_cta_no_atomic_sync_kernel`
      (python/test/gsan/test_symmetric_memory.py), re-cut at gpu
      scope on one device. LANDED 2026-07-11 as trb025 (pattern
      "comm-comp", control + 3 racy twins, tritonracebench 56 rows)
      plus 4 static-track e2e pins (test_comm_comp_pattern.py):
      control proves at T1+assumes-termination, relaxed-poll /
      poll-initial-value / role-branch-skips-poll all report on the
      payload pair with needle-exact witnesses. One machinery note
      recorded in the corpus: the arrive is a release XCHG — a
      release ADD-arrive plus the add(0) acquire poll puts two
      value-interacting RMW records on the semaphore (the S6
      ticket-lock boundary) and the control then reports; the true
      multi-arrival counting arrive lands with the S6 stretch.

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

## 3b. Real-kernel corpus growth: TritonBench_G_v1 (landed 2026-07-10)

- [x] thunlp/TritonBench `data/TritonBench_G_v1` (184 real-world
      GitHub-crawled operator files, Apache-2.0) VENDORED under
      evaluation/kernels/tritonbench_g_v1/ (byte-identical, LICENSE +
      README pinning upstream commit 603e28a5; excluded from repo
      formatters) — vendored rather than submodule/pip for artifact
      self-containment (archived tarballs keep it, runs offline).
      Launches captured ONCE on a CUDA box by
      evaluation/tritonbench_capture.py (test blocks execute at import
      on GPU): a JITFunction.run hook records per (file, kernel) the
      first real launch — name→value binding split into runtime args /
      constexprs, tensor descriptors (shape/dtype/init class incl.
      observed int ranges so index tensors stay in-bounds/contiguity/
      alias groups), exact scalars, resolved grid → 202 launches from
      179/184 files (5 genuine failures: 2× removed triton.ops, 2×
      smem over hardware limit, 1× autotune timeout; 24 kernels
      skipped with reasons: 14× non-contiguous, 6× tl-dtype constexpr,
      rest misc). evaluation/kernels/tritonbench_g.py rebuilds CPU
      launches anywhere: execs only pre-separator kernel sections,
      None-valued optional pointers stay positional placeholders and
      double as constexpr None for the static signature (the harness
      dedupes the kwarg — the middle-None shift bug broke the dynamic
      column before), Autotuner/Heuristics unwrapped BY TYPE (the
      wrappers proxy arg_names). Corpus.provenance carries the
      upstream commit into the results header (liger's version+commit
      recording landed alongside).
- [x] Sweep (202 rows): 99 proofs (69 proved@T1 + 30 proved@T0, 49%
      on unfiltered real code), 77 honest abstentions (36 indirect
      addressing — the documented DataDep boundary — 7 data-dependent
      bounds, 4 nested loops, 2 unstructured cf), 23
      races-unclassified, 3 kernels that no longer compile upstream.
      The 23 flagged rows were triaged by a 23-agent workflow with
      independent cross-checks: 46/46 verdicts agree — ALL are the
      T1 any-grid semantics meeting wrapper-coupled launches (the
      kernel is safe only because grid = cdiv(dim, TILE); the any-grid
      witness pids exceed the captured grid, e.g. (0,10,0) vs (2,2,4)),
      not corpus artifacts and not detector bugs; the dynamic column
      is clean on every one.

## 3c. Launch-scoped verdict tier (LANDED 2026-07-15)

Decision (Hao): (c)-semantics on (b)-machinery with three guardrails —
scope is per-verdict, not a global binary (the taxonomy already had
per-scope proofs; this adds the missing rung).

- [x] Machinery: after any any-grid SAT, `_launch_scoped_requery`
      re-asks the SAME encoding with every grid axis pinned to the
      launch extent (generalizing symbolic_grid's unread-axis pinning
      to all axes; `tl.num_programs` interns `grid_i` by name, so the
      pin is an `extra_assumptions` equality — no re-encode, zero
      solver changes). Extent-UNSAT ⇒ `proved@T1-launch` +
      `grid_fragile` attribute carrying the any-grid evidence (hazard
      wording, never "race"); extent-SAT ⇒ the race path continues
      with the PINNED reports (witnesses in-extent by construction —
      C2-replayable); Z3-unknown ⇒ fall back to the any-grid reports,
      fail-closed on the claim. Sound from widened evidence too:
      widening only enlarges footprints, so over-approx extent-UNSAT
      implies real extent-UNSAT.
- [x] Guardrail 1 (wording pair): verdict attrs gain
      proved_scope="this-params-this-grid" + independent grid_fragile
      bool; evidence in static["grid_fragile"], never in witnesses.
- [x] Guardrail 2 (counting): SWEEP_REPORT §3 splits decided-clean by
      scope (any-grid vs launch-scoped), grid-fragile its own column;
      findings stay 3. Concretization map gained the
      "pid + trip (grid = launch)" y-row.
- [x] Guardrail 3 (order, (c) ⊃ (b)): pinned-UNSAT relabels; the
      in-extent boundary keeps carrying race-confirmed (aiter
      unchanged on the re-sweep).
- [x] Full 14-corpus re-sweep at the landed state: ground-truth
      scorecard IDENTICAL (precision=recall=1.0, 12 race-confirmed,
      13 races-unclassified all in-extent, ZERO grid-fragile rows in
      GT — no claim inflation); 51/52 wrapper-coupled rows →
      proved@T1-launch (+3 borderline rows joined; net T1-launch=52);
      the 1 holdout (torchao common split-k matmul) stays
      races-unclassified because the pinned query is Z3-undecidable
      even at 120s (nonlinear split-k scheduler arithmetic) — the
      terminal now precisely MEANS "any-grid SAT + launch-scoped
      undecidable". Pins:
      test_out_of_extent_exact_sat_lands_launch_scoped_proof,
      test_widened_out_of_extent_sat_lands_launch_scoped_proof.

## 3d. Address-position lifting (PRIORITIZED 2026-07-11, Hao)

Promoted from the backlog on the TritonBench evidence: 37 of 202
rows abstain on indirect addressing (36× arith-over-loaded-data +
1× direct loaded value), the single largest class, and the
interpreter currently refuses them too. The model already
covers the lifting (paper §4: the same select machinery as
value/mask position); what is missing is validation, because
address position has NO sound fallback direction (a free address
makes every query SAT; a wrong one breaks witness soundness AND
can hide real overlaps). The hand-off spec LANDED as
`address_position_lifting_spec.md` (2026-07-11, adversarially
verified 6/6 against the code): the lift is interpreter-front-end
only per the §I.3 placement rule, the entire snapshot/domain/
read-only machinery already exists for value position, and the
happy path needs only the `_VALUE_DEPENDENT_ADDRESS_OPS` gate
change — the spec's work items below are validation + tests.

- [x] (i) select(A_T, t) terms in event ADDRESS expressions with
      per-lane lowering (an index TILE means lane λ addresses
      dst + select(A_T, base+λ)) and domain constraints
      t ∈ dom(T) so out-of-domain indices cannot fabricate or hide
      overlaps.
- [x] (ii) read-only flow check extended to INDEX-source tensors,
      exactly like value sources (region tracking; a kernel that
      writes an index tensor fail-stops — stale snapshots in
      address position are wrong in both directions).
- [x] (iii) the byte-overlap query over select-containing
      addresses (arrays + linear integer arithmetic; validate the
      encoding shape and cost over the m² query loop).
- [x] (iv) witness-soundness revalidation: re-walk the A1/A2
      transport of Theorem thm:witness with select in addresses;
      the acceptance tests ARE the backing — written-index
      fail-stop, OOB-index domain tests, index/data tensor
      aliasing, masked-gather default interplay.
- [x] (v) Definition of done — ALL LANDED 2026-07-11: scatter
      litmus pair race@interp/proved@interp with needle-exact
      witnesses; trb013 plain-fetch flipped (counting-axiom rows
      pinned unchanged); tritonracebench 56 rows at
      precision=recall=1.0, witness 25/25, audit zero; TritonBench
      37-row migration measured (11 decided: 7 proved@interp +
      4 race@interp; abstention buckets: 10 pid-divergent host
      control flow, 7 per-instance bounds, 5 snapshot cap, 3
      missing-other, 1 wrapper coercion; corpus unsupported
      76→55); RQ5 refreshed with BOTH directions (mask-position
      erasure + the new ADDRESS-position FABRICATION demo — the
      no-sound-fallback premise, empirically). Composed-dispatcher
      terminals race@interp/proved@interp landed with
      dynamic-witness serialization and the interp-disagreements
      audit bucket (6 on TritonBench: randint index-table rebuild
      collisions — reconstruction fidelity, not unsoundness).
      FOLLOW-UP LANDED 2026-07-12: int/bool tensors ≤8192 elements
      now carry exact VALUE SNAPSHOTS at capture
      (evaluation/capture_common.py; supersedes the randperm design
      — snapshots also preserve legitimate duplicates and monotone
      offset tables, which randperm would have destroyed). GPU
      re-capture + re-sweep outcome for the 6-row bucket: 2 retired
      (tb_token_softmax_bloom/llama → proved@interp), 4 fully
      triaged — 2 GENUINE races in the crawled corpus
      (tb_nested_loops_processing: kernel never reads program_id
      under grid=(2,), all-pairs WAW; tb_quantize_kv_copy:
      snapshot-faithful duplicate scatter destinations, witness pids
      match the duplicated Dest_loc positions), 1 interpreter
      semantic divergence (tb_masked_select: Python `and` on block
      tensors — interpreter truthiness drops the select_mask store
      predicate; compiled lowering is elementwise logical_and, so
      the GPU kernel is race-free), 1 DETECTOR BUG FIXED
      (tb_cache_transform: ReduceSymbolicExpr folds over ONE
      symbolic lane, so tl.max in an address degenerated to a
      solver-chosen element and fabricated 0/1/2 WARs
      nondeterministically at a fixed seed; the reduce family —
      sum/max/min/xor_sum/reduce_or/argmax/argmin — is now gated in
      _VALUE_DEPENDENT_ADDRESS_OPS, flipping the row to a
      deterministic honest abstention; lift only with a true
      per-lane fold). The tb_triton_argmax crash row is the SAME
      `and`-truthiness divergence inside the C3 differential
      replay: the all-True mask sends the interpreter's native
      masked load ~533MB past a 4MB tensor → SIGSEGV with empty
      stderr. Original definition: scatter litmus pair (racy overlap +
      disjoint-index control) with confirmed/exact witnesses; the
      three doubly-undecided benchmark rows (trb010 gather/scatter,
      trb013 plain-fetch) flip from unsupported to verdicts; a
      sample of the 36 TritonBench indirect rows decides through
      the composed dispatcher (per-launch scope; the captured
      launches record observed index ranges precisely so snapshots
      stay in-bounds); RQ5 complementarity numbers refresh.

## 3e. Small fragment extensions (approved 2026-07-11, Hao; independent, any order)

- [ ] Snapshot-lifted loop bounds (8 TritonBench rows — 7 upper
      bounds + tb_block_sparse_attn's lower bound): a loop
      bound loaded from a read-only tensor becomes a select term
      inside the iteration-existence premise (the T0-stretch
      machinery shape); per-instance bounds are then sound where a
      single concrete bound from the analyzed instance would not
      be. Same read-only side condition and fail-stop as value
      sources.
- [ ] Nested-loop support in the TTIR reader (4 TritonBench rows +
      the trb011 pair): the interpreter already handles nested
      loops (trb011 decides correctly in the dynamic column), so
      the composed dispatcher rescues these today; reader support
      moves them into the static track's scope with grid-generic
      claims.
- [ ] Unstructured control flow (2 TritonBench rows): encode
      cf.cond_br / early-return as path conditions per the
      existing scf.if machinery (structurize or gate records on
      the branch condition). Note the interpreter CANNOT rescue
      these (instance-dependent control flow breaks the
      full-template assumption), so the reader is the only route.

## 3f. Real-kernel corpus growth: flash-linear-attention (landed 2026-07-12)

- [x] fla-org/flash-linear-attention as the THIRD real-code corpus:
      pip-pinned fla-core==0.5.1 per the liger pattern (upstream tag
      v0.5.1 = 2e38c1fa, recorded in every results header via
      runner._fla_provenance); evaluation/kernels/fla.py HARD-FAILS
      on version drift (installed != captured) and on any unresolved
      kernel — never a silently shrunken corpus. Capture:
      evaluation/fla_capture.py drives 64 GPU-validated cases (23 op
      families × chunk/fused_recurrent/parallel × fwd+bwd, dense +
      varlen cu_seqlens) under the shared hook layer
      (evaluation/capture_common.py, extracted from the TritonBench
      capture; autotune left ON — benchmark launches are real
      launches, first config captured). 378 kernel specializations.
      Sweep (jobs=8): 122 static proofs (107 proved@T1 + 15
      proved@T0), 12 proved@interp, 1 race@interp — triaged GENUINE:
      fused_chunk_based_fwd_kernel's z store omits the `if i_v==0:`
      guard its own bwd twin applies at 8 sites, giving a benign
      same-value inter-program WAW (seed-independent, pid pair
      (0,0,0)/(1,0,0), addresses pid-only) — a label-error row, not
      an FP; 9 races-unclassified (the §3c launch-scoped class); 227
      unsupported = indirect-address 147 + control-flow 31 +
      nested-loop 20 + data-dependent-bound 19 + other 7 + solver 1;
      5 timeouts (fused_recurrent T-loop T1 cost); 2 compile-errors
      (path_attn cumprod_householder_bwd). Ladder audit PASS.
- [x] KEY DISCOVERY (corrects the plan's premise): tl.make_block_ptr
      NEVER reaches the shared TTIR reader — triton's make_ttir
      pipeline runs rewrite_tensor_pointer, so block pointers arrive
      as plain addptr arithmetic. The 91-of-153-files block-ptr
      prevalence is IRRELEVANT for ASTSource corpora; the real fla
      coverage lever is §3e-style lifting in the COMPILED track —
      147 indirect-address rows are dominated by varlen
      cu_seqlens/chunk_indices load chains (small read-only int
      tensors: exactly the snapshot-select shape §3d proved out on
      the interpreter track), plus nested loops (20) and scf
      control flow (31). This multiplies §3e's row support by ~10×.
- [x] Capture-layer hardening (adversarial review, 7 confirmed
      findings, all fixed + re-captured): launch-opt kwargs that
      name DECLARED kernel params bind as args (recovered
      fused_recurrent kda/gdn2 fwd kernels — `num_stages:
      tl.constexpr` shadowing); dedup fingerprints cover the FULL
      record incl. scalar values/snapshots/aliases (un-merged gsa's
      scale=1 chunk_gla_bwd twins); InterpretedFunction accepted in
      kernel resolution (TRITON_INTERPRET=1); mkstemp + guarded
      parse in both capture drivers (shared-/tmp collisions).
      runner --jobs N landed for parallel sweeps (~35 min vs ~5 h at
      367 rows; keep DEFINITIVE paper sweeps at jobs=1 — wall_s and
      near-watchdog rows shift under load).
- [x] Upstream fixes for the three genuine races filed 2026-07-12
      (PR text describes mechanism + repro only — no tool/paper
      mention, double-blind): fla-org/flash-linear-attention#1018
      (fused_chunk based fwd z store guarded to i_v==0, matching the
      bwd twin's own convention; upstream test_based 5 passed;
      patched row re-checked 4→0 reports), thunlp/TritonBench#10
      (nested3 grid clamped to min(n_cols//4, 1) — byte-identical
      outputs incl. the n_cols=2 empty-grid case),
      thunlp/TritonBench#11 (DestLoc randint→randperm, unique
      KV-cache slots; Case-4 invalid input untouched — assert fires
      pre-launch). Our vendored TB copy and the fla-core 0.5.1 pin
      stay UNCHANGED (the racy versions are the evaluation
      evidence); on upstream merge the paper gains
      "confirmed/fixed upstream" citations.
- [ ] Interpreter `and`-truthiness divergence class (advisor
      review): Python `and`/`or` on block tensors silently drops
      mask terms under the interpreter (upstream patches
      tensor.__bool__ → True), while compiled lowering is
      elementwise logical_and — fabricates tb_masked_select's WAW
      and SIGSEGVs the C3 differential replay on tb_triton_argmax
      (all-True mask → native masked load ~533MB OOB, empty-stderr
      crash row). Candidate: pre-trace AST scan for BoolOp over
      tensor expressions → mark the row interp-divergence-suspect
      and refuse replay (fail-closed), vs. an upstream interpreter
      fix.
- [ ] Reduce per-lane fold (lifts the new reduce gate): fold
      reduces lane-wise over the arange/snapshot domain instead of
      the current single-symbolic-lane collapse, then re-admit
      reduce results into event addresses — decides
      tb_cache_transform-class rows (max-of-prefix-cumsum
      addressing) instead of abstaining.

## 3g. Real-kernel corpus growth: FlagAttention (landed 2026-07-12)

- [x] FlagOpen/FlagAttention as the FOURTH real-code corpus (13
      kernels: flash/piecewise fwd+3-bwd, split-kv pair, paged +
      v2-reduce, total-attention; Apache-2.0, active upstream, runs
      UNMODIFIED on triton 3.6). No PyPI release → git-pinned pip
      install (flag_attn @ git+...@41fc31d); provenance flows from
      pip's direct_url.json through _package_provenance, no release
      table. Shared plumbing extracted on the rule of two:
      capture_common.run_case_capture/capture_one_case/fingerprint
      (case-driven capture main, was fla_capture-private) and
      kernels/_captured.build_captured_corpus (version hard-check +
      fail-loud unresolved + name disambiguation, was fla.py-private)
      — fla regression-checked at 378/378 with identical provenance.
      Capture: 10 fp16 cases (causal/non-causal, GQA, dropout/philox,
      non-divisible seqlen, aux outputs, split-kv decode, paged ×2,
      piecewise), 28 specializations, 0 failures, no autotune (sm89
      falls back to the hand-written 32x32 config).
- [x] Sweep (28 rows): proved@interp 1 (split-kv combine — interp
      rescues its nested loops), races-unclassified 10, unsupported
      17, audit PASS. ALL 28 attributed:
      * NEW abstention class, 14 rows — PID-AFFINE LOOP BOUNDS
        ("other: loop bound is not concrete at launch"): the flash
        causal inner loop runs to (pid_m+1)*BLOCK_M-style bounds,
        affine in pid, which T1 refuses (wants concrete scalars) and
        one-shot symbolic capture concretizes. Distinct from
        data-dependent bounds and representable in the existing
        affine machinery — lift candidate below.
      * 10 races-unclassified: all witnesses have a pid OUTSIDE the
        launch extent (grid=[4,2,2] vs witness pid_0=4/12, pid_1=3/5
        — symbolic pid overflow walks into the next head/batch slice
        via flat strides). The §3c wrapper-coupled any-grid class,
        joining TritonBench's 22 and fla's 9.
      * paged lands EXACTLY on two queued §3e fragments:
        single-split → loaded context_lens loop bound
        (snapshot-lifted loop bounds), v2 → cf.cond_br. Both tracks
        abstain today; §3e now has attention-serving rows behind it.
      * flash_dropout bwd dynamic track aborts with
        "NotImplementedError: Patching math ops not yet supported" —
        philox/math interp front-end gap (small, separate).
- [ ] Pid-affine loop bounds lift (advisor review; NEW, motivated by
      14/28 flagattn rows + every flash-attention-style kernel): T1
      loop iteration-existence premises already quantify over pid —
      admit loop bounds affine in pid (and in concrete scalars) into
      the same premise instead of requiring launch-concrete bounds.
      The causal-attention inner loop is the canonical shape; expect
      most of the 14 rows to flip to proved@T1.

## 3h. Real-kernel corpus growth: FlagGems (landed 2026-07-12)

- [x] flagos-ai/FlagGems as the FIFTH real-code corpus and the
      race-relevant one: production ATen operators in Triton with ~150
      tl.atomic_* sites (scatter/index/histogram/embedding-bwd/loss),
      cumsum-addressed stores (unique/masked_select), and mm_streamk's
      inter-CTA spinlock. Git-pinned pip install @1051e56c (PyPI lags
      master by 1000+ commits; --no-deps dodges its numpy==1.26.4 pin;
      sqlalchemy added to the venv). 66 GPU-validated cases across 10
      families -> 82 specializations, 0 failures. Runtime-CODEGEN
      kernels (pointwise_dynamic modules under ~/.flaggems/code_cache
      with process-dependent names) are filtered to skipped_kernels via
      capture_one_case(module_prefix=...) — un-importable at rebuild;
      a tritonbench-style source-embedding scheme could recover them
      (backlog).
- [x] Sweep (82 rows, audit PASS): 42 decided-clean — proved@T1 22 +
      proved@T0 11 + proved@interp 9 (51% coverage, best of the real
      corpora; the counting axiom's first at-scale field test:
      vdot's atomic scalar accumulate proves at T0, bincount/histc/
      scatter_reduce/index_reduce duplicate-index variants all clean).
      36 unsupported = indirect-address 12 + pid-affine bounds ("other")
      12 + nested-loop 6 + control-flow 3 + solver 1 + spin-shape 1 +
      data-dependent-bound 1. 1 races-unclassified (bmm — witness
      pid_1=8 outside grid=[8,8,4], the §3c any-grid class). 1 timeout
      (mm_streamk's classic_mm sibling, 180s cap).
- [x] mm_streamk first_wave — the S6 PRODUCTION INSTANCE: static track
      abstains "spin-shape: scf.while carries values (iter args or
      results) — only the argument-free spin form is the await shape".
      Stream-K's spin (atomic_xchg arrive + atomic_cas busy-wait +
      partial-sum handoff) carries loop state, exactly outside C1.1's
      argument-free domain — first production motivation for the
      carried-value spin extension (S6 stretch).
- [x] Both race@interp rows triaged INTERPRETER-ARTIFACT, each naming
      a distinct toolchain defect:
      * weight_norm_kernel_first — the `and`-truthiness class, THIRD
        instance (weightnorm.py:83/93 `col_offset < N and row_mask`
        collapses to row_mask under the interpreter; store broadcasts
        over 2048 cols instead of 128; empirically pinned with an
        interpreter probe). The §3f BoolOp gate item now has three
        manifestations across two corpora. Cosmetic upstream PR
        candidate: `and` -> `&` (flag_gems's own convention in
        aminmax/svd/index_put).
      * embedding_dup — NEW DETECTOR BUG (two-copy solver lane model):
        _lane_identity_differs (two_copy_symbolic_hb_solver.py:507-530)
        treats ANY arange var differing across copies as two distinct
        lanes, but a kernel calling tl.arange twice on the SAME axis
        (embedding.py:27 mask arange, :28 cols arange) has both vars
        bound to the SAME lane coordinate physically; Z3 picks
        l27-differs + l28-equal -> phantom intra-instance same-address
        WAW (seed-independent, reproduced with a minimal two-arange
        twin). FIX QUEUED below.
- [ ] Two-copy lane-model coupling (detector bug, from embedding_dup):
      group a record's arange vars by the tile axis they span and
      constrain same-axis vars EQUAL within each copy (a lane has one
      coordinate per axis); "any arange differs" stays correct only
      ACROSS axes. Until then, intra-instance same-address claims on
      multi-arange records are fabrication-prone; consider gating
      records with >1 same-extent arange in address/mask as
      interp-divergence-suspect (fail-closed interim).
- [ ] Codegen-kernel recovery (backlog): embed the generated module
      SOURCE in the capture record (tritonbench-style exec at rebuild)
      to admit pointwise_dynamic/scatter-codegen kernels — today 3
      such kernels are filtered per run with visible skip reasons.

## 3i. Real-kernel corpus growth: torchao (landed 2026-07-13)

Record: 67 rows from pytorch/ao @ `bfbc842` (git-pinned `USE_CPP=0
--no-build-isolation` install — Triton kernels are pure Python, no
torch-ABI coupling; provenance via direct_url.json, version string
embeds the sha). Reality check: the repo holds ~102 hand-written
`@triton.jit` kernels (not the rumored 2000+ — that figure can only
count inductor-generated kernels, the codegen class we exclude by
design). 44/44 capture cases, 67 specializations, zero skips; the
sm89-unreachable families (fp8_sdpa: torch-2.11 init; nvfp4/mxfp8-CUDA
/mx-dim0/dim1: sm100 gates; comms: torch.distributed; one dead-code
kernel; common-matmul fp8 path: upstream KeyError) are documented in
torchao_capture.py's docstring.

Corpus-driven extensions landed with it (all generic, older corpora
byte-identical): strides capture + empty_strided rebuild for
non-contiguous args (17 skips unlocked; stride-0 broadcast handled via
de-overlapped slice copy); tl.dtype/torch.dtype constexpr round-trip
as tagged JSON (19 skips unlocked); _resolve_kernel namespace-scan
fallback + torchao corpus module publishes lazy-init closure kernels
(CustomOpDef closes over the gemm autotuner). Detector/harness fixes
it surfaced: MLIR fp8 spellings in the shared reader's _DTYPE_BITS
(15 pseudo-abstentions), host-compile GPU target now the real device
capability (fp8 false compile-errors).

Sweep: 23 decided-clean (5 T0 / 9 T1 / 9 interp), 36 abstain, 8
races-unclassified — all 8 witness-out-of-extent (§3c class), zero
genuine races.

- [ ] Scalar-pointer atomic_rmw reader shape (2 rows abstain with
      "atomic_rmw of a non-pointer value"): tl.atomic_max/min on a
      single-element global scalar — the fp8 global-amax idiom
      (f8nc _amax_atomic, moe 3d-transpose scales atomic_min). The
      reader only lifts tensor-of-pointer RMWs today.
- [ ] Non-contiguous in-bounds premise (11 rows): the T1 in-bounds
      premise assumes dense layout; column-major quant outputs need a
      strided-footprint premise (capture side already rebuilds them).
- [ ] Runtime-scalar loop bounds (8 rows: "loop upper bound is not
      concrete at launch"): bind non-constexpr scalar args to their
      captured values under the launch-scoped tier — rides §3c.

## 3j. Real-kernel corpus growth: tritonbench_meta (landed 2026-07-13)

Record: 41 rows from meta-pytorch/tritonbench @ `1edaf3e` (Meta's own
benchmark suite — DISTINCT from thunlp/TritonBench = the tritonbench_g
corpus). Git-pinned pip install; the dist version is a constant 0.0.1,
so the corpus module hard-checks the installed direct_url.json commit
directly. Reality check: ~102 hand-written @triton.jit in-repo (not
2000+; that counts only inductor codegen, our excluded class).

Capture is HARNESS-DRIVEN, not a case table: each case instantiates the
suite's own `BenchmarkOperator` with `--only <impl> --num-inputs 1
--input-id 0 --test-only --force` and runs it once, with
`module_prefix="tritonbench."` keeping only the suite's own kernels
(its liger/inductor/vendor backends are filtered — liger is already a
corpus, inductor is codegen). Registry-disabled impls were each tried
under `--force` and dropped only on a verified structural failure
(xformers/cutlass-ck/fbgemm/mslk deps, stream-k TensorDescriptor TMA
args, multi_cta cluster launch) — all documented in the capture
docstring. Generic reader extension it needed: `_resolve_kernel` now
also scans module-level CLASS bodies (tritonbench's softmax Operator
carries its @triton.jit kernels as class attributes).

Sweep: 20 decided-clean (5 T0 / 8 T1 / 7 interp), 20 abstain, 1
races-unclassified (out-of-extent flash-TMA artifact), zero genuine
races. gdpa atomics + layer_norm/softmax/rms_norm backward
lock-reductions all decide clean.

- [ ] Stream-k / TMA-descriptor operators (addmm+gemm streamk, TMA
      persistent matmuls): host-side TensorDescriptor args — capture,
      rebuild, and reader support are the M4 track; ~13-min autotune
      each, so excluded from the sweep for now.

## 3k. Detector fix: exact-race confirmation at unrolled same-line stores (landed 2026-07-13)

- [x] The C2 ambiguous-site gate (stops a dropped-mask WIDENED report
      riding an unrelated same-line access's overlap into a fabricated
      confirmation — test_c2_focus_blocks_fabricated_upgrade) also
      skipped EXACT reports whose store is unrolled by tl.static_range
      onto one source line (count>1 ⇒ ambiguous bucket). The aiter#3091
      kernel is that shape, so its genuine in-extent cross-block WAW
      landed on races-unclassified instead of race-confirmed. Fix: gate
      WIDENED reports only (`is_widened and any(... in ambiguous)`) — an
      exact report is a definite SAT witness whose access is live by
      construction, so the same-line bucket is its OWN real footprint
      and confirming it is sound. Pinned by
      test_c2_confirms_exact_waw_at_unrolled_ambiguous_site; the
      tritonracebench ground-truth scorecard and every out-of-extent
      §3c artifact (torchao 8, tritonbench_meta 1) are unchanged.

## 3l. Real-kernel corpus growth: tilebench (landed 2026-07-15)

Record: 56 rows (45 operators) from the group's own TileBench
(Deep-Learning-Profiling-Tools/Tilebench @ `224ec81`, branch
exp/llm_and_analysis_code_only). First LOCAL-CHECKOUT corpus: TileBench
has no packaging metadata, so `TILEBENCH_ROOT` (default
~/workspace/Tilebench, env-overridable) goes on sys.path and the
checkout HEAD commit is the pin — capture refuses tracked-dirty trees,
and `build_captured_corpus` grew an `installed_version=` parameter so
non-pip corpora ride the same drift guard.

Capture is harness-driven (tritonbench_meta pattern): each case runs
the suite's own `core.engine.run_benchmark_suite(op)` with
`case_indices=[0]` and `report_benchmark` monkeypatched out, so the
ONLY Triton launch is the engine's verification run on a normal stream
(the Proton/CUDA-graph timing path never executes — keeps recorder
tensor reads off a capturing stream). `autotune` stays False → every
impl calls its raw @triton.jit kernel with `_DEFAULT_CONFIG`, one
deterministic launch. 45/45 cases, 56 specializations, zero failures.

Strategic point: every operator also ships a cuTile twin
(impl_cutile.py) — this corpus is the Triton-side baseline for the
planned cuTile frontend (same-operator cross-DSL differential).

Sweep: 41 decided-clean (21 T0 / 15 T1 / 5 interp) = 73%, the highest
clean rate of any real-code corpus (small single-purpose benchmark
kernels). 11 abstain, 3 timeout (bitonic XOR-pair math, gaussian_blur
div/mod stencil — Z3-hard shapes; batched_matmul's bmm is a borderline
row that flips between the loop-accumulator abstain and the 180s
watchdog run-to-run), 1 races-unclassified
(linear_self_attention `_kv_kernel`: witness pid (0,32,0) outside
grid [32,32] — §3c out-of-extent artifact, 52nd instance), zero
genuine races. Notable proof: top_k_selection's bitonic exchange
network PROVES at T1 (div/mod pair-partition disjointness across
CTAs). destindex (duplicate-destination scatter, the quantize_kv_copy
family) abstains honestly on both tracks (indirect address; dest_loc
2048 elements > 1024 interp snapshot cap) rather than silently
passing. streamk first_wave is spin-shape (S6 production instance #2,
after flaggems mm_streamk).

- [x] Detector defect: interpreter-track `tl.cumsum` overrider required
      `axis` while the tl-module patch intercepts before triton binds
      tl.cumsum's own defaults — bare `tl.cumsum(x)` (radix_sort)
      aborted the dynamic track. FIXED: overrider mirrors the tl
      signature defaults; pinned by
      test_cumsum_overrider_defaults_axis_like_tl_cumsum
      (SWEEP_REPORT §6.9). radix_sort dyn now abstains cleanly.
- [ ] destindex value-aware check: raise (or premise-gate) the interp
      contents-snapshot cap so 2048-element index tensors replay —
      would turn the honest abstain into a values-clean/race verdict.

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
- [x] Tranche 2 — TMA descriptors + mbarrier expect-tx modeling.
      Two protocols, both proved on fresh golden dumps
      (matmul_tma_s3/s1_sm90 from `tl.make_tensor_descriptor`
      sources): PERSISTENT (prologue-initialized rotating barriers:
      the wait at iteration k targets arming (k+b_w) div S of slot
      (b_w+k) mod S with parity ((k+b_w) div S) mod 2 — the parity
      chain is SIMULATED over 4S+4 steps advancing all constant-init
      iter_args in lockstep, and coverage collapses to the linear
      k'+b_e ≤ k+b_w given slot equality) and ONE-SHOT (in-loop
      init: fresh phase-0 barrier per iteration; a copy issued
      before its same-body wait is covered for all same-or-later
      reads). A read holds ALL its preceding wait_barriers as
      guards (one per input buffer); coverage is any-guard. Arming
      validation: expect/copy predicate equality, prologue armings
      = exactly slots 0..b_e-1, expect bytes vs arrivals (under ⇒
      uncovered ⇒ RAW; over ⇒ deadlock ⇒ unsupported). The
      generic→async proxy gate refined: an IMMUTABLE
      (single-assignment) alloc read by wgmma/TMA-store is ordered
      iff a fence_async_shared sits between store and read —
      missing fence is a RAW report (and the in-loop immutable
      store joins the WAR writers: its storage is reused across
      iterations). Storage reuse after dealloc (the stock TMA
      epilogue) is allowed only under a PROVEN drain: epilogue
      pendings=0 / num=0 waits before the reuse plus the TMA
      prefetch-stop predicate d ≥ b_e - b_w (parsed from
      iv < upper - d), checked AFTER the race queries so a racy
      pipeline reports races rather than hiding behind the reuse
      abstention. Mutation battery (all pinned e2e): delete
      wait_barrier / break the parity flip / expect undercount →
      RAW; delete dot-wait → WAR; delete fence → RAW; expect
      overcount / wrong barrier slot → honest deadlock-unsupported;
      weakened prefetch stop → honest drain-unsupported. Sweep:
      TMA matmul proves at stages 1–4 (incl. the one-shot cell that
      first exposed a guard-matching false positive — fixed by the
      any-guard rule), CS4 case study (missing mbarrier phase wait).
      ADVERSARIALLY VERIFIED (2026-07-10, 5 attack agents + independent
      cross-check, 18 agents total): 12 findings confirmed (11
      soundness, 1 precision), ALL FIXED and pinned in
      tests/end_to_end/test_tma_adversarial_regressions.py — the big
      ones: a WAW query now covers async-writer pairs (two byte-exact
      co-armed TMA copies to one buffer used to prove clean; stock
      pipelines still prove because every same-slot writer pair is
      retired by the wait in effect before the later write);
      init_barrier must precede every protocol op on its barrier
      (use-before-init is UB — the one-shot init-after-wait and the
      never-initialized-protocol attacks both proved clean before);
      finite-window chain validation gained a periodicity guard (all
      constants reachable from a phase/slot chain must fit the
      simulation window — an out-of-window constant is exactly what
      defers divergence past the window); the reuse drain now requires
      lower=0/step=1, drains the TMA-store agent via
      async_tma_store_wait {pendings=0}, and uses the b_w-aware
      prologue-arming bound; a loop fence is no longer credited with
      ordering prologue→epilogue pairs (trip 0 skips it); the one-shot
      phase accepts provably-zero loop-carried chains (precision); and
      _simulate_chain advances only dependent iter_args.
- [ ] Tranche 3 — `ttg.warp_specialize`: cross-warp-group
      producer/consumer regions synchronized by count-128 ARRIVE
      barriers (thread-arrival counting, ttng.arrive_barrier,
      per-region phase chains) — a different HB model from
      expect-tx. Scoping artifact landed: matmul_tma_ws_s3_sm90
      golden dump (`tl.range(..., warp_specialize=True)`); stays
      honest-unsupported (pinned: fails closed on the first
      count-128 init_barrier).

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

- [ ] M4 tranche 4 — Blackwell tensor memory (tcgen05): model
      ttng.tmem_alloc/load/store and tc_gen5_mma completion, TMEM
      descriptor ALIASING (the smem allocation-aliasing analog:
      aliased descriptors over one tmem region), and warp-to-chunk
      mappings as layout closed forms. Definition of done: a
      distilled reproduction of the TMEM Membar gap
      (facebookexperimental/triton #1993 — a P store through an
      aliased descriptor vs pending qkT reads, warp-vs-warp inside
      one task, no barrier between them; the full kernel also needs
      the TLX dialect and warp_specialize/tranche 3, so the
      distillation targets plain-dialect tmem aliasing first).
      Verified 2026-07-11: the current track fail-stops on sm100
      TTGIR with "ttng.tmem_alloc is not modeled" — the honest
      refusal, exactly the paper's named boundary. The triton 3.6
      wheel host-compiles sm100 (tl.dot lowers to tc_gen5_mma +
      tmem), so golden dumps need no hardware.

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
- [ ] Category 8b — communication kernels, cross-device half (Keren
      2026-07-11): symmetric-memory / UVM peer-GPU access without
      NCCL. gsan's symmetric-memory tests are the reference litmus
      source: sys-scope `atomic_add` + `atomic_poll` spin on a
      rendezvous'd buffer, then a peer-payload load; the racy twin
      omits the sync. Model extension needed before any of it runs:
      a rank coordinate next to pid (two-copy across ranks; the
      alpha-renaming argument is unchanged), sys scope in the
      mutual-inclusion table (already in the vocabulary), and
      symmetric-buffer identity (peer pointer on rank r = local
      buffer on rank r', same abstract location). `atomic_poll` maps
      onto the await abstraction as-is. Scope as Tier E in the paper
      catalog; single-GPU miniatures (map a "peer" buffer to a
      second region of one device) can precede real multi-GPU.
- [ ] gsan as an external baseline (paper RQ5, alongside racecheck):
      upstream `triton.experimental.gsan` is execution-based
      GLOBAL-memory detection (TritonInstrument pass, vector-clock +
      shadow-memory runtime), i.e. the direct dynamic counterpart of
      our global track. Applicability pass first: which of our 52
      rows it accepts, whether it runs single-GPU, and what its
      per-launch overhead is vs our 34 ms. GPU-gated like racecheck.
- [ ] External-baseline adapters (paper RQ5): GPU-GATED. Two of the
      planned baselines are already covered by the ablation switches
      (no-hb = the overlap checker, no-load-values = the concrete
      replayer); the external ones (compute-sanitizer racecheck,
      thread-level tools) need real hardware and an applicability
      pass first (racecheck covers shared memory; our litmus corpus
      is mostly global).

## Decision points (not tasks)

- PR layout: `race-detector-z3-demo` now carries the plan-doc
  restructure plus S1–S6, the evaluation phases (A–C, mutation, RQ
  instrumentation, T0 stretch), and the docs commits; decide
  whether to merge as one PR, split per step, or split
  detector-core / evaluation-harness before opening against main.
- Next advisor alignment carries: Q5 (M4/sm90 submission scope),
  the landing-figure question, and confirmation of the executed
  contribution-triad reframing.
