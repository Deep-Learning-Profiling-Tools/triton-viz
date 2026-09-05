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


## 3h. Real-kernel corpus growth: aiter_ops (landed 2026-08-28)

113 captured launches from ROCm/aiter's Triton ops (checkout at
AITER_ROOT, commit-pinned b0d56a0; NOT pip-installable on NVIDIA,
loaded through the package stubs of kernels/_aiter_loader.py:
skipped ROCm-requiring inits, synthetic dtypes/chip_info/
torch_guard/jit.core, and a meta-path mirror of the real
backward-compat module redirects). Captured by
evaluation/aiter_capture.py from the 103 op_tests/triton_tests
files (98 succeed; residue: pa_decode x2, conv2d empty, one
fusion, one mxfp4 case), with AMD-only launch kwargs stripped
and unrebuildable AMD-fp8 dtype records filtered. Distinct from
aiter_originals (the 2-row A1 case corpus). Launch validation on
the 4090: 108/113 rows run as plain GPU launches (median 2.1 s);
the 5 failures are sm_89 shared-memory OOM at the captured
configs (record precedes the run), analyzable only by the
GPU-free tracks on this machine. Survey provenance in the paper
repo (TODO.md rq2, baselines/results/aiter_census.json).

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

## 3m. cuTile front-end: first non-Triton DSL (LANDED 2026-07-16)

Record: the detector now analyzes NVIDIA cuTile (cuda.tile) kernels.
Architecture bet paid off exactly as designed — new FRONT-END, zero
core changes: `clients/common/cutile_ir_reader.py` parses the final
CuTile IR text into the SAME AccessGraph/Term algebra as the TTIR
reader; encode_graph, the two-copy solver, tier selector, and the §3c
launch-scoped rung run unchanged.

- Toolchain: cuda-tile 1.5.0 (+[tileiras]) in the project venv; sm89
  works (only fp8/fp4 dtypes are arch-gated). IR captured AT LAUNCH by
  evaluation/tilebench_cutile_capture (patches ct.launch, records then
  runs; engine verify validates the recorded launch) and compiled to
  text in-record — corpus rebuild needs neither cuda-tile nor a GPU.
- Key semantic mappings: tile addressing → index*tile_shape+arange
  affine terms + implicit OOB-clip AS mask terms; pointer_offset +
  tile_atomic_rmw / load_pointer / store_pointer ≡ TTIR raw-pointer
  shapes; python floor-div lowers to c_mod + BOOLEAN-xor sign-fix
  (modeled exactly as (a∧¬b)∨(¬a∧b)); ct.Constant params surface as
  typed_const with python names; array params flatten to
  p_0/base + p_1..p_r shapes + p_{r+1}..p_2r strides (harness binds
  from captured descriptors).
- Sweep (tilebench_cutile, 61 rows): 17 T0 / 19 T1 / 2 T1-launch
  (+grid-fragile — §3c working through the new front-end) / 23 abstain
  (9 nested-loop, 8 indirect-address, 6 control-flow), zero crashes,
  zero races-unclassified. Cross-DSL differential vs the Triton twins:
  30/45 operators agree (incl. identical abstention kinds on the
  data-dependent ones); cuTile AHEAD on 4 (both matmuls prove @T1
  where Triton timed out / Z3-undecided — structured tile indices beat
  flat-pointer arithmetic), behind on 10 (7 multi-pass-loop shapes +
  no interpreter channel), scope-split on 1 (top_k @T1 vs @T1-launch).
- Pins: tests/unit/test_cutile_reader.py (7 — proof AND detection
  directions end-to-end, atomic lowering, bool-xor floor fix, int-xor
  abstention, load_pointer, while-form refusal).

Queued lifts (v2):
- [ ] multi-loop support (7 rows) — sequential + nested loop slots.
- [ ] while-form `loop` / `if` blocks (6 rows) — path conditions.
- [ ] integer xor in addresses (bitonic partner indexing) — bitvector
      side-channel or pattern lift.
- [ ] C2 confirmation: mini tile-op evaluator (29-op numpy-like
      surface) or real-launch replay on the 4090 — restores the
      confirmed/unconfirmed distinction for cuTile race SATs.
- [ ] LLM-generated cuTile kernels (TileBench benchmarks/llm_generated,
      51 @ct.kernel) as a second cuTile corpus — race detection of
      LLM-authored tile kernels ties into the group's pipeline paper.

## 3n. Content-fragile attribute: launch-scoped philosophy for widened evidence (LANDED 2026-07-16; decided (b), Hao)

Provenance: the 2026-07-16 paper-vs-implementation comparison found
the composed dispatcher's short-circuit at `harness.py:494` — when a
static WIDENED report is demoted by replay (reason contains
`race-unconfirmed`), the dispatcher returns abstain BEFORE consulting
the interpreter, even when the interpreter ran clean and holds the
launch-scoped proof (dd_mask_dead: dyn ok(0), terminal
race-unconfirmed). Paper §2 says the dead launch must not be reported
and C1 owes it a proof. The short-circuit is principled (the widened
SAT is an any-contents hazard a launch-scoped proof cannot refute)
but uses the blunt instrument; 3c's proof-plus-attribute pattern is
the elegant one. Decision (b): compose to `proved@interp` and carry
the demoted hazard as a `content_fragile` attribute.

- [x] Dispatcher (`evaluation/harness.py`, `_classify`): in the
      static-unsupported branch, when the reason is the demoted
      widened report AND the dynamic track ran ok:
      n_reports == 0 ⇒ ("race-free", "proved@interp") with
      `content_fragile=True` (today: abstain/race-unconfirmed);
      n_reports > 0 ⇒ ("race", "race@interp") as today (the concrete
      interp reports subsume the widened hazard). When the dynamic
      track did not run or was unsupported ⇒ UNCHANGED
      ("abstain", "race-unconfirmed"): no proof exists, fail closed.
- [x] Attribute plumbing: `verdict_attrs` gains `content_fragile`
      (exactly parallel to `grid_fragile`): evidence = the demoted
      report's site pair + which terms were widened; wording is
      hazard-only ("some memory contents enable an overlap"), never a
      race claim. Soundness note for the docstring: widening only
      enlarges footprints, so the hazard reading is sound from
      widened evidence — the same argument 3c recorded for
      grid_fragile.
- [x] Guardrails (mirror 3c's three): (i) the attribute fires ONLY
      when replay ran faithfully and found no overlap AND the
      interpreter proved clean at the same launch — a demotion in the
      structurally-unconfirmable classes (duplicate-lane, RMW pairs,
      widened same-line, await-bearing) must NOT become a proof;
      (ii) premises compose: the proved@interp carries the
      contents-snapshot premise exactly as dynamic["premises"]
      reports it; (iii) Z3-unknown or replay-declined anywhere ⇒
      today's behavior, fail-closed on the claim.
- [x] Scope: composed dispatcher only; encoders and the two-copy
      solver untouched (the client only RETAINS the refuted hazard as
      last_content_hazard in the faithful-demotion branch — evidence
      plumbing, no decision logic).
- [x] Tests to pin: tests/unit/test_composed_dispatcher.py (6 pins:
      demoted+clean-interp => proof, demoted+interp-reports =>
      race@interp, dyn-absent/failed => fail-closed, generic demotion
      keeps the plain composition, two-run determinism) +
      test_replay_channels extensions (last_content_hazard populated
      on the faithful demotion, client-side attribute stays False,
      capped/no-replay demotion carries NO hazard evidence). Corpus
      level: trb006_dd_mask_dead_no => proved@interp+content-fragile,
      live twin unchanged race-confirmed.
- [x] Scorecard impact VERIFIED on the re-sweep: TN 23 -> 24,
      abstain-unconfirmed 1 -> 0, coverage 55/56, precision = recall
      = 1.0, witness-matched 25/25, ladder audits zero (PASS). Grep of
      every corpus jsonl: exactly TWO rows corpus-wide carried the
      demotion (trb006_dd_mask_dead_no, smoke_dd_mask_dead_no — both
      dyn ok(0), both flipped with the attribute); ZERO real-code
      rows, so the real-code tables are unchanged.
- [x] SWEEP_REPORT §2 updated (terminals + the content-fragile
      paragraph); RESULTS regenerated by the re-sweeps (the +content-
      fragile marker renders next to the terminal, keyed on the
      ATTRIBUTE so a failed-closed demotion stays unmarked).
- [x] Paper linkage (tracked in the paper repo's TODO): fig:ddmask's
      caption and C1 then hold strictly; §4.4 gains the
      content-fragile sentence next to grid-fragile; §6.1's
      "demotion caught the false positive" narrative becomes the
      fragility-attribute narrative (rides the rq1/rq2 realignment).
      DONE 2026-08-30: the §4.4/§6.1/fig:ddmask edits had landed
      2026-07-16 (paper commit 0addcf1); the last gap was the
      result-taxonomy prose itself, which still ended every refuted
      widened report at race-unconfirmed and left the composition
      stated only in §6.1 and the appendix. Closed in the paper's
      round 119 (commit ee5ba3f): the taxonomy subsection now
      states the composition, its guards (capped/unavailable never
      composes; interp race stands as race; no interp adjudication
      fails closed), and the table trigger gains "no interpreter
      proof composes". Test gap noted there for this repo:
      run_one's content_fragile=True stamp has no direct test
      (dispatcher pins cover only _classify).


## 3o. The ladder switch (L0/L1/L2) and the L1 rung: concrete per-instance enumeration (on branch `route1-concrete-enumeration`, 2026-09-04; default L0)

Provenance: the paper repo's abstention analysis (2026-09-04) — the
pinned run abstains on 492/1062 real-code rows, 217 of them in the
`indirect-address x interpreter-unsupported` class (indirect
addressing plus nested loops / pid-dependent control flow: the
destindex, kv_cache_filling, fla varlen families). Both frontends
refuse by construction (the reader has no contents, the one-shot
symbolic capture has no per-instance control flow). Design:
`design-route1-concrete-enumeration.md` (paper repo, Route 1) and the
ladder-switch decision (Hao; `design-route3-multipath-capture.md`
section 4b): ONE ladder-depth configuration, three levels, stamped
everywhere, consulted at exactly one gate.

Machinery (all landed on the branch, 986 tests pass incl. 82 new):

- `triton_viz/clients/race_detector/ladder.py`: `LadderLevel`
  (L0 shipped behavior, L1 = + the concrete rung, L2 = + forked
  capture, future; L2 implies L1), `parse_ladder_level` (strict).
  NOT an environment variable: a constructor parameter on
  `SymbolicRaceDetector` and `CompiledRaceDetector` (the `ablations`
  precedent), stamped into `verdict_attrs.ladder_level` by the
  compiled client, into every harness row (`row["ladder_level"]`),
  and into the results-JSONL header (`ladder_level`); the runner
  writes deeper levels to `<corpus>_L1.jsonl` so the L0 datasets
  (the paper's numbers) can never be overwritten unnoticed.
- `triton_viz/clients/race_detector/concrete_enum.py`: the L1 rung.
  `ConcreteFootprintRecorder` runs EVERY block sequentially under the
  interpreter on per-STORAGE clones (aliased arguments keep aliasing;
  trb009's in-place shift is the pin), records per-operation byte
  intervals with lane multiplicity (duplicate lanes of one plain
  store = the A1 shape; atomics stay one interval per lane so the
  compatible-pair judgment is per exact address and width), carries
  CONCRETE TAINT through every builder op (a generic wrapper over the
  interpreter builder, the tl-level reduce/scan, block-pointer and
  descriptor materialization; `tl.tensor.__bool__`/`__index__` hooked
  through the interpreter's own language patcher so helper re-patches
  keep the hook; loop bounds through the range-wrapper factory), and
  refuses BY NAME: `atomic-return` (an atomic return reaches an
  address, mask, host branch, or loop bound: ticket, last-block,
  atomic-poll spins — the spin refuses at its FIRST poll, no hang),
  `value-source` (a load whose value reaches a footprint position
  overlaps ANY write footprint: the A2 premise, extended to branches
  and bounds), `instance-ceiling` (`ENUM_MAX_INSTANCES = 65536`,
  refused before executing), `no-grid`, `no-contents`, `scope`,
  `timeout`, `interpreter-error`. `analyze` mirrors
  `conflicting_access_modes` byte-for-byte: overlap + at least one
  writer; atomic-atomic exempt iff same width, same start, no cta
  scope across instances; plain-vs-atomic races; program order within
  an instance; the premise violation refuses the whole launch before
  any race is reported. Witnesses are translated back to the caller's
  tensors and carry the byte range. Unknown-provenance values
  (constructed outside the builder) taint conservatively.
- Harness (`evaluation/harness.py`): `_enum_track` (spin pre-gate from
  the static reader's `spin-shape`/`assumes_termination`, fresh
  `make_args`, watchdog = the remaining row budget capped at 150 s),
  the ONE gate in `run_one` (`ladder_level >= L1 and verdict ==
  "abstain"`), `_classify(static, dynamic, enum)` (the L1 leg fires
  only on an abstention: `proved@enum` / `race@enum`, analyzed-launch
  extent, `content_fragile=True`, `proved_scope=this-params-this-grid`,
  `race_evidence=concrete`), `--ladder-level` on harness and runner.
  `report.py` reads enum witnesses and audits `race@enum` on
  race-free labels as `enum_disagreements` (surfaced, like interp);
  `concretization_map.py` gains the bottom y-row "nothing (every
  instance enumerated)" with `proved@enum`/`race@enum` at (3, 0).
- Tests: `tests/end_to_end/test_concrete_enum.py` (33: scatter pair,
  A1 lanes, program order, plain-vs-atomic, compatible/cta/torn
  atomics, mixed widths, plain RW reported, value-source through
  address and mask, ticket/last-block/loop-bound/spin refusals,
  pid-branch + nested loops decided, data-dependent trips, masks,
  block pointers, single counting of unmasked accesses, ceiling,
  callable grid, patch cleanup, aliasing), `tests/unit/
  test_concrete_enum_analysis.py` (19 synthetic pins of the
  predicate and the premise), `tests/unit/test_ladder_level.py` (30:
  parsing, constructors, attrs stamp, header, `_classify` legs).

Verification so far (2026-09-04, this machine):

- TritonRaceBench at L0 vs L1 (`--jobs 4`): 61 rows, ZERO flips (no
  benchmark row abstains at L0, so the gate never fires; the level
  stamps verify).
- Cross-validation (design section 7.2) on the 51 benchmark rows the
  interpreter decides (tritonracebench, golden_smoke, rmw_sync,
  await_sync): 35 AGREE, 16 DISQUALIFIED by name (`atomic-return`:
  rows decided through the counting axiom / RMW-return modeling —
  lbd, splitk, amax, acq/rel families — by design), 0 DISAGREE (the
  one disagreement found, trb009's aliased in-place shift, was the
  per-argument clone bug, fixed by per-storage cloning and pinned).
- Real-code rows at L1 (tritonbench_g): tb_destindex_copy race@enum
  (32768 instances, 45.7 s, 1.35 ms/instance, duplicate randint
  destinations at lines 45/46 — the Leads-30 reading);
  tb_destindex_copy_kv1 race@enum (65.9 s, 1.97 ms/instance; timed
  out at the first 60 s watchdog, hence the row-budget watchdog);
  tb_quantize_copy_kv proved@enum (8192 instances, 24.6 s);
  tb_context_attn_mistral proved@enum (192 instances, 5.8 s, 28 ms/
  instance); tb_kv_cache_filling race@enum (10 instances, 0.2 s; the
  captured all-zero BlockOffsets make two instances fill one block);
  kv_cache_copy / kcache_copy_triton stay proved@interp (the gate
  does not fire on decided rows).

Addendum (2026-09-04, after the first change-surface stretch, 52
aiter_ops rows at L1, jobs=1: 36 proved@enum, 16 residual):

- Spin pre-gate narrowed: the harness refuses before executing only
  on `assumes_termination` (a reader-recognized await); the reader's
  `spin-shape` kind also covers carried-value `scf.while` iteration
  (SWEEP_REPORT §7) and had cost three rows (two now proved@enum,
  one refused by the rung's own taint: an atomic poll in a host
  branch, the correct reading).
- The A2 premise is cross-instance for this rung, with taint through
  memory: a store records the taint of its value, a later
  same-instance load of those bytes inherits it (a relayed atomic
  return refuses `atomic-return ... through memory`; a relayed loaded
  value makes the original load a value source, checked in turn);
  same-instance in-place updates are admitted (four rows: the
  causal-conv state updates and the fused KV-cache fusions, now
  proved@enum). Soundness argument in the design doc section 2.4.
- Projected-cost refusal (Hao): first instance excluded, 5 s grace,
  running mean x remaining instances + elapsed > budget refuses by
  name (`projected-cost`, `projected_cost_refusal` is pure and
  pinned); the refusal fires only beyond TWICE the budget (Hao), so
  a projection between one and two budgets keeps running with the
  watchdog as the bound. The four chunked/paged-prefill rows (10240
  instances at 100 to 114 ms) refuse after 5.1 s instead of 150 s
  (projected 1021 to 1164 s); chunk_delta_attn intra_token_parallel
  (2048 instances at 87 ms, projected 178 s, about 186 s needed)
  keeps running under the factor but hit the watchdog at 150 s, so
  the per-row budget became LEVEL-DEPENDENT (Hao): 180 s at L0 (the
  paper's protocol, untouched), 200 s at L1+
  (`runner.row_timeout_s`, stamped into the header as
  `row_timeout_s`); the rung's watchdog is that budget minus the
  symbolic tracks' time and a 10 s margin. Verified through the real
  harness path: that row decides proved@enum (2048 instances, enum
  188.9 s against a 189.6 s watchdog, row wall 191 s), a 0.7 s margin
  that says the budget edge is a real class, not a one-off.
- Precision bug fixed: the interpreter's synthesized all-True mask
  for unmasked loads/stores carried no taint tag and counted as
  unknown provenance, so every unmasked load after an atomic
  inherited the atomic marker (spurious `atomic-return` refusals);
  the builder wrapper now tags it empty before the masked op runs.
- Re-verified: 972 tests pass (94 in the Route 1 files);
  cross-validation unchanged (35 agree, 16 disqualified, 0
  disagree). Residual of the 52-row stretch after the fixes: 10
  rows = 5 projected-cost, 1 atomic-return, 4 interpreter-error (the
  interpreter itself cannot run those kernels: `_semantic` helper
  calls, an `Assume failed` on rebuilt inputs, a `to_tensor` on None;
  all reproduced with the plain C2 replay recorder).

Change-surface run DONE (2026-09-04, all 492 pinned-abstain rows at
L1, jobs=1, 200 s per row, 1.42 h; commit 5ba8b6a; report:
`evaluation/CHANGE_SURFACE_L1.md`, dataset
`evaluation/results/change_surface_L1.jsonl`):

- 407 decided (391 proved@enum, 16 race@enum), 1 decided by commits
  since the pin, 84 residual = 7.9% of 1062 (from 46.3%). Residual by
  kind: 29 interpreter-error, 23 cuTile, 12 row-crash, 7
  atomic-return, 6 projected-cost, 4 instance-ceiling, 3 row-timeout.
- The 16 race@enum rows, triaged: 11 capture-rebuild artifacts
  (index tensors above the 8192-element snapshot cap rebuilt at
  random, sometimes next to a snapshotted tensor derived from the
  real one: masked_select's part_sums vs its mask), 2 A8-class
  out-of-bounds (iplr varlen bwd; chunk_gla merge whose A is captured
  8x too small), 3 model races with identical values (unique_dup's
  duplicate lanes, ttt layer_norm_bwd's overlapping dx tiles x2).
  None counted (Leads-30 discipline). The design's 7.3 expectation
  (permutation-scatter rows prove clean) failed for the capture
  reason, not a rung reason.
- The 12 crashes are deterministic, all inside the rung, all
  SIGSEGV/SIGABRT from out-of-bounds stores on raw host pointers
  (L0 abstains cleanly in 3 s); subprocess isolation contained them,
  but two rows emitted output before dying, so an OOB kernel's
  verdict is not trustworthy. The in-bounds premise is enforced by
  fail-stop on the symbolic frontends and NOT yet by the rung.
- [x] In-bounds premise enforced in the rung (Hao, 2026-09-05):
      `ConcreteFootprintRecorder(bounds=...)` checks every access's
      active lanes against the cloned storages' spans in the
      before-callback and refuses `out-of-bounds` by name before the
      interpreter dereferences (masked-off lanes exempt; the storage,
      not the view, is the bound; ~4 us per access, 1-4% end to end).
      All 14 affected rows (12 crashes, 2 OOB race@enum) re-run as
      named refusals in 2.6-6.5 s with no signal; cross-validation
      unchanged (35/16/0); 5 new kernel-level pins; report addendum in
      `evaluation/CHANGE_SURFACE_L1.md`. Restated: 391 proved@enum,
      14 race@enum, 86 residual (8.1%).
- [x] The same check in the C2/C3 replay channel (Hao, 2026-09-05):
      `bounds.StorageBounds` is shared; `run_replay` builds the spans
      from its clones and `FootprintRecorder` checks every access
      before the interpreter executes it; an out-of-bounds replay
      declines as `unavailable: replay failed: out-of-bounds ...`
      (the existing fail-closed path, the report stays
      races-unclassified) instead of corrupting the process. Cost:
      ~4 us per access on the ~2% of rows that reach replay (24 of
      1062 in the pin), under 1 ms per row. Two pins.
- [x] Runner process reuse, DEBUGGING ONLY (Hao, 2026-09-05:
      `--debug-reuse-workers`; FORBIDDEN for a pinned rerun or any
      quoted number, because the paper's per-row wall times are
      per-row subprocess walls). A debugging dataset is unmistakable:
      the `_debug-reuse` file suffix, `worker_reuse.debugging_only` in
      the header, a stderr banner at start, `report.py` labelling it,
      `headline.py` / `concretization_map.py` skipping it, and
      `runner.assert_protocol_dataset` (for the pinned driver to call
      on every input) refusing it. Mechanism: `harness --serve` is a worker that
      runs rows requested on stdin (corpus loaded once), `runner._Worker`
      drives it under the per-row budget with select (a silent worker
      is killed: `timeout`; a dead one: `crash` with the stderr tail;
      both respawned), recycles workers every `--worker-rows` (50) rows
      or above 8 GB RSS, and stamps `worker_reuse` into the header
      (wall_s then excludes process start-up). Row independence: the
      worker snapshots the interpreter-patched language state before
      its first row and restores it after every row, logging what
      leaked. The probe of 50 rows in one process found the leak that
      breaks the next real compile (core/trace.py's warmup-only note):
      the L1 recorder's cleanup, running AFTER the trace's own restore
      on the mid-kernel refusal path, re-installed the interpreter's
      reduce/scan and the builder's PatchOps it had captured; fixed
      (cleanup restores only attributes that still hold its wrapper)
      and pinned. Remaining known leak: tl.core.tensor.__repr__ from the
      symbolic frontend (harmless; the worker restores it). Saves the
      2-3 s per-row start-up (~30 of the 85 min of the 492-row run).
      Pins: served rows equal subprocess rows on golden_smoke, crash
      and hang fault injection, recycling and the header stamp.
- rope_fwd_3d budget regression (81.9 s in the first stretch, >200 s
  in the full run): DIAGNOSED AND FIXED. The memory-taint rewrite of
  the premise check scanned the whole interval buffer per
  value-source load (quadratic; 35520 loads x 97.6M intervals) and
  ran outside the watchdog. Now bisection over the op-sorted buffer,
  and the analysis phase runs under the remaining budget (a slow
  sweep ends in a named `timeout:` refusal, never a row-level
  timeout); pinned by a 6000-load scaling test. The row decides
  proved@enum in 157 s through the harness (85 s run, 68 s sweep).
- [ ] Strided footprints: rope's accesses do not coalesce (916
      intervals per op, 97.6M intervals, 2.3 GB for 11840 instances;
      a 65536-instance row of this shape needs ~12 GB and the sweep
      ~6 min). Design sketch: per-op footprint = bounding box +
      uniform-stride run (base, stride, count, segment length) with
      raw intervals as the fallback; sweep boxes; same-stride runs
      compare as rectangles (row range x column residue) in O(1);
      materialize lanes only where boxes of distinct instances
      overlap; atomic compatibility by lane alignment. A rewrite of
      the soundness-critical sweep: do it as its own step with the
      synthetic pins, the kernel-level tests, and the
      cross-validation rerun.

## 3p. Corpus capture: every int/bool tensor value-snapshotted (branch `route1-concrete-enumeration`, 2026-09-04; recapture awaits Hao's go)

Decision (Hao, 2026-09-04): capture and STORE the real values of
every integer and bool tensor (floats stay by-descriptor), replacing
the 8192-element inline cap that made the L1 rung's 11
capture-artifact rows. Landed on the branch, backward compatible
(every existing spec rebuilds unchanged, verified over all 1060 rows):

- `capture_common.ValueStore`: content-addressed (SHA-256 of the raw
  bytes) compressed `.npz` sidecar, `<corpus>_values.npz` beside the
  specs JSON (gitignored: ~200 MB raw across the corpora, tens of MB
  compressed; the hashes live in the JSON, so integrity is checked on
  read; git LFS is the alternative if Hao wants it tracked). Small
  int/bool snapshots (<= 8192) stay INLINE as before; larger ones
  carry `values_ref`. A referenced-but-missing snapshot is a HARD
  error (`MissingValueSnapshot`), never a random rebuild. A capture
  without a store marks `values_dropped` instead of pretending.
- `LaunchRecorder` owns a store; the per-case child processes write
  it beside their JSON (`write_case_result`), `run_case_capture` and
  `tritonbench_capture` merge the children's stores, prune to the
  referenced hashes and save the corpus sidecar; both loaders
  (`kernels/_captured.py`, `kernels/tritonbench_g.py`) pass
  `ValueStore.beside(specs)` to `make_args_fn`. Fingerprints include
  the reference, so dedup stays content-based.
- Tests: `tests/unit/test_capture_values.py` (10).
- [x] RECAPTURED all 8 Triton corpora (2026-09-05, this machine, RTX
      4090, 51 min end to end; every installed upstream matched the
      recorded pin: aiter b0d56a0, fla 0.5.1, flaggems 1051e56, torchao
      bfbc842, tritonbench_meta 1edaf3e, Tilebench 224ec81, FlagAttention
      41fc31d, TritonBench_G_v1 603e28a). Every row rebuilds and every
      sidecar array passes its hash; no `values_dropped`; no old row
      lost. Rows: flagattn 28 (specs byte-identical), torchao 67,
      tilebench 56, tritonbench_meta 41, fla 378, aiter_ops 113 (the
      same 5 failing cases) unchanged; flaggems 82 -> 84 and
      tritonbench_g 202 -> 224 gained rows that capture-side changes
      landed AFTER the old capture now admit (non-contiguous args,
      dtype constexprs, two files no longer failing) -- not the
      snapshot change; the pinned rerun's row set is therefore +24
      against the fb91fc0 pin and must align by name. Sidecars
      (`<corpus>_values.npz`, 7 files, 73.8 MB: tritonbench_g 27.9,
      tritonbench_meta 26.2, tilebench 17.8, aiter_ops 1.3, flaggems
      0.4, torchao 0.1, fla <0.1; flagattn needs none; 52 arrays in
      all) are gitignored, present in the branch worktree and the
      main checkout, and backed up to `~/workspace/triton-viz-values-
      backup/`; storage decision (LFS vs out-of-tree) pending Hao.
- [ ] Pinned rerun at L0 AND L1 on the new contents (Hao: together,
      after the recapture). Contents change every analyzed-launch
      verdict's basis, so the paper's 66 proved@interp and the L1
      numbers move; a fresh pin.
- Note: destindex-class rows (upstream tests that draw duplicate
  indices with randint, casebook A6) will still say race@enum on the
  real snapshot; that is the honest analyzed-launch reading of the
  upstream test's inputs.

Open (blocking any paper use of L1; default stays L0 until done):

- [ ] Change-surface diff: every currently-abstaining real-code row
      (the 492) at L1 vs the pinned L0 run, jobs=1; classify the
      residual by refusal kind (the design's residual floor: 23
      cuTile + 9 spin + 4 over the ceiling = 36 rows, plus the
      classifier-pinned atomic-return / value-source classes).
- [ ] Fresh pinned rerun at L1 (a separate stamped dataset next to
      the L0 pin), then the selective-pricing check: every L0-decided
      row verdict-identical and wall-time-stable. Time (measured
      2026-09-04/05): the rung adds 34.6 min over the 492 L0-abstain
      rows; the 492 rows alone take 85 min with one subprocess per
      row, about 55 min with `--reuse-workers` (development runs
      only: the pinned protocol keeps per-row subprocesses so wall
      times stay comparable with the L0 pin; the paper repo's
      `pre-submission/pinned-rerun.md` section 2 carries the full
      estimate, about 4 h for an L1 sitting).
- [ ] Docs when the rerun lands: SWEEP_REPORT §2/§3/§7 (terminals,
      counting by scope, the queued-lift ledger), the plan's §I.1
      five-state table and §I.2 reachable-regions table (a "nothing
      symbolic" row), address_position_lifting_spec §0/§5.3/§6, the
      "interpreter CANNOT rescue these" sentence in §3e above, the
      paper's §4.5/§6.3 and the race casebook (the destindex and
      kv_cache_filling race@enum rows are capture-content readings,
      Leads-30 discipline: none counted).
- [ ] Route 3 (L2) lands its fork gate at the per-instance
      control-flow refusal site and hands path-ceiling rows to
      `_enum_track` (the same invocation).

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

## 9. A2 gate — atomic-ordering barrier coverage (LANDED 2026-08-27)

Shipped on Hao's request in one day, spec-first
(`impl-spec-a2-gate.md` in the paper repo; as-built record in
`race_detector_static_hybrid_plan.md` §8.1): a third verdict
surface (`last_lowering_status`) checking that the lowering
emitted the CTA barriers non-relaxed atomic semantics require
(the triton PR #10816 rule; the paper casebook's A2 class).
Structural coverage over the captured PTX, no SMT; fail-closed
named refusals. Headline results: the pre-fix/post-fix pair
(7aab98ee violation, c57bbbd8 verified) flips exactly on the fix;
the corpus pin triton 3.6.0 itself predates the fix and reports
violation (A2 live in the pinned toolchain; benchmark validity
unaffected, its rows use no shared memory). Paper consequences
deliberately deferred: A2's no-detection-claim discipline stands
until the Keren revisit (paper TODO.md `baselines`/compiled-mode
notes).

Open v2 items: AMD (`asm["amdgcn"]`), clusters, `atomic_poll`
rendezvous matching, full Membar aliasing verification, the
in-compiler MLIR SMT placement.

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
