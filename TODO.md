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
      the rotation is already depth 1). sm90 column stays gated on M4
      (advisor Q5).
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

## 4. M4 — sm90/Hopper (GATED on Q5 with the advisor; align before starting)

- [ ] `ttng.warp_group_dot_wait {pendings}` agent,
      `fence_async_shared`, nvmma layouts (formula already
      verified); then TMA descriptors + mbarrier phase/arrive-count
      modeling + `ttg.warp_specialize`. Needs fresh golden dumps
      from descriptor-based kernels (`tl.make_tensor_descriptor`
      sources — block-ptr kernels get rewritten to plain pointers).

## 5. Optional: results landing figure (GATED on advisor alignment)

- [ ] The 2-D concretization map of plan §I.2, exported from the
      results JSONL (each row's terminal state + front-end
      determines its point), as an evaluation-section figure.
      Formerly "the core figure"; demoted 2026-07-09 per the
      contribution-triad feedback — the symbolic/concrete axis is
      not the paper's headline and the benchmark table already
      carries the data. Whether it enters the paper at all is
      pending the next advisor alignment. Figure script separate
      from the harness.

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

- [ ] C2 footprint precision: key replay footprints per access SITE
      (e.g. by user source line, matching the TTIR loc) instead of
      (tensor, kind) — the current ambiguity gate declines to
      classify reports on tensors with multiple same-kind access
      sites; site-level keying would recover those confirmations.
- [ ] Interpreter × numpy 2.x: `range(0, n_scalar_arg, BLOCK)` in a
      kernel raises `TypeError` under the interpreter (triton wraps
      scalars as shape-(1,) arrays; numpy 2 refuses `__index__` on
      them), so C2/C3 replay degrades to `unavailable` for
      scalar-bound loop kernels. Sound but loses coverage;
      upstream-shaped fix or a scalar unwrap shim in the replay
      path.

## 8. Repo hygiene: pre-existing test-isolation bugs (full-suite only; the race-detector suites are unaffected)

- [ ] `tests/unit/test_multithreading.py` sets TRITON_INTERPRET=1
      at module level, poisoning the jit kernels of any module
      imported after it during collection (14 replay-channel tests
      fail in full-suite runs); fixture-scope the env var.
- [ ] `triton_viz/visualizer/draw.py` raises the process recursion
      limit at import, defeating
      `test_deep_term_chain_never_escapes_finalize`; move the bump
      out of import.
- [ ] Five wrapper/CLI test failures are environmental
      (FileNotFoundError on the console script); diagnose or mark.

## Decision points (not tasks)

- PR layout: `race-detector-z3-demo` now carries the plan-doc
  restructure plus S1–S6, the evaluation phases (A–C, mutation, RQ
  instrumentation, T0 stretch), and the docs commits; decide
  whether to merge as one PR, split per step, or split
  detector-core / evaluation-harness before opening against main.
- Next advisor alignment carries: Q5 (M4/sm90 submission scope),
  the landing-figure question, and confirmation of the executed
  contribution-triad reframing.
