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
trip count). 227 race-detector tests pass. The checked-item
histories live in this file's git log and the commit messages.

What remains, ordered by paper impact:

## 1. M5 — shared-track evaluation (the only item still blocking paper placeholders)

DESCOPED 2026-07-10 per the advisor: sell the idea with z3py; the
per-query SMT-LIB2 emission / interchange-format deliverable is
dropped (z3's native to_smt2 covers any future need). Remaining:

- [ ] Evaluation sweep: tutorials × `num_stages ∈ {1..4}` × {sm80,
      sm90}: proofs, solve times, mutation-detection matrix. The
      sm80 half does NOT depend on M4 and can start now; it feeds
      the paper's "Compiled-Mode Evaluation" and §7 pipeline
      placeholders.
- [ ] Case studies from historical pipeliner bugs (missing
      `async_wait`; insufficient `num_stages` letting a producer
      overwrite a buffer still being read).

## 2. Benchmark corpus growth (feeds the paper's rq1 tag)

- [ ] Author the four planned litmus variants as TritonRaceBench
      pairs: partially overlapping masks (racy + race-free);
      release-only and acquire-only guarded producer/consumer (both
      expected racy: one side of the sw edge missing);
      acquire-on-failure positive case (a consumer CAS that FAILS —
      cmp never matches — but reads the released value and guards
      on it: expected no race, exercising the
      reader-success-independence of rf-val); an oversized (>K)
      flag variant of the guarded idiom (expected: conservative
      race report, demonstrating the over-report direction of the
      monotonicity lemma).
- [ ] cta-scope atomic-pair litmus (lands together with item 3).

## 3. Moral-strength conflict refinement (feeds the paper's memory-model tag)

- [ ] Tile IR alignment: the conflict predicate currently exempts
      ALL atomic pairs, while Tile IR's moral strength classifies
      scope-mismatched atomic pairs as racy. Implement the
      moral-strength check in the solver's conflict predicate
      (`two_copy_symbolic_hb_solver.py`): a conflicting atomic pair
      is exempt only when the scopes are inclusive (for cross-CTA
      pairs: both in {gpu, sys}). Adds reports only, preserving the
      over-report direction. The paper then updates Def. conflict
      and drops its divergence caveat.

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
