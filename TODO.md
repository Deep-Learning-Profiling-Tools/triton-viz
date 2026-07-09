# Race Detector — Remaining Work

Companion to `race_detector_static_hybrid_plan.md` (Part III S1–S4 are landed:
shared TTIR reader, scf.if path conditions, per-term DataDep policy, the T1
global-memory track, the T0/T1 tier selector, and the C2/C3 channels — all five
terminal states are materialized. S6 — RMW-return modeling (spec part B) and
the await abstraction (spec C1) — is landed too: observation symbols with
rf/coherence justification, RMW immediacy, reads-through release sequences,
the guarded counting axiom, `tt.get_num_programs` modeling, the scf.while
await shape with termination premises, and the `rmw_sync` / `await_sync`
litmus corpora, all at precision/recall 1.0). What remains:

## 1. S5 — Evaluation (the paper's data; plan Part III S5, revised)

Protocol modeled on DataRaceBench / the LLOV three-outcome scoring; full design
in the plan doc's S5 section. Departures from DRB (paper differentiators):
witness-level scoring via labeled access pairs, a deterministic one-run
protocol, the proof-strength (provenance) dimension DRB's binary "no" cannot
express, and SCOPED ground truth — labels attach to (kernel, launch-params),
so parameterized pairs derive a kernel-level "∃ racy input" truth that audits
the claim ladder itself: proved@T0 against a premise-compatible yes-launch is
`ladder-unsound` (a severity class above FP), race-confirmed on a no-launch is
`replay-unsound`; both must be zero.

Build order:

- [x] (1, ~½ day) Harness skeleton: `evaluation/{kernels/,harness.py,runner.py,
      report.py}` — landed (golden_smoke corpus, 7 kernels, one per terminal
      state; per-spec subprocess + timeout; dynamic + C3 columns; now also the
      `assumes_termination` row field and a SIGALRM watchdog on the dynamic
      phase for spin kernels).
- [x] (2, ~1 day) Phase A — "TritonRaceBench" landed
      (`evaluation/kernels/tritonracebench.py`, run with
      `uv run python -m evaluation.runner --corpus tritonracebench`):
      18 patterns / 40 rows — 8 new micro pairs (pid-stride, fixed-range,
      tail-boundary mask-vs-clamp, atomic-vs-plain accum, pid-branch,
      loop-carried, aliased in-place, indirect scatter, nested loop) plus the
      golden_smoke parameterized rows and the S6 `rmw_sync`/`await_sync`
      corpora folded in under stable `trbNNN_` names. Report upgrades:
      witness-level scoring (race_pair needles resolved to source lines at
      harness time; subset matching against reported witnesses), per-pattern
      table, and the ladder audit grouped by (kernel, constexprs)
      specialization with the aliased-launch exemption. First full numbers:
      precision = recall = 1.0, coverage 34/40 (all 6 abstentions at
      documented boundaries: indirect ×3, nested-loop ×2, dd-mask
      race-unconfirmed ×1), witness-matched 16/16, ladder audit PASS
      (ladder-unsound = replay-unsound = 0), all seven terminal buckets
      populated (proved@T0=7, T1=5, T1+assumes-termination=3,
      race-confirmed=8, race-unconfirmed=1, races-unclassified=11,
      unsupported=5). C3 now reports replay-failure as channel-unavailable
      rather than a fake mismatch (numpy-2 scalar-bound loops).
- [ ] (3, ~1 day) Phase B — triton tutorials (vendored for triton 3.6,
      hand-written LaunchSpecs, ~10–12 kernels; autotuned kernels: take `.fn`,
      pin one config).
- [ ] (4, ~1–2 days) Mutation sensitivity mode (every PROVED kernel: mutate the
      TTIR pid-stride constant, assert the verdict flips — proofs are not
      vacuous) + Phase C — real library (liger-kernel or TritonBench subset,
      20+ kernels; `unsupported` dominating is itself the data).
- [ ] Headline numbers for the paper:
  - kernels reaching `proved@T0` (the "any scalar params" claim neither the
    dynamic mode nor T1 can make);
  - kernels the dynamic mode marks unsupported for pid-dependent branches that
    now get a static verdict (S2's acceptance criterion, quantified);
  - the `unsupported` kind distribution (guides where the next modeling
    investment pays).
- [ ] The core figure: the 2-D concretization map (plan §I.2), exported from
      the results JSONL (each row's terminal state + front-end determines its
      point); figure script separate from the harness.

## 2. S5 — T0 stretch (off the critical path; interleave with evaluation)

- [ ] Symbolic loop bounds at T0: `lower ≤ i < upper` plus a step-divisibility
      constraint instead of requiring concrete bounds; accept that
      `pid × sym_stride`-style nonlinearity yields Z3 `unknown` → the kernel
      simply lands on T1 per the ladder. Whatever reaches T0 feeds the paper's
      "upper bound" section.

## 3. Track 1 — M4/M5 (shared-memory track; plan Part II §7)

- [ ] M4 — sm90/Hopper: `ttng.warp_group_dot_wait {pendings}` agent,
      `fence_async_shared`, nvmma layouts (formula already verified); then TMA
      descriptors + mbarrier phase/arrive-count modeling +
      `ttg.warp_specialize`. Needs fresh golden dumps from descriptor-based
      kernels (`tl.make_tensor_descriptor` sources — block-ptr kernels get
      rewritten to plain pointers).
- [ ] M5 — paper artifacts: per-query SMT-LIB2 emission with a metadata header
      (event ids, locs) as the interchange format; evaluation sweep (tutorials
      × `num_stages` × {sm80, sm90}: proofs, solve times, mutation-detection
      matrix); case studies from historical pipeliner bugs.

## S6 stretch items (require B + C1 together; not part of either's DoD)

- [ ] Ticket lock: needs the bounded reads-through chain OVER unmodeled grid
      instances beyond the counting axiom's single-record guard (two RMW
      records — next_ticket and now_serving — interact).
- [ ] Looped work-queue fetch: RMW inside scf.for needs per-iteration
      observation symbols (one var per iteration, or an uninterpreted
      function of the loop index) before the counting axiom can extend.
- [ ] pingpong_phase (await nested in scf.for with expected = f(LoopVar)):
      parses and encodes today, but the awaited atomic keeps
      old_value=None inside loops (no rf), so it lands on reports, not
      proofs.

## Refinements noted during verification (small, non-blocking)

- [ ] C2 footprint precision: key replay footprints per access SITE (e.g. by
      user source line, matching the TTIR loc) instead of (tensor, kind) — the
      current ambiguity gate declines to classify reports on tensors with
      multiple same-kind access sites; site-level keying would recover those
      confirmations.
- [ ] Interpreter × numpy 2.x: `range(0, n_scalar_arg, BLOCK)` in a kernel
      raises `TypeError` under the interpreter (triton wraps scalars as
      shape-(1,) arrays; numpy 2 refuses `__index__` on them), so C2/C3 replay
      degrades to `unavailable` for scalar-bound loop kernels. Sound but loses
      coverage; upstream-shaped fix or a scalar unwrap shim in the replay path.

## Decision points (not tasks)

- PR layout: `race-detector-z3-demo` carries the plan-doc restructure plus
  S1–S4 as seven commits; decide whether to merge as one PR or split per step
  before opening against main.
