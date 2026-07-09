# Race Detector — Remaining Work

Companion to `race_detector_static_hybrid_plan.md` (Part III S1–S4 are landed:
shared TTIR reader, scf.if path conditions, per-term DataDep policy, the T1
global-memory track, the T0/T1 tier selector, and the C2/C3 channels — all five
terminal states are materialized). What remains:

## 1. S5 — Evaluation (the paper's data; plan Part III S5)

- [ ] Evaluation harness: run the triton tutorials + a real kernel corpus (e.g.
      a TritonBench subset) through `CompiledRaceDetector`; record for each
      kernel the terminal state — `proved@T0` / `proved@T1` / `race-confirmed`
      / `race-unconfirmed` / `unsupported` — plus the `unsupported` kind bucket
      (`UnsupportedTTIR.kind` taxonomy) and wall-clock. The provenance /
      confirmation / kind surfaces all exist; the harness is collection and
      aggregation.
- [ ] Headline numbers for the paper:
  - kernels reaching `proved@T0` (the "any input, any grid along read axes"
    claim neither the dynamic mode nor T1 can make);
  - kernels the dynamic mode marks unsupported for pid-dependent branches that
    now get a static verdict (S2's acceptance criterion, quantified);
  - the `unsupported` kind distribution (guides where the next modeling
    investment pays).
- [ ] The core figure: the 2-D concretization map (axis 1: what is concretized
      — nothing / scalar params / memory contents / paths; axis 2: what stays
      symbolic), every benchmark kernel plotted on it (plan §I.2).

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
