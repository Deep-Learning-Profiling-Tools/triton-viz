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
      populated. CURRENT reproducible distribution (after the 5ad7a72
      soundness fix narrowed trb015 from T0 to T1; independently verified
      2026-07-09, two terminal-identical runs): proved@T0=6, T1=6,
      T1+assumes-termination=3, race-confirmed=9, race-unconfirmed=1,
      races-unclassified=10, unsupported=5. (The original commit
      message's 8/11 confirmed/unclassified split was a stale pre-final
      count.) C3 now reports replay-failure as channel-unavailable
      rather than a fake mismatch (numpy-2 scalar-bound loops).
- [x] (3, ~1 day) Phase B — landed (`evaluation/kernels/tutorials.py`,
      `--corpus tutorials`): triton 3.6 tutorials 01/02/03/04/05/07 vendored
      verbatim (autotune stripped, one config pinned per spec; 8 kernels,
      9 rows). 5/9 proved — including tut05's layer-norm backward LOCK
      kernel (`proved@T1+assumes-termination`, ~45 s: the await abstraction
      + awaited-CAS machinery on real tutorial code) and dropout (philox in
      value position doesn't block the proof, proved@T0). 4 abstentions at
      documented boundaries: persistent grid-stride loop (02), grouped-
      swizzle //-% arithmetic hits the new T1 Z3 timeout → deterministic
      `unsupported (solver: ...)` (03 ×2), multiple sequential loops (05
      fwd).
- [x] (4, ~1–2 days) Mutation mode + Phase C — landed.
      Mutation (`--mutate`): three TTIR mutants per proved row — pid-pin
      (per-pid disjointness), sem-relax (synchronization), atomic-to-store
      (atomicity) — solver-only re-verdicts; report classes flip /
      degraded / SURVIVOR. Across all corpora: 35/37 proofs flip, 1
      degraded (work-queue: atomic→store lands unsupported — the proof
      hinged on atomicity), 1 survivor (bounded n=0: a genuinely dead
      launch).
      Phase C (`evaluation/kernels/liger.py`, needs
      `uv pip install liger-kernel`): 23 kernels across 15 liger ops —
      **17/23 proved@T1, all 17 mutation-validated**; 5 abstentions
      (pid-slab loop bounds ×2, nested loops ×2, cf.cond_br early-return
      ×1), 1 compile-error (liger 0.8 `tl.float32(...)` call vs triton
      3.6 — version skew is sweep data). The sweep also hardened the
      reader (bare `cacheModifier = cs` attribute suffixes on load/store)
      and the client's synthetic-launch binding (mid-signature constexpr
      kwargs no longer shift positional capture).
- [x] Headline numbers (RQ2) — landed: `evaluation/headline.py` aggregates
      the results JSONLs. Coverage corpus (tutorials + liger, 32 rows):
      proved@T0 = 3, proved@T1 = 19 (1 conditional on termination), static
      verdict where the dynamic mode abstains = 10 (all corpora: 34),
      unsupported kinds led by nested-loop / solver-timeout / other.
- [x] RQ3 scaling sweeps — landed: `evaluation/scaling.py` (synthesized
      TTIR, single-dimension sweeps; per-query stats via the solver's new
      `query_stats`; writes `results/SCALING.md`). All five predicted
      shapes CONFIRMED: grid (4→2^20), tile (32→2048) and trip count
      (2→512) flat within noise; site count m: queries 6→20→72→272
      (~m^2); atomic count c: base constraints 18→84→504→3504 (~c^3, the
      coherence writer×reader×interposer triple); zero timeouts.
- [x] RQ5 ablation switches — landed: solver `ablations=("hb"|"coherence")`
      (client + dynamic detector plumb-through), dynamic
      `ablations=("load-values",)` single-observation mode;
      `evaluation/ablation.py` writes `results/ABLATION.md`. Flip matrix
      over the litmus corpora (25 rows): 7 rows flip — no-hb kills exactly
      the ordering proofs (lbd, splitk, pc_wait, mutex, lookback) while
      footprint proofs survive; no-coherence kills exactly the
      counting/immediacy proofs (work-queue, mutex, lbd) while pure
      sw proofs (pc_wait, lookback) survive; no-load-values demonstrably
      erases a real value-gated race on mixed flag data.
- [x] Verdict-attribute emission — landed:
      `CompiledRaceDetector.last_global_verdict` carries the taxonomy
      directly (verdict: race-free/race/potential-race/abstain,
      proved_scope, race_evidence exact/confirmed/widened, conservative,
      conditional=("termination",), unsupported_kind); harness row field
      `verdict_attrs`; 13 unit tests. Fixing its test surface exposed and
      closed a REAL soundness gap: for atomic-bearing graphs the
      used_pid_axes pinning rule's identical-behavior justification fails
      (observations distinguish no-pid blocks), so `symbolic_grid` now
      sizes unread axes from the real launch at T1 and keeps them symbolic
      at T0 — a no-pid narrow-slot work queue no longer proves falsely
      (regression tests added; atomic kernels' T0 rung correctly narrows
      to T1, e.g. trb015 amax).
- [ ] Results landing figure (OPTIONAL; formerly "the core figure", demoted
      2026-07-09 per the advisor's contribution-triad feedback — the
      symbolic/concrete axis is not the paper's headline, and the paper's
      benchmark table already carries the data): the 2-D concretization map
      of plan §I.2, exported from the results JSONL (each row's terminal
      state + front-end determines its point), as an evaluation-section
      figure; whether it enters the paper at all is pending the next
      advisor alignment. Figure script separate from the harness.

## 2. S5 — T0 stretch (off the critical path; interleave with evaluation)

- [x] Symbolic loop bounds at T0 — landed. Instead of a step-divisibility
      constraint over the induction VALUE, the encoder keeps the iteration
      INDEX k and swaps the concrete trip count for the iteration-existence
      premise `k ≥ 0 ∧ lower + k·step < upper` (linear — the step must
      still be a compile-time constant; a symbolic step is the k·step
      nonlinearity and falls to T1 per the ladder, as does any
      pid×param address via the existing linearity gate). Zero-trip
      launches are SUBSUMED: upper ≤ lower makes the premise UNSAT, so
      in-loop events are inactive with no phantom footprint. The T1 path
      is byte-identical (pid-dependent bounds keep the 'not concrete at
      launch' abstention the liger rows exercise). Corpus evidence:
      `trb019_symbolic_trip_no` — a real triton kernel
      (`for k in range(0, n): store(out + pid*SEG + k, mask=k<SEG)`) now
      lands on **proved@T0 for ANY trip count**; the spilling twin
      (mask k < 2·SEG) reports at n=128. 10 unit tests
      (test_t0_symbolic_loop_bounds.py). Existing corpus rungs unchanged —
      real kernels with symbolic bounds all carry pid×param addresses, so
      the gate correctly holds them at T1; the stretch feeds the paper's
      "upper bound" section with the linear-address family.

## 3. Track 1 — M4/M5 (shared-memory track; plan Part II §7)

- [ ] M4 — sm90/Hopper (submission scope — sm90 in or out — is the open
      Q5 with the advisor; align before starting): `ttng.warp_group_dot_wait {pendings}` agent,
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

- PR layout: `race-detector-z3-demo` now carries the plan-doc restructure
  plus S1–S6, the evaluation phases (A–C, mutation, RQ instrumentation),
  and the docs commits; decide whether to merge as one PR, split per step,
  or split detector-core / evaluation-harness before opening against main.

Independent verification stamp: all checked items above re-verified from a
clean run at 5ad7a72 (2026-07-09): 217 race-detector tests; TritonRaceBench,
tutorials, liger, scaling, ablation, and headline all reproduce; benchmark
determinism confirmed by terminal-identical back-to-back runs.
