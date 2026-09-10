# Arithmetic conflict prechecks for RWKV7 and ABC

This change targets two different slow-query shapes seen in the selected
FLA L2 static profiles. It does not change the memory model, snapshot
eligibility, symbolic grid, feasibility obligation, or full-query budgets.

## RWKV7: avoid irrelevant array reasoning

The existing snapshot encoder exposes small captured integer tables as
conditional expressions, retaining the original array Select outside the
captured domain. RWKV7 uses `cu_seqlens=[0,29,64]`, block size 16 and
`chunk_offsets=[0,2,5]`; the sequences therefore have two and three loop
iterations. Its record activity already contains the table-index domains.

A necessary address-conflict condition retains those domains and all
conditional guards, but replaces array reads consistently by fresh
integer or Boolean values. If linear integer arithmetic proves this
weaker condition unsatisfiable, the original pair cannot conflict.
An active or insufficiently constrained array read remains unconstrained;
the optimization must then fall back whenever disjointness cannot be proved.
It neither deletes the original fallback nor assumes a smaller grid.

## ABC: preserve mixed-radix uniqueness in linear arithmetic

The difficult ABC store address contains `pid0 * grid2 + pid2`. The grid
already supplies `grid2 > 0` and `0 <= pid2 < grid2`. For two such terms
with the same radix, equal flattened values imply equal quotient and digit.
The precheck abstracts the nonlinear terms and retains this implication.
It preserves outer address terms, lane bounds, and actual byte-interval
overlap, including partial overlaps. A separate address contribution can
cancel a flattened-index difference, so the implication is never applied
to an entire address without establishing equality of the flattened terms.

## Soundness and integration

Take any model of a complete original pair query. Assign every fresh
variable the value of its replaced original expression in that model.
This extends the model to the relaxed conditions. The mixed-radix
implications are arithmetic consequences of original asserted bounds,
so the extension also satisfies them. Thus an unsatisfiable relaxation
implies an unsatisfiable original query. SAT and unknown are inconclusive.

The relaxed condition retains activity, conflicting access modes, byte
overlap, grid bounds, lane ranges, and the existing extra assumptions
(including launch-grid pins). Omitting HB, reads-from and the remaining
base constraints can only admit more assignments. In same-instance
queries, substitution follows the existing equality of pid and copy-local
variables; lane variables remain independent. Cross-instance copies
remain independent. Feasibility is checked separately by its original
solver, even when every pair is excluded by the shortcut.

The optional precheck has its own 500 ms limit. An initial 50 ms trial
was too close to the observed solve time: isolated necessary conditions
took up to 213 ms. This limit controls optional work, not proof scope. Expiration returns to the
complete query with its original budget; it never yields a proof, changes
enumeration state, or reduces the original 120 s / 10 s policy. Pair
statistics retain one decision per candidate and include precheck time.
Profiling distinguishes precheck unknowns from complete-query timeouts.

## Measurement protocol

Measure the complete selected static pipeline, including construction,
HB, solver initialization, checks and replay, with
`baselines/profile_fla_static.py` in the paper repository. Compilation and
process startup precede the timer. Fix the source commit, captured TTIR,
snapshot sidecar, seed, ladder level, fence policy and solver budgets.
Run fresh processes serially and retain every attempt and scope change.
The two private shortcut switches permit a within-commit component
ablation. These selected cases are not a full FLA speedup estimate and do
not replace the stopped official rerun.

## Validation

The complete unit and end-to-end suite passes: 1269 passed, 11 skipped.
The new regressions cover mixed-radix domain and coefficient errors,
copy-local radices, outer-term cancellation, lane carries, partial-byte
and atomic overlap, independent array reads, out-of-domain snapshots,
launch pins, independent feasibility, and unknown fallback.

Development trials reduced the selected RWKV7 static pipeline from about
139 s to 7.8 s and ABC to 2.3 s. These are pilot observations, not final
within-commit ablation statistics. A larger correlated piecewise rewrite
was tested and removed: bypassing it was faster while retaining the same
proof scope and the real extra-grid counterexample. The shipped change
therefore leaves the snapshot encoder unchanged. Final controlled
measurements and all variants are archived with the paper experiment.
