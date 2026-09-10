# Independent review of seven race-free repair variants

Date: 2026-09-08.

Scope: source review of `/private/tmp/tritonracebench_repairs.py`, compared with the original kernels in the isolated detector checkout `/tmp/triton-viz-seven-race-free-repairs` on 4090. This reviewer did not run detector or GPU experiments. Detector statuses below are the implementing agent's initial diagnostic observations, not independently reproduced measurements.

## Assessment

All seven kernel bodies have coherent race-free arguments under the paper's memory model for the declared, non-aliasing input allocations and participating grids. These arguments justify ground-truth labels independently of the detector's current ability to prove them. They do not establish that all seven receive conclusive detector verdicts.

The final source preserves the paired original's 64-element producer-consumer payload, 128-element output, four-element mutex output, and two-instance mutex grid. The queue retains its original 64-element allocation, of which its two-slot batches use the first eight elements. The queue and vector handoff deliberately adapt their original controls rather than claiming a single textual mutation with identical launch geometry.

## Per-case arguments and distinct coverage

1. **`trb021_role_specific_order_no`.** The producer stores data, fences, and publishes with a release CAS. The consumer's acquire CAS guards its payload access on observing one, then fences before loading. With initial flag zero, an active consumer cannot obtain one from the initial state; the publication supplies the value, possibly through compatible identity RMWs. Its release/acquire path and the two fences order the payload accesses. A consumer that runs before publication may skip the load, so one successful empty-output run is insufficient validation. Unlike the existing common `acq_rel` CAS instruction, this variant represents the roles by separate release-only and acquire-only atomic records.

2. **`trb013_batch_ticket_no`.** Four compatible atomic adds of two, starting from zero, reserve bases 0, 2, 4, and 6 in some order. Each instance stores two adjacent lanes, producing disjoint intervals that partition `[0, 8)`. The increment cannot overflow on this launch, and the 64-element allocation contains every store. All four instances still write; uniqueness comes from ticket allocation, not a disabled participant. This extends the original scalar narrow-slot case to batch reservation, requiring both counter spacing and lane offsets. It is an adapted repair, not an unchanged scalar control. Failure of the implementation's counting admission does not invalidate this mathematical argument.

3. **`trb016_atomic_flag_observation_no`.** The producer's flag observation remains a real access with an output; replacing its plain load by device-scoped identity atomic OR makes it compatible with the consumer's polling RMWs. The OR preserves the flag value. Publication after the producer's fence and acquisition before the consumer's fence order all payload stores before the payload loads. The producer output at index zero and consumer output beginning at `BLOCK` are disjoint. This directly repairs the original flag-read conflict and exercises compatibility with pre-exit polling accesses. Producer `flag_value` and consumer vector `value` now have distinct variable names, resolving the initial scalar/vector branch-variable collision without changing the synchronization argument.

4. **`trb017_cas_unlock_no`.** A successful acquire CAS changes zero to one; indivisibility prevents two critical sections from acquiring the same zero. The lock fence orders the critical-section accesses after acquisition, and the unlock fence orders them before the release CAS from one to zero. While an instance holds the lock, unsuccessful acquisition attempts preserve one, so its release CAS can unlock. A subsequent successful acquisition of zero acquires that release. Within one instance the load-to-store value dependency orders the scalar increment. Outputs are per-instance and disjoint. This retains the lock and shared increment but uses conditional release CAS instead of the existing release exchange. The spin loop requires the termination premise for a summarized completed launch; the example makes no scheduling or fairness guarantee.

5. **`trb025_failed_cas_arrival_no`.** Before publication, CAS `(0, 0)` may succeed while preserving zero, but zero never satisfies the loop's exit test. An exit observation of one is a failed CAS read with acquire semantics; it obtains the producer's released one. Both consumers fence before reading the complete payload, and their output slices are disjoint. The initial-value bug is repaired while specifically testing acquisition on CAS failure, with two participating consumers. The proof of post-loop accesses retains the termination premise.

6. **`trb025_both_consumer_branches_no`.** Grid three activates the producer, the first consumer's acquire-add loop, and the second consumer's acquire-OR loop. Both identity operations preserve the flag, both exits require the published one, and a fence follows either loop before payload access. The two consumers therefore acquire the same producer publication, possibly through compatible identity RMWs, and write disjoint output slices. This repairs the skipped-poll branch without removing its participant and exercises two independently captured wait sites. A detector result marked `vacuous` is not evidence of this proof: the concrete program has a feasible execution in which the producer publishes and both consumers complete. Preserve the limitation and the case rather than substituting a simpler program solely to obtain a favorable detector result.

7. **`trb026_fenced_tile_handoff_no`.** A single scalar flag publication, with a fence on each side, orders every element of the producer's 16-element tile before the consumer's guarded tile load. Producer-only and consumer-only masks leave one writer and one reader per payload element; the consumer alone writes the output tile. An active consumer requires observing the publication, as in the scalar guarded protocol. This extends the existing scalar repair `trb021_guarded_acq_rel_no` to vector payload coverage. It must be described as an adapted vector repair, because the original fence-dropped control is scalar. Confirm that all 16 positions participate and are checked; changing only the case name would add no coverage.

## Initial detector observations

The first diagnostic run reported the following, before all source and registration fixes:

| Case | Initial observation | Interpretation |
|---|---|---|
| role-specific ordering | unsupported | No conclusive detector proof; retain the branch/atomic capture limitation. |
| batch ticket queue | unsupported | Counting was not admitted; no conclusive detector proof. |
| atomic flag observation | compile error | The scalar/vector variable collision was fixed; rerun the final source and record its actual outcome. |
| CAS unlock mutex | unsupported | No conclusive detector proof; retain the CAS unlock limitation. |
| failed CAS arrival | `proved@T1+assumes-termination` | A qualified detector proof; preserve the exact input domain and termination assumption. |
| both consumer branches | vacuous | Not accepted as a race-free proof; investigate capture or feasibility failure. |
| fenced tile handoff | `proved@interp` | A proof for the interpreter's recorded domain; do not promote it to an unrestricted IR proof. |

Final receipts supersede these observations for the final source revision. Unsupported, compilation failure, and vacuity must remain distinguishable. No aggregate claim of seven successful proofs is supported by this initial run.

## Validation and adoption

- Bind each final receipt to the source revision, exact case identity, dimensions, initial contents, and participating grid. Confirm that the repaired taxonomy contributes exactly one clean case to each deficient family except comm/comp, which gains two.
- Check non-aliasing, active in-bounds footprints, and all expected output positions. The queue should reserve eight slots and leave its head at eight; the mutex should complete both increments and leave its lock zero. Both semaphore consumers must produce complete output slices.
- For conditional, non-spinning CAS guards, establish at least one execution with the consumer active and correct output. An execution that skips the consumer is legal but cannot alone establish meaningful repaired coverage. For spin cases, observe completion as a functional check while retaining the proof's termination qualification.
- Preserve the original racy controls and their observed detector outcomes. Particularly useful unscored mechanism checks would reduce the new queue reservation from two to one while retaining two stores, remove the vector handoff's producer fence, or remove the role-specific producer's release. These are additional validation suggestions, not experiments performed by this reviewer.
- GPU compilation, termination, and correct outputs provide functional evidence. They do not prove absence of races over all schedules. Keep such receipts separate from the source argument and solver verdicts; report successful GPU checks only after their receipts exist.
- Keep valid but unsupported repair variants in the benchmark. Do not replace them merely to obtain a balanced set of detector successes. A balanced ground-truth inventory is distinct from balanced or perfect detector outcomes.
- The paper's adopted 63-case benchmark remains frozen. Adding seven source cases does not authorize restating the old measurements as a 70-case run. Before adopting new paper counts, update the pinned roster and validation, verdict/abstention/error totals, frontend contributions, timings, and external-tool scoring. The 35 racy labels and 29 designated endpoint checks remain unchanged when only these clean cases are added; cuTile's denominator remains separate until real counterparts are added and validated.


## Corrected development validation

With the initial scalar/vector variable collision and taxonomy corrected,
all seven native kernels compiled and passed ten GPU output checks each.
The two non-spinning guarded consumers were not observed active in these
launches; their allowed empty outputs are recorded, not treated as evidence
of active communication. Non-vacuity follows from the explicit admitted
schedule in which the producer stores and publishes before the consumer
CAS. The spinning producer/consumer completed ten times, and each of the
two semaphore consumers completed ten times. Both mutex increments and
every reserved queue batch were checked. Native outputs do not prove race
freedom across schedules.

The corrected complete-system run proves atomic_flag_observation and
failed_cas_arrival with a termination premise, and fenced_tile_handoff
through the interpreter. The other four abstain (three unsupported, one
vacuous). All seven original racy controls still report. These results
are development validation; clean-commit acceptance is recorded separately.
