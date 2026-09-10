# cuTile twins for the seven race-free repair rows

Date: 2026-09-10. Corpus: `evaluation/kernels/tritonracebench_cutile.py`
(62 rows to 69). Capture: `evaluation/tritonracebench_cutile_capture.py`
on cuda.tile 1.5.0, torch 2.10.0+cu128, RTX 4090 sm_89, the same
environment that captured the original 62 rows; each new row compiled
and launched once as a smoke check. The 62 pre-existing captures are
byte-identical after the merge.

## What was added and why

The seven race-free repair rows added to the Triton corpus at detector
`cf099aa` (`evaluation/kernels/tritonracebench_repairs.py`,
correctness arguments in `TRITONRACEBENCH_REPAIRS.md`) had no cuTile
twin, leaving the cross-DSL comparison at 62 of the Triton roster's 71
rows. Each twin carries the same row name, ground-truth label, pattern,
grid and argument contents as its Triton row, so the pairing stays a
name join.

| row | cuTile kernel | port note |
|---|---|---|
| `trb021_role_specific_order_no` | `trb021_role_order_kernel` | separate release-only producer CAS and acquire-only consumer CAS, in the twin's two branches |
| `trb013_batch_ticket_no` | `trb013_batch_ticket_kernel` | one relaxed fetch-add of two, `ct.broadcast_to(first, (2,)) + lanes` for the two reserved slots |
| `trb016_atomic_flag_observation_no` | `trb016_flag_observation_kernel` | the producer's flag observation stays a relaxed identity `ct.atomic_or` |
| `trb017_cas_unlock_no` | `trb017_cas_unlock_kernel` | acquire CAS spin plus a release CAS unlock (not the `atomic_xchg` of the pre-existing mutex row) |
| `trb025_failed_cas_arrival_no` | `trb025_failed_cas_arrival_kernel` | acquire CAS(0 to 0) poll whose failing read still acquires |
| `trb025_both_consumer_branches_no` | `trb025_both_branches_kernel` | both consumer branches poll, with identity add and identity OR |
| `trb026_fenced_tile_handoff_no` | `trb026_tile_handoff_kernel` | a BLOCK-wide payload published through a scalar acq_rel CAS |

## What replaces the fence

Six of the seven Triton kernels write `tl.debug_barrier()` around the
synchronizing atomic. The seventh, `batch_ticket_queue`
(`tritonracebench_repairs.py`), has no fence at all: its ticket
disjointness follows from RMW indivisibility and needs no
intra-instance order.

cuda.tile 1.5.0 exposes no fence, barrier or membar. Its compiler's
token pass does NOT order everything unconditionally; read from the
captured IR it emits exactly two kinds of edge:

- accesses through the SAME array parameter chain directly.
  `trb026_reread_fenced_no`: the read-back `load_pointer` takes
  `token=$66`, the tile store's own token.
- a RELEASE or ACQ_REL atomic receives `join_tokens` of the
  program-preceding memory operations, and later accesses receive
  `join_tokens` carrying that atomic's result token. A RELAXED or
  ACQUIRE-only atomic receives no such join.
  `trb021_acquire_only_yes`: its ACQUIRE CAS takes the entry
  `token=$token` while the producer's data store is `$80`, so nothing
  orders them, which is why that row is racy.

Every twin added here rests on the second edge, because its publication
is a RELEASE or ACQ_REL atomic, or needs no order at all
(`trb013_batch_ticket_no`). Atomics map one to one, with Triton's
sem/scope written out as `MemoryOrder`/`MemoryScope`.

### The cross-allocation edge, a new language fact

`trb026_fenced_tile_handoff_no` is the first row in this corpus whose
label rests on a token edge BETWEEN TWO ALLOCATIONS: the BLOCK-wide
store to `data` must be ordered before the scalar acq_rel CAS on `flag`,
and the consumer's load of `data` after it. The pre-existing
`trb026_reread_fenced_no` only needs same-array chaining, a weaker
claim, so this edge is not covered by any earlier row and is recorded
here as an observation about cuda.tile 1.5.0 rather than inherited from
`TRITONRACEBENCH_REPAIRS.md`. In the captured IR: the payload store
produces `$108`; the CAS takes `token=$token.0 = join_tokens(($token,
$108))` and produces `$181`; the consumer load takes `token=$token.1 =
join_tokens(($108, $181))`.

These chains are asserted from the captured IR by
`check_tritonracebench_cutile_twins.py`, for all seven rows, together
with a negative control (two racy rows that must have no RELEASE or
ACQ_REL atomic at all), so a recapture that lost an edge fails a check
instead of silently inverting a label.

## The two rows that stay Triton-only

`trb026_reread_unfenced_yes` and `trb026_guarded_no_producer_fence_yes`
are RACY BECAUSE a fence is absent: the label depends on two accesses of
one instance being unordered. There is no fence to drop from a cuTile
twin, and the decisive point is textual: the cuTile port of each
fence-dropped kernel is identical to an already-registered row that
carries the OPPOSITE label, `trb026_reread_fenced_no` and
`trb021_guarded_acq_rel_no` respectively. So no semantics-preserving,
name-matched twin exists. The only other spelling considered, binding
two kernel parameters to one allocation as `trb009_shift_inplace_yes`
does, is disqualified twice: it changes the argument contents, and it
needs `aliased=True`, which is outside the T0 premise. The cuTile track
therefore covers 69 of the Triton roster's 71 rows.

## Verdicts at L2 (detector `5ce2574`)

Three prove; four abstain on boundaries that already bound pre-existing
rows, so the abstentions are properties of the reader's fragment rather
than of these rows:

| row | verdict | reason |
|---|---|---|
| `trb016_atomic_flag_observation_no` | race-free | `proved@T1+assumes-termination` |
| `trb025_failed_cas_arrival_no` | race-free | `proved@T1+assumes-termination` |
| `trb025_both_consumer_branches_no` | race-free | `proved@T1+assumes-termination` |
| `trb013_batch_ticket_no` | abstain | indirect-address: pointer offset data-dependent (atomic result), as `trb013_work_queue_no` |
| `trb021_role_specific_order_no` | abstain | cas-value, as `trb021_guarded_acq_rel_no` |
| `trb017_cas_unlock_no` | abstain | cas-value |
| `trb026_fenced_tile_handoff_no` | abstain | cas-value |

The cas-value refusal is raised in
`triton_viz/clients/race_detector/compiled/global_records.py` for a CAS
that is not awaited (not in a spin loop) when the graph carries
`has_value_changing_integer_casts`; the cuTile reader
(`triton_viz/clients/common/cutile_ir_reader.py`) sets that flag whenever
the IR contains both `tile_atomic_cas(` and any `= tile_astype(` line,
and all 69 captured IRs contain the latter. So in this corpus an
ordinary, non-spin CAS always abstains, and six pre-existing rows already
satisfy the same trigger (`trb021_guarded_acq_rel_no`,
`trb021_release_only_yes`, `trb021_acquire_only_yes`,
`trb022_acquire_on_failure_no`, `trb022_acquire_on_failure_relaxed_yes`,
`trb023_oversized_flag_conservative`). One nuance in the mutex family:
`trb017_mutex_cas_no` unlocks with `atomic_xchg` and its only CAS is the
awaited spin, so it proves; `trb017_cas_unlock_no` is the family's first
abstention, forced by the very release CAS unlock that defines the
Triton row. No new row reports a race, and no pre-existing verdict
changes.

## Independent checks

`evaluation/check_tritonracebench_cutile_twins.py` verifies, without
consulting any detector verdict: the 69-row catalog, the name/label/
pattern/grid pairing against the Triton corpus, that exactly the two
rows above are Triton-only, and (with `--gpu`) ten launches of each new
row whose outputs match what the Triton twin is specified to produce.
GPU success checks execution and outputs; the race-free arguments are
the ports of the Triton arguments in `TRITONRACEBENCH_REPAIRS.md` and do
not follow from the absence of observed output errors.

Adoption note: the pinned evaluation numbers are unchanged by this
commit. A cuTile population that includes these rows requires a new
capture-pinned run; the current paper counts keep their 61-scored-row
cuTile denominator until such a run is reviewed.

## Roster consequence

The frozen definitive roster grows from 1,249 to 1,256 configurations
(`evaluation/pinned_manifest.py`, `evaluation/PINNED_RESUME.md`), the
same mechanism that moved it from 1,242 to 1,249 when the seven Triton
repair rows landed. Runs pinned before this commit keep their own
1,249-row manifests and remain valid at their pins; a run that includes
these twins is a new pin.
