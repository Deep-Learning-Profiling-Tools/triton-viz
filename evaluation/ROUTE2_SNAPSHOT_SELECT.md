# Route 2: loaded values as snapshot Selects in the static frontend

Design and implementation record, 2026-09-05. Status: MERGED into
`race-detector-z3-demo` (commits 490e73e the feature, 0eaee36 the review
fixes, c7d99d1 the least-concretization rule), the second half of the
ladder's L2 next to Route 3's multipath capture (paper repo
`design-route3-multipath-capture.md`; Route 1's record is
`CHANGE_SURFACE_L1.md` beside this file). Section 7 has the measured
change surface; the L2 pinned rerun that restates the paper's numbers is
the paper repo's `pre-submission/pinned-rerun.md`.

## 1. Problem and target

The static frontend (the TTIR reader and its global-memory encoder)
refused every kernel whose ADDRESS depends on a loaded value
(`store(out + idx[pid])`): 229 of the pinned run's 492 real-code
abstentions, the largest family, 237 rows once Route 3's reader admits
the loops and guards around them. A loaded value also widened every
MASK and branch CONDITION it reached (the mask dropped to its modelable
conjunct, the region guarded) and refused every LOOP BOUND (40
`data-dependent-bound` rows, the CSR row-pointer shape). Path
enumeration (Route 3) produces no data values; Route 1's per-instance
enumeration decides such rows only at the analyzed-launch extent with
no symbolic claim.

The interpreter frontend has modeled loaded values since the first
version: the launch's tensor contents are snapshotted before the kernel
runs and a loaded value becomes an SMT-array Select over that snapshot,
at the analyzed-launch extent. Route 2 gives the static frontend the
same source of values, so that the symbolic claims it already makes
(any grid along the read axes, this launch's scalar parameters) extend
to kernels whose behavior depends on contents: the proof is then
CONTENT-QUALIFIED (this launch's contents, any grid), one rung above
Route 1's analyzed-launch extent and one below Route 3's input extent.

## 2. The mechanism as built

Gated under the L2 reader mode (`parse_ttir(multipath=True)`, the same
flag that carries Route 3), so L0 and L1 are byte-identical: a loaded
value stays `DataDep` there and every refusal and widening fires as
before.

- **Reader** (`triton_viz/clients/common/ttir_reader.py`). An integer
  `tt.load` whose mask is modeled binds a
  `Loaded(access_index, base, offset, mask, other)` term instead of
  `DataDep`; float pointees and dropped masks keep `DataDep` (on a
  masked-off lane the value is `other` or undefined, which only a
  modeled mask can keep apart from the snapshot value). `other` is kept
  when it is a modelable term (a scalar or dense constant, a pid
  expression); an unmodelable `other` is dropped and the lane is
  unspecified (section 2, "unspecified lanes"). The term flows through
  arithmetic, comparisons, selects, `expand_dims` retagging (its lanes
  follow the consumer's dimension like an arange's), `addptr`, masks,
  `scf.if` and `cf.cond_br` conditions, and `scf.for` bounds. Every
  term walker descends into it (`loaded_leaves`, `mentions_loaded`).
- **Encoder** (`compiled/global_records.py`). `Loaded` evaluates to
  `If(mask, snap_base[off], other-or-free)`: `snap_base` is a Z3 array
  constrained by the equalities `snap_base[i] = v_i` over the tensor's
  pre-launch snapshot, asserted once in the solver base
  (`GlobalEncoding.assumptions`, the solver's `extra_assumptions`) and
  shared by both program copies because contents are launch-global.
  Any record that went through a snapshot Select marks the encoding
  content-qualified; the client appends `+content` to the proof rung
  (`proved@T1+content`, `proved@T1-launch+content`) and stamps
  `content_qualified` on the verdict attributes, on races too.
- **The domain premise.** Every CONSUMER of a loaded value carries the
  load's in-bounds premise, `mask → 0 ≤ off < numel`, as a local
  constraint (`domain_premises_for`; a loop whose bound goes through a
  loaded value carries it in its existence premise). An instance whose
  load reads outside its source is outside the model, exactly as the
  load's own access record already says, so an instance beyond a
  snapshotted table is EXCLUDED rather than given a free value. The
  first build did the latter and the any-grid query then found two
  out-of-table instances agreeing on a free address (review finding 1,
  section 6): a fabricated definite race whenever the launch-scoped
  requery could not run (a callable grid) or decide.
- **Unspecified lanes.** A masked-off lane of a load without a usable
  `other` holds an unspecified value: a free copy-local array `pad_i`
  (one per load, so no two lanes or instances are forced to agree).
  Wherever that value can reach an ACTIVE lane of a consumer (its
  address, mask, path, exit predicate or loop bound) the consumer is
  uncertain (`_pad_reaches`): its reports are widened, never definite;
  proofs stay sound (a free value only enlarges the executions). A
  consumer whose own mask or path repeats every conjunct of the load's
  mask keeps those lanes inactive and stays exact, which is the common
  `tl.load(p, mask=m)` / `tl.store(q, v, mask=m)` pair.
- **Free values.** Where no snapshot is usable the whole value is free
  (the widening Route 3 applied to unmodeled loaded values): T0 (no
  launch, so no T0 claim ever quantifies over contents), a float, too
  large, non-contiguous or uncaptured source, and a source that
  overlaps a tensor the kernel writes (the read-only-source premise,
  the interpreter frontend's fail-stop transposed: the pre-launch
  snapshot stands for the loaded value only if no instance writes the
  source first, and the static frontend cannot order instances; checked
  per launch by allocation-interval overlap before the loop bounds are
  bound). In mask, condition, exit-predicate and bound position the
  record is uncertain (`_widened_by_free_loaded`); in ADDRESS position
  the row refuses by name (`snapshot-bound` for a table beyond the
  bound, `indirect-address` otherwise), decided STRUCTURALLY on the
  address term (`free_reason_for`), never on which evaluation of the
  term came first (review finding 3). A loop-carried pointer whose
  per-iteration advance is a loaded value refuses by name too
  (`offset0 + k·delta` stands for the pointer only for a loop-invariant
  advance; review finding 10).
- **Least concretization** (c7d99d1). Contents are a concretization and
  are used only when the verdict needs them. `content_free_view(access)`
  is the access with every loaded value free: mask and path conjuncts
  built on one are dropped (they cannot help a proof and can be
  NONLINEAR, `offs < pid * len[pid]`, which would keep the kernel out of
  T0's linearity gate) and the record is flagged `mask_dropped` /
  `guarded`. T0 encodes this view; the client's T1 runs a content-free
  attempt first (the view, tensors without snapshots): an UNSAT over a
  feasible base is the any-contents proof single-path parsing made for
  the same kernel, so the row keeps `proved@T1`; a SAT, a refusal (an
  address on a loaded value) or an undecided query is not reported and
  the snapshot attempt decides exactly. Without this rule a more
  precise model was SHRINKING claims: `proved@T0` rows came back
  `proved@T1+content`.
- **The address-snapshot bound.** The client captures an integer
  tensor's contents at `pre_warmup` (before the kernel mutates it) up
  to 16384 elements (`ADDRESS_SNAPSHOT_MAX_ELEMENTS`), only at L2. This
  is the encoding-size boundary that already existed as the interpreter
  frontend's 1024-element load-source cap; 16384 equalities keep the
  solver's base check sub-second on this machine. Above it the source
  is unusable (free value; a refusal in address position).
- **Rungs.** Read-only tensor groups are skipped at T0 by construction
  (read/read cannot conflict), so a kernel whose loaded values steer
  only its reads (the gather litmus) still proves at T0, any input and
  any contents. A kernel whose loaded values steer a write goes to T1:
  content-free first, then with the snapshot; the launch-scoped rung
  applies as before when only the any-grid query is SAT.
- **Evaluation plumbing.** The concretization map places the
  `+content` rungs on the memory-contents column (the column Route 2
  makes reachable: memory concretized, paths still symbolic) at the
  underlying rung's y extent; the headline counts content-qualified
  proofs separately.

## 3. The claim, exactly

`proved@T1+content`: for this launch's scalar parameters and tensor
contents, every grid along the axes the kernel reads, under the
model's standing premises (in-bounds accesses, including the loads that
feed addresses, masks and bounds; distinct pointer arguments as distinct
allocations at T0 only; read-only load sources).
`proved@T1-launch+content`: the same at the launch's grid.
`race … content_qualified`: the witness is realized under these
contents (the C2 replay confirms it on the snapshot clones as for any
witness). The extent taxonomy of the paper gains one qualifier; the
paper's `evaluation.md` section 12 mapping and verdict taxonomy are the
`l2-adoption` items in its `TODO.md`.

## 4. Correctness obligations and how they are met

1. **Value faithfulness.** On an active in-bounds lane the modeled
   value is the snapshot element at the lane's offset: exact by the
   read-only-source premise and by the snapshot being taken before the
   kernel runs. An active out-of-bounds lane is outside the model (the
   domain premise). Everywhere else the value is unspecified (a free
   array), an over-approximation, and the record is marked uncertain
   unless the free value sits in an address, where the row refuses.
2. **Copy locality.** The snapshot array is shared by the two program
   copies (contents are launch-global); the padding arrays are
   copy-local, so unspecified values never couple two instances.
3. **No T0 claim over contents.** T0 has no launch and therefore no
   snapshot: every `Loaded` is free at T0 (the content-free view), and
   an address on one refuses inside the record builder, so the tier
   selector falls to T1.
4. **Uncertainty propagation.** A free or unspecified value in a mask,
   path, exit predicate, loop bound or (for unspecified lanes) address
   widens the record, the same channel `mask_dropped` and `guarded`
   use, so the client never certifies a report over it.
5. **Single-path invariance.** `parse_ttir` without the flag binds
   `DataDep` as before; the client captures no snapshot below L2; the
   refusal messages are unchanged (the pinned run's L0 rows are
   byte-identical, section 7).
6. **Claim monotonicity across levels.** A kernel single-path parsing
   proved at T0 or T1 proves at the same rung at L2 (the content-free
   view is the single-path widening, tried first).

## 5. Verification

Unit tests: `tests/unit/test_route2_snapshot_select.py` (19 tests) and
the updated Route 3 tests (`test_multipath_races.py`, 22, whose loaded
guards are now modeled): the scatter litmus (a permutation proves
content-qualified at the any-grid rung, a planted duplicate races with
the right witness pair, single-path still refuses; through the client
the L0 refusal, the L2 rungs, and a callable grid that leaves the
any-grid verdict on its own); the any-grid domain premise (an instance
beyond the table is excluded, the record carries the premise); the
address refusals by name, order-independent (a guard evaluates the
loaded value before the address); the written-source premise; T0
free-with-address-refusal; the nonlinear loaded mask proving at T0 with
the record widened and exactly at T1; masked-off lanes taking `other`
(other = 0 keeps them out of a guard, other = 1 sends them through it);
a masked load without `other` widening a guard and staying exact under
the same mask; an unmodelable `other`; the loaded pointer advance
refusing; a loaded index tile through `expand_dims`; CSR loop bounds
from a row-pointer table (disjoint segments prove, overlapping ones
race, no snapshot widens, the bound's domain premise excludes
out-of-table instances); the compiled gather golden (a masked index load
with a dense-constant `other` steering a float source: T0 proves, T1
content-qualified). The full unit suite passes (789 tests; the pinned
driver's timing-dependent rehearsal test is excluded on this machine).

## 6. Review

The feature commit (f76fb53 before the rebase) was reviewed the way
Route 3's two readers were: three independent finders (semantics of the
free/snapshot lowering and the read-only-source premise; T0/L0
invariance and the client's rungs; tests and evaluation plumbing), 12
findings, each judged by three refuters with distinct lenses against the
reviewed commit; every finding survived (three dissenting votes, all on
the concretization-map finding's impact). All fixed in 0eaee36 before
the merge:

- (high) an out-of-table instance's free value in address position
  became a definite race when the launch-scoped requery could not run
  or decide: the domain premise (section 2); `tb_apply_penalty`, a
  false race in the first change-surface run, proves;
- (high, twice) the address refusal keyed on "newly free during this
  evaluation", bypassed when the same load was evaluated earlier by a
  mask, a loop bound or an atomic operand: structural refusal;
- (high) a loaded per-iteration pointer advance encoded as
  `offset0 + k·delta`, a false-proof shape that single-path refused:
  refused by name again;
- (medium) a second loop sharing an already-free loaded bound was not
  widened: structural free-bound detection;
- (medium) an unmodelable `other` silently became a free lane with no
  uncertainty mark; (low) a masked load without `other` in a guard gave
  an exact report over an unspecified value: the unspecified-lane rule;
- (medium, low) the `+content` rungs unmapped in the concretization
  map: mapped to the memory-contents column;
- (medium) the mask/`other` arms of the lowering untested (a mutant
  dropping both passed); (low) fixtures without the mask/`other` every
  real `tl.load` carries; (medium) four Route 3 regression tests
  dropped by the feature commit: tests added and restored.

The least-concretization rule (c7d99d1) came out of the change-surface
comparison, not the review: a row proved at T0 under L0 had moved to
`proved@T1+content`.

## 7. Change surface and results

All runs on this machine (RTX 4090, Triton 3.6, z3 4.15.3), jobs 1 or 2,
seed 0, the runner's 200 s per-row protocol, datasets
`<corpus>_L2.<tag>.jsonl` beside the recorded ones (never over them),
compared by row name against the pinned run (`PINNED_fb91fc0.jsonl`,
level L0) and against Route 3's final L2 datasets
(`results/route3-change-surface/final-2e25373/`).

The full Route 2 surface is 261 rows (the pinned rows whose TTIR
mentions a loaded value in an address, mask, condition or bound, plus
the benchmark); only its short form was run before the pinned rerun,
which restates every row anyway: the 61 benchmark rows, the 20 rows the
pinned run had DECIDED (the only rows that can move down), and the
28-row `aiter_ops` smoke (the corpus with the most indirect rows).

**Benchmark (61 rows) at c7d99d1, jobs 2:** against the pinned run 55
rows unchanged, 6 upgrades, 0 downgrades: `trb006_dd_mask_dead_no`
`proved@interp` to `proved@T1+content`, `trb010_gather_no` to
`proved@T0` (the loaded index steers only reads), `trb010_scatter_yes`
`race@interp` to `race-confirmed` (witness pair 0/3, the planted
duplicate), `trb013_work_queue_plain_yes` `race@interp` to
`race-confirmed`, and Route 3's two trb011 rows. Against Route 3's L2
datasets 57 rows unchanged and the 4 Route 2 upgrades. The IR frontend
now decides 55 of the 61 rows (the paper's section 6.1 said 49 at L0,
Route 3 predicted 51).

**The 20 decided rows at c7d99d1, jobs 1:** 4 unchanged against the
pinned run (`pa_decode_sparse_reduce` stays `proved@T0` and
`tb_cross_entropy2` stays `proved@T1`, both through the content-free
attempt; the two `proved@T1+assumes-termination` layer-norm rows), 16
moved, all upward: 14 `proved@interp` rows now decide on the IR
frontend (13 `proved@T1+content`, `tb_kcache_copy_triton`
`proved@T1-launch+content`), `tb_quantize_kv_copy` `race@interp` to
`race-confirmed`, and `flaggems_embedding_dup__embedding_kernel` from
the pinned run's `race@interp` to `proved@T0`: that pinned race is the
casebook's phantom intra-instance WAW (the interpreter frontend's
same-axis arange coupling defect, `SWEEP_REPORT.md`), and the kernel's
only write is its own output row. Static solve time of the 20 rows:
median 0.4 s, p90 19 s, max 44 s (`tb_kv_cache_copy`). The first build
(f76fb53) had reported `tb_apply_penalty` as a definite race
(`races-unclassified`; the interpreter proved the launch): the
out-of-table free value of review finding 1, gone with the domain
premise.

**The `aiter_ops` smoke (28 rows, the corpus with the most indirect
rows) at c7d99d1, jobs 2:** 27 pinned abstentions and one `proved@T0`
row (unchanged). Of the 27: 13 decide on the IR frontend (10
`proved@T1+content`, 3 `proved@T1-launch+content`: the rope cache
kernels, the fused kv-cache kernels, mha_v3's split-K, paged attention
2d, kda's segment kernel, cat_and_cache_mla), 6 by Route 1's rung
where the static track still refuses (5 non-contiguous tensors, 1
loop bound that stays symbolic), 4 keep refusing (prefill attention
kernels whose loop bound is a loaded value inside an unmodeled
expression), 1 exceeds the row budget (`flash_kda_seg_scan`: it
proved `proved@T1-launch+content` in 195 s of static time at f76fb53,
and the content-free attempt of c7d99d1 adds a second solve on top;
the pinned driver's 320 s retry budget is the place it decides), and 3
are NEW RACE REPORTS: aiter's three causal_conv1d update kernels
(`_causal_conv1d_update_kernel`, `_causal_conv1d_update_single_token_
kernel`, its reshape variant), exact and content-qualified, an
intra-instance WAW on the conv-state store for the first two and
cross-instance WAR/WAW pairs on the reshape variant, `races-
unclassified` because the interpreter frontend abstains on these
kernels (host-side control flow on loaded data) so C2 cannot replay
them. Their labels say race-free; they are untriaged (the conv-state
store's 2-D tile addresses through captured strides and a loaded batch
coordinate), and the pinned rerun's race-row triage is where they get
a casebook entry or a modeling fix, before any number quotes them.
Static solve time of the 27 rows: median 0.6 s, p90 18 s, max 56 s.

Raw data: the main worktree's
`evaluation/results/route2-change-surface/` (gitignored like the pinned
file; `<corpus>_L2.r2final.jsonl`, `tritonracebench_L2.r2final.jsonl`,
the earlier `.r2dec` / `.r2` / `.r2bench` datasets at f76fb53, and the
`--only-file` lists). Regenerate with `evaluation.runner --ladder-level
L2 --only-file <list> --out-suffix .r2final` at the demo head.

## 8. Not done

- The cuTile reader has no Route 2 (8 indirect-address rows in the
  cuTile corpus): captured cuTile launches carry no tensor values yet,
  so there is nothing to snapshot; when the capture does, the encoder
  side is shared and only the reader needs a `Loaded` binding.
- The address-snapshot bound is a size boundary, not a tuned budget:
  rows whose index table exceeds 16384 elements refuse by name
  (`snapshot-bound`) and Route 1's rung decides them at the
  analyzed-launch extent.
- The content-free attempt does not run the launch-scoped requery: a
  kernel that proves content-free only at the launch extent lands on
  `proved@T1-launch+content` through the snapshot attempt (a smaller
  claim than `proved@T1-launch`, never a wrong one).
