# Address-position lifting — hand-off spec (TODO §3d)

Status: IMPLEMENTED 2026-07-11 (§8 steps 1-6; step 7's TritonBench
migration + RQ5 refresh follow the corpus rerun). The six acceptance
families live in tests/end_to_end/test_address_position_lifting.py;
composed-dispatcher terminals race@interp / proved@interp landed in
evaluation/harness.py with dynamic-witness serialization and the
interp-disagreement audit bucket. Originally: hand-off spec,
2026-07-11, ADVERSARIALLY VERIFIED — every
factual anchor below was independently checked against the code at
commit `7e71ac0` (6/6 claims confirmed; the §1 co-admitted-atomics
note, the CAS constraint-discard fragility, and the §4 latent-trap
pins are the findings of that pass). Companion to
`race_detector_static_hybrid_plan.md`; implements TODO §3d (i)–(v).

Driving evidence: 37 of 202 TritonBench_G_v1 rows abstain with
`indirect-address` (36× `addptr offset: data-dependent (arith over
loaded data)`, 1× `… (loaded value)` — the largest abstention class),
and the interpreter refuses the same rows, so the composed dispatcher
has no route at all. The corpus rows this must decide are the KV-cache
scatter family (`tb_destindex_copy*`, `tb_kv_cache_*`,
`tb_quantize_*kv*`), index-select/embedding (`tb_index_select_*`,
`tb_embedding_triton_kernel`), and attention block-table indirection
(`tb_flash_attn`, `tb_context_attn_*`, `tb_token_softmax_*`).

## 0. Placement decision and claim scope

The lift lands in the **interpreter front-end only**. This is the
§I.3 placement rule verbatim (plan:91-93: *"loaded value in an
address chain → a free address makes the query meaningless (nearly
always SAT) → route the kernel to the interpreter front-end"*) — the
route exists; today the interpreter rejects at the door
(`race_detector.py:449-461`). The static track's `indirect-address`
abstention (`ttir_reader.py:1274-1284`) is NOT touched: it is the
routing signal, and the static tiers cannot carry this claim — T0 has
no values by definition, and a T1 select over launch contents is a
different (weaker) verdict scope than what `proved@T1` currently
means. A static-T1 contents tier is a possible follow-up, not part of
this spec (§7).

**The claim being made.** A dynamic verdict on a lifted-address kernel
is scoped to *this launch's parameters AND the pre-access contents of
the index tensors*: the loaded index is modeled as a `Select` over a
snapshot array of the live tensor. The verdict therefore carries a new
premise, `contents-snapshot` (§6), alongside the existing per-launch
scope. Within that scope both directions are exact: the snapshot IS
what every instance's load returns, because the read-only side
condition (§2) fail-stops any same-kernel write that could make
snapshots instance-divergent or stale.

Why there is no sound fallback (TODO §3d preamble, restated
operationally): a FREE address makes `_byte_overlap`
(`two_copy_symbolic_hb_solver.py:1077-1084`) satisfiable for
essentially every pair (fabricated races everywhere — the mask-position
free-variable trick from plan:94-98 is one-directional and address
position has no such direction), while a WRONG address (stale snapshot,
out-of-domain index) can both fabricate and hide overlaps. Hence the
three load-bearing side conditions: read-only index sources (§2),
domain pinning (§1), byte-exact snapshot addressing (§1).

## 1. Part (i) — select(A_T, t) in event address expressions

### What already exists (do not rebuild)

- Snapshot arrays: `_snapshot_array_for_tensor`
  (`race_detector.py:685-693`) builds `K(IntSort(), IntVal(0))` +
  `Store` chains mapping concrete byte addresses `base + i·elem` to
  concrete values, returns `(arr, known_addrs)`, caches by
  `(base, elem_size, numel, dtype)`. Guards: torch tensor, contiguous,
  int/bool dtype (`_is_modelable_dtype`, rd:613-635), `numel ≤
  _MAX_LOAD_SOURCE_ELEMENTS = 1024` (rd:263).
- Value lowering: `_load_value_provider_impl` (rd:711-795) —
  unmasked lanes become `Select(arr, a)` with domain term
  `Or(*(a == k for k in known_addrs))` (rd:766-768); masked lanes
  become `If(m, Select(arr, a), other)` with
  `Implies(m, Or(…))` (rd:785-786); masked load without explicit
  `other` is a hard unsupported (rd:770-773).
- The constraint channel: the provider's domain terms return as
  `extra_constraints`; `_safe_eval` (rd:396-398, provider installed
  around EVERY detector eval, pointer evals included) surfaces them as
  the second element of `(z3_addr, ptr_constraints)`
  (rd:1678-1691), which `_record_access_event` stores on the event and
  `_lower_record` folds into `active`
  (`two_copy_symbolic_hb_solver.py:669-678`).
- Per-lane structure: an index TILE `idx_ptr + offs` with
  `offs = pid·B + arange` lowers with ONE symbolic lane per arange
  site (`solver.py:588-609`, range constraints `start ≤ λ < end`
  asserted in every query, `solver.py:1506-1509`). So lane λ's lifted
  address is exactly the TODO's shape:
  `dst + elem·Select(A_T, base_T + es_T·(pid·B + λ))`.

### The change

`_VALUE_DEPENDENT_ADDRESS_OPS` (`race_detector.py:425-432`) currently
gates `("load", "tensor_pointer_load", "atomic_cas", "atomic_rmw",
"sort", "cumsum")`. Remove **only** `"load"`. Everything else stays
rejected:

- `atomic_cas` / `atomic_rmw` returns are **interleaving-dependent**,
  not snapshot-stable — a snapshot of their value would be wrong in
  both directions. They remain admitted in addresses exactly and only
  under the counting axiom (the work-queue pattern; solver-side guard
  `_assert_no_uncounted_observation_addresses`,
  `solver.py:1470-1498`). No change.
- `sort` / `cumsum` have no snapshot semantics (kernel-computed
  permutations of runtime data). No change.
- `tensor_pointer_load` (block-ptr loads) shares snapshot semantics
  in principle but has a different lowering path (descriptor exprs,
  rd:1663-1676); out of scope here, noted in §7.

**Co-admitted surface (deliberate, verified):** the same gate function
serves all three record sites (`rd:1660` load/store, `rd:1747` CAS,
`rd:1818` RMW), so removing `"load"` also admits plain-load-derived
addresses of ATOMIC accesses. This is consistent with the spec's own
logic — snapshot-stability attaches to the loaded value's SOURCE
(read-only, §2), not to the consuming access's atomicity; atomic
RETURNS stay gated by the remaining list entries; atomics register
their write targets (rd:1494, rd:1570) so index/target aliasing
fail-stops; and the counting-axiom guard (solver:1470-1498) keys on
observation VARS, which a concrete-array Select never introduces. §4
adds an atomic-consumer acceptance test so the surface is exercised,
not just argued.

**Known fragility to fix during implementation:** the CAS record site
DISCARDS the pointer eval's constraint conjunction (`addr_expr, _ =
result`, rd:1756); today the domain terms survive only because the
earlier full-CAS eval (rd:1749) already folded the pointer constraints
into the event via `AtomicCasSymbolicExpr._to_z3_impl`
(symbolic_engine.py:2434-2439) AND the per-node cache returns the
identical conjunction. Make the rescue explicit: keep the tuple's
constraints at rd:1756 like the RMW site does (rd:1823, 1846-1848),
and pin it with a test regardless.

After the gate change, no further wiring is needed for the happy path:
`_safe_eval(addr_attr, …)` already runs under `_load_value_semantics`,
so an embedded plain `tl.load` in the pointer chain lowers to the
masked/unmasked select shapes above, and its domain terms ride the
existing `ptr_constraints` channel into the event's `active`.

One docstring must be corrected alongside: `_record_access_event`'s
claim that the address eval is *"independent of any load-value
provider"* (rd:1644-1652) becomes false by design — rewrite it to
state the new semantics (pointer chains lower embedded plain loads to
snapshot selects; CAS/RMW/sort/cumsum pointers still reject).

### Domain constraints: why `active`-folding is sound both ways

The domain fact `a ∈ known_addrs` holds in **every real execution**
(the hardware load read some slot of the real table — the snapshot's
address set is exactly that table, byte-for-byte, because the snapshot
is taken from the live tensor at `base + i·elem`). Conjoining a fact
that holds in all real executions into `active`:

- cannot HIDE a real overlap — a real racing pair satisfies the fact,
  so it remains a model of the query;
- prevents FABRICATION — Z3 cannot choose an out-of-table inner
  address to manufacture an overlap (this is precisely TODO (i)'s
  "out-of-domain indices cannot fabricate or hide overlaps").

For masked gathers the guard is `Implies(mask, domain)`: a masked-off
lane's inner address is unconstrained but its VALUE is `other`, so the
lifted outer address is `dst + elem·other` — which is the semantically
true address the consuming access would use on real hardware. No
special case; §4's acceptance tests pin it.

### Snapshot-time correctness (when is the array built?)

`_snapshot_array_for_tensor` (rd:637-693) reads the LIVE tensor at
eval time.
Events are recorded during the interpreter run, i.e. the snapshot is
taken when the traced load executes. The read-only side condition (§2)
guarantees no same-kernel write precedes or follows it on that region,
so "at eval time" equals "pre-launch" equals "what every instance
reads" — instance-uniformity of the snapshot is exactly what §2
enforces, and is the A1-transport obligation of §4.

## 2. Part (ii) — read-only flow check for index-source tensors

**This is already free, and the spec's job is to pin it, not build
it.** The provider path calls `_note_load_source_or_raise` (rd:745,
impl rd:594-609) for every snapshot it builds — including snapshots
that will now serve address position — and writes/atomics register
their targets via `_note_written_tensor` (rd:558-592, called from
`_record_access_event` rd:1372-1374). The tracking is bidirectional
over byte-interval regions (`_tensor_region`, rd:534-543):

- write-then-snapshot: registration rejects a source overlapping any
  prior write (*"tl.load value from a tensor written by this kernel is
  unsupported"*);
- snapshot-then-write: the write rejects against prior load-source
  regions (*"tl.store/atomic into a tensor previously read as a
  tl.load value source"*);
- unknown write target: poison flag `_unknown_written_region_seen`
  (rd:568-575), re-checked before every snapshot (rd:731-735).

Fail-stop mechanics: `_raise_or_mark` (rd:549-556) marks the launch
unsupported BEFORE raising, so `finalize` can never read a clean
verdict past a violation. TODO (ii)'s *"stale snapshots in address
position are wrong in both directions"* is discharged by exactly this:
any interleaving that could make instance i's load differ from the
snapshot requires a same-kernel write to the region, and every such
write fail-stops.

Deliverables for this part are therefore tests only (§5): the
written-index fail-stop pair (store to `idx_ptr` before/after the
gather), and index/data aliasing (the same underlying storage passed
as both `idx_ptr` and `out_ptr` — caught by region overlap since
regions are address intervals, not tensor identities).

New abstention reasons must stay legible in the harness results: the
existing strings above classify as `unsupported` with the
interpreter's reason. The cap message (rd:656-660) already prints the
element count and the cap; add the tensor's role ("index source") so
corpus rows that die on table size in ADDRESS position are
distinguishable from value-position cap hits when bucketing the
TritonBench migration (§5.3).

## 3. Part (iii) — the overlap query over select-containing addresses

### Encoding shape (validate, not redesign)

The pair query is unchanged: `_race_expr` = conflict ∧ no-HB
(`solver.py:1094-1099`), overlap = interval intersection over Int
addresses (`solver.py:1077-1084`). What changes is the address TERM:
`IntVal(dst_base) + elem·Select(arr, inner)` where `arr` is a
**closed** (variable-free) concrete array and `inner` is linear in
(pid, λ, loop vars). Consequences to validate:

- **Alpha-renaming** (`_lower_record` sub build, solver:642-648)
  substitutes only pid/arange/copy-local vars; `Select`'s array
  argument is constant so `apply_sub` rewrites only `inner`. Add a
  unit test that the a/b copies of one gather event get DISTINCT
  `Select(arr, …pid_a…)` vs `Select(arr, …pid_b…)` terms over the
  SAME array object.
- **Decidability/cost**: QF_ALIA (arrays + linear integer arithmetic)
  with closed arrays of ≤1024 `Store`s. Z3 handles this by
  store-chain axiomatization; the risk is per-query cost in the m²
  loop (`find_races`, solver:391-415). The existing `query_stats`
  instrumentation (solver:398/408) is the measurement tool: the
  acceptance run records per-query mean/p95 on the scatter litmus
  (table size 64) and on 3 TritonBench sample rows (real table sizes),
  under the harness's per-spec 180 s budget. If p95 per query exceeds
  ~1 s at 1024 entries, the fallback is lowering the effective cap for
  address-position snapshots (a policy constant, NOT a correctness
  change — over-cap rows abstain with the §2 cap reason).
- **`unknown` policy**: unchanged — `_race_query_is_sat`
  (solver:417-439) already converts Z3 `unknown` into
  `UnsupportedSymbolicRaceQuery`, so a theory blow-up degrades to
  honest abstention, never a silent verdict. No linearity gate blocks
  selects anywhere: the T0 gate lives in the compiled track's Term IR
  (`compiled/global_records.py:750-801`) which never sees Z3 arrays,
  and the dynamic track has no syntactic gate.

### Witness extractability (feeds §4)

`_make_report` (solver:1555-1597) already evaluates with
`model_completion=True` (solver:1571-1572); with a closed concrete
array, `Select(arr, inner)` completes to a numeral under any model
(even an out-of-table inner completes via the array's K-default), so
`as_long()` is total — verified by probe. The domain fact's job is
therefore MEANING, not totality: it keeps the completed value equal to
what the real table holds. Pids and source lines are
model-independent (static on the record). Add an assertion-backed test:
the trb010 scatter witness must carry the CONCRETE clashing byte
address (`out + 0` for the all-zero index table) and two distinct
pids.

## 4. Part (iv) — witness-soundness revalidation (A1/A2 transport)

Theorem `thm:witness` and its A1/A2 side conditions live in the paper
(§4); they are not in-repo. The transport argument to re-walk in the
paper text, stated here in repo terms so the acceptance tests are the
backing:

- **A1-shape obligation (the model corresponds to a reachable launch
  state):** the new model component is the memory-contents premise.
  Transport: the snapshot equals the pre-launch table contents for
  every instance (§1 snapshot-time + §2 fail-stop), the domain fact
  holds in every real execution (§1), and the masked-default If-shape
  reproduces the hardware address of masked-off lanes (§1). Therefore
  a SAT model's address valuation is realized by the actual launch
  under the same contents — the witness transports with the premise
  `contents-snapshot` attached.
- **A2-shape obligation (side conditions under which UNSAT is a
  proof):** UNSAT now quantifies over all models satisfying the
  domain facts, a superset of all real executions of THIS launch with
  THIS table (again §1/§2); the proof claim is scoped accordingly
  (§0) and the verdict attributes must say so (§6).

The acceptance tests ARE the backing (TODO (iv) verbatim); the four
named families, concretely:

1. **written-index fail-stop**: gather whose kernel also stores to
   `idx_ptr` (both orders: write-before-load and load-before-write) →
   `unsupported`, never a verdict; plus the unknown-write-target
   poison variant.
2. **OOB-index domain**: an index table whose VALUES point outside
   `dst` — the lifted addresses are the true (OOB) addresses; the
   query must neither crash nor exclude them artificially (two
   instances scattering through the same OOB slot still race). This
   pins that domain constraints restrict the inner address, not the
   outer one.
3. **index/data tensor aliasing**: `idx_ptr` region overlapping the
   written `out_ptr` region (same storage or offset views) →
   fail-stop via §2 region overlap.
4. **masked-gather default interplay**: masked index load with
   `other=c` feeding a store address; instances whose mask differs
   must race/not-race exactly per the `If(mask, Select, c)` address —
   include the missing-`other` hard-unsupported case.
5. **atomic consumer** (the co-admitted surface of §1): an
   `atomic_add(dst + idx, v)` with a plain-loaded `idx` — all-equal
   index table must NOT race (mutually-atomic same-address adds),
   and the same kernel with a plain `tl.store` twin must; plus the
   CAS-site constraint-discard pin (the domain terms must reach the
   query even through rd:1756's tuple discard).
6. **latent-trap pins** (verified fragilities, not new machinery):
   (a) `_force_eval_record_templates` drops constraint conjunctions
   for record fields still symbolic at finalize (rd:1324) — pin that
   no lifted-address record ever reaches that path; (b) per-node
   `_to_z3` caching is first-lowering-wins (symbolic_engine.py:
   1021-1028) — pin that a load node lowered under the provider is
   never first lowered outside it in a detector run.

## 5. Part (v) — definition of done

1. **Scatter litmus pair** (extend `tritonracebench.py`):
   - `trb010_scatter_yes` (exists, rmw-style all-zero index table,
     labeled `race`, `expected` witness = colliding store line): flips
     from `unsupported` to a DYNAMIC race verdict with
     confirmed/exact witness (concrete byte + distinct pids, §3).
   - NEW `trb010_scatter_no`: identity-permutation index table, same
     kernel, labeled `race-free`: dynamic proves clean (per-launch +
     contents scope).
   - `trb010_gather_no` (= `smoke_gather_no`): upgrades from abstain
     to a dynamic clean verdict.
2. **trb013 work-queue family** (`rmw_sync.py`):
   - `trb013_work_queue_plain_yes` (`wq_plain_fetch_kernel`,
     rmw_sync.py:106-110): the plain-loaded head is read-only in the
     kernel → snapshot gives every instance the SAME `idx` → all
     instances store `buf + idx` → WAW SAT → race verdict. Flips from
     both-tracks-abstain to detected; the module docstring's "honest
     coverage miss" note (rmw_sync.py:8-11) is updated to record the
     lift.
   - `trb013_work_queue_no` / `_narrow_yes` (atomic fetch): UNCHANGED
     — counting-axiom path; regression-pin that their terminals do
     not move.
3. **TritonBench sample through the composed dispatcher** — DONE
   2026-07-11, the measured migration of the 37 indirect rows:
   7× proved@interp + 4× race@interp (11 decided), 10× host-side
   pid-divergent control flow (the interpreter's structural
   boundary), 7× per-instance loop bounds, 5× the snapshot cap (the
   new distinct reason), 3× masked load without `other`, 1×
   SymbolicExprDataWrapper coercion. Corpus-wide: unsupported
   76 → 55, +15 proved@interp, +6 race@interp (the composed
   terminals also rescued rows outside the indirect set). The 6
   race@interp-on-race-free rows are the audit's
   interp-disagreements bucket: descriptor-rebuilt randint index
   tables collide where the real workload's indices were unique —
   a RECONSTRUCTION-fidelity artifact, not detector unsoundness;
   the capture-side fix (record observed index uniqueness, rebuild
   unique tables via randperm sampling, needs a GPU re-capture) is
   queued in TODO §3d as the follow-up.
4. **RQ5 complementarity refresh** (`evaluation/ablation.py`): the
   `no-load-values` ablation must now also erase the address-position
   verdicts (both the trb010 confirmations and the trb013 plain-fetch
   detection), and the headline complementarity counts update.
5. **Bookkeeping**: TODO §3e's "7 rows" corrects to 8 (the results
   file carries 8 `data-dependent-bound` rows; `tb_block_sparse_attn`
   is the lone lower-bound case), and §3d's "36" reads "37 (36 arith
   + 1 direct)".

## 6. Verdict-attribute plumbing

The dynamic result gains the premise marker: when any recorded event's
address contains a snapshot select, the launch's dynamic verdict
carries `premises: ["contents-snapshot"]` (new field beside
status/reason in the harness dynamic dict, and folded into
`verdict_attrs.conditional` for the composed row). The ladder audit
needs one new compatibility rule: a `contents-snapshot` verdict is
launch-scoped evidence and must not be scored against any-params
claims — mirror of the existing `+assumes-termination` handling.

## 7. Non-goals (this spec)

- Static-track (TTIR) select-in-address at any tier — the
  `indirect-address` abstention remains the router. A T1-contents
  static tier over `GlobalTensor.init_values` is a candidate follow-up
  but changes the meaning of a static rung; do not fold it in here.
- Block-pointer (`tensor_pointer_load`) index sources.
- Atomic returns in addresses beyond the counting axiom (unchanged).
- `sort`/`cumsum`-derived addresses.
- Tables over the snapshot cap (policy abstention, §2/§3).
- Float index tensors (dtype guard unchanged).
- §3e items (snapshot-lifted loop bounds, nested loops, cf.cond_br) —
  they reuse pieces specified here (the loop-bound one inserts the
  same select shape into the iteration-existence premise
  `k ≥ 0 ∧ lower + k·step < upper`,
  `compiled/global_records.py:256-269`) but are separate hand-offs.

## 8. Suggested implementation order

1. Gate change + docstring fix (§1) with the trb010 scatter pair as
   the driving tests — the happy path should light up with no solver
   changes.
2. Keep the CAS site's pointer constraints explicitly (rd:1756 →
   match the RMW site's shape) — small, removes the cache-coincidence
   dependency before anything is built on it.
3. §4's six acceptance-test families (they mostly test EXISTING §2
   machinery from the new entry point).
4. Alpha-renaming unit test + witness concreteness test (§3).
5. Cap-reason reword (§2) and premise plumbing (§6).
6. trb013 plain-fetch flip + regression pins (§5.2).
7. TritonBench corpus rerun + bucket table + RQ5 refresh (§5.3-5.4),
   with `query_stats` cost numbers recorded alongside.
