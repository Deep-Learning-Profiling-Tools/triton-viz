# cuTile token-order capture (2026-09-07)

## Scope and semantics

The user authorized this implementation after the final `31c48f5` experiment
sequence reached `measurements_complete_needs_analysis`. Development uses
isolated worktrees and the existing Python environment. Frozen captures,
measurement checkouts, and the original result directory are preserved.

The reader records transitive operation-pair token reachability. `make_token`
has no memory predecessor, memory operations preserve their incoming token
ancestors, and `join_tokens` merges those ancestors without ordering them
against one another. A masked-off intermediate memory access still transmits
its input token. Exact scalar branch predicates guard token selections;
unknown, lane-dependent, or unresolved token relations refuse explicitly.

Token edges are not translated into full fence cuts. The encoder carries
the relation through T0 tensor partitioning, T1, the content-free proof
attempt, and launch-scoped requery. An empty token map means no token order;
an absent map selects the Triton discipline. A cuTile graph with missing
metadata refuses instead of falling back to full source order.

In token mode the shared solver uses guarded reachability in both
happens-before construction and same-instance candidate queries. Neither
source order nor ordinary value dependence supplies extra memory-ordering
edges. Independent atomic operations may read from later source operations;
coherence respects token order and happens-before rather than imposing full
source order. Atomic-observation value causality remains a separate
well-formedness constraint. Triton missing-source-fence advice is not emitted
for token-mode reports.

The normative distinction is documented in NVIDIA's
[Tile IR memory model, Sections 7.5 and 7.12](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html):
token dependencies determine memory ordering even where source program
dependencies appear to imply an order; release/acquire payloads need the
appropriate incoming/outgoing token paths.

## Loop boundary

The existing same-instance query shares each symbolic iterator between its
two lane roles. The reader therefore checks each write-involving pair across
iterations, including each write's self-pair. Each temporal direction needs
a carried slot whose continuation collects the earlier access and whose
input reaches the later access. The two directions may use different slots.
Each carried slot must retain its prior ancestry, so an established edge
survives intervening iterations.

Unordered pairs become explicit `AccessGraph.loop_token_conflicts`
obligations. Before any solver path or tensor partition can omit them, the
encoder checks that their allocations cannot overlap. T1 uses verified
allocation byte intervals under the existing in-bounds premise. T0 uses its
explicit non-aliasing premise; the public client checks the actual allocation
intervals before accepting a T0 proof. Different formal names or different
view pointers alone do not establish separation. Overlapping or unavailable
intervals cause a named `token-order` abstention, including disjoint views of
one allocation that would need finer footprint reasoning. This check applies
to the L2 content-free proof path as well.

Read/read pairs have no conflict, and zero-trip or single-trip loops have
no pair of distinct iterations. Loop exits preserve actual output slots and
the zero-trip initial token. Reset/swapped ancestry and unsupported token
control structures still refuse. No allocation check introduces a token or
happens-before edge.

Await summaries additionally require token-serialized polls, preserving the
justification for the pre-exit representative. The actual break operands
determine whether a later operation is ordered after the exit poll. The
existing termination condition and closed-world source domain are unchanged.

## Additional proof-path correction

Inspection found that `_t1_content_free` did not pass any ordering metadata
to the solver. This early proof path now passes the same fence/token policy
as the other compiled paths. A regression with two same-instance stores,
one under a free loaded mask, verifies that neither frontend can obtain a
proof by accidentally using full source order on this path.

## Validation and measurement provenance

The integrated selection passes all 366 tests: the four cuTile reader suites,
token encoding and public-client tests, shared solver and HB construction,
compiled TTIR and Route 2/multipath regressions, and the previous diagnostic
and fence-capture suites. This includes the new unknown-token-operand refusal,
an await with a nonserial plain-load poll, T0 order across omitted tensor
groups, actual aliased formals at all three levels, and both token-connected
and token-disconnected release/acquire payloads. The initial capture-test
attempt encountered a read-only default compiler cache; isolated cache
directories resolved all nine environmental failures. All 366 then pass
together in 19.23 seconds. Repository hooks pass after formatting; an AST
comparison confirms that the formatter changed no Python semantics.

The first affected-corpus checks completed at source commit `6e1d3eb` on
2026-09-07 at 15:31:28 UTC, under exclusive host admission. Each configuration
used a fresh subprocess, seed 0, a 200-second outer cap, and two workers.
These are successor correctness checks, not paper performance measurements
or relabelings of `31c48f5` results.
The harness stamps cuTile receipts with `fence_order_applies: true` and
`intra_instance_order: token` when the fence-order configuration is enabled.
The existing full common-pin paper measurements remain tied to their
original detector pin until the submission rerun is adopted.

| Corpus | Level | Configurations | Race-free | Race | Abstain |
|---|---|---:|---:|---:|---:|
| TritonRaceBench cuTile | L0 | 62 | 12 | 12 | 38 |
| TritonRaceBench cuTile | L1 | 62 | 12 | 12 | 38 |
| TritonRaceBench cuTile | L2 | 62 | 20 | 23 | 19 |
| TileBench cuTile | L0 | 61 | 38 | 0 | 23 |
| TileBench cuTile | L1 | 61 | 38 | 0 | 23 |
| TileBench cuTile | L2 | 61 | 47 | 0 | 14 |

All 369 configurations completed without an error. All 214 decided results
match the existing labels; the benchmark's one unscored row abstains at
every level. Relative to the frozen `31c48f5` datasets, the only verdict
changes are `trb008_loop_stride_no` (race-free to abstain) and
`trb008_loop_stride_yes` (race to abstain), at all three levels. Their
mixed read/write loops fail that pin's stronger requirement for one boundary
covering every body access. Inspection of the actual captured IR corrects
the earlier "token-independent write iterations" description: stores already
consume a carried token and return their output on the backedge; loads use
the external root token. Their separate input/output allocations discharge
the remaining load/store obligations under the refined check above.
Four unchanged abstentions, the L0/L1 `trb018_lookback_no` and
`trb018_lookback_cta_yes` rows, now name `token-order` rather than
`control-flow`. All other verdicts are unchanged.

Raw receipts, both capture hashes, the run manifest, per-row comparison,
test output and hook output are retained under
`evaluation/results/cutile-token-order-6e1d3eb/` in the canonical detector
checkout. The same directory in the isolated worktree preserves the original
run location. `summary.json` records all six dataset hashes and baseline
hashes; `manifest.json` records original paths, commit, commands' policy and
timestamps. Independent verification confirms unique and matching names,
all six result hashes, both capture hashes, token-policy flags and labels.
The old datasets still match their frozen completion hashes.

Because the content-free proof-path repair also affects Triton L2, a final
common-pin paper adoption must cover the full required experiment set at
this or a successor implementation pin. The affected cuTile receipts do not
replace that submission requirement.

## Conflict-aware loop refinement (2026-09-07)

The user authorized recovery of the two loop-stride configurations by
requiring token ordering only for pairs that can conflict. The reader and
encoder implement the pairwise checks and allocation obligations described
above. The original captures and expected labels are preserved.

All 433 integrated regression tests pass in 19.97 seconds, including 41
dedicated allocation/loop cases. The selection covers the cuTile reader,
token encoding, shared solver, storage extents, compiled capture, Route 2,
multipath analysis and the previous fence/diagnostic regressions. New cases
exercise actual and shifted aliases, partially overlapping intervals,
adjacent allocations, both temporal directions, independent serial chains,
write self-pairs, T0 premises and runtime zero/single-trip loops. An
independent source review checks the proof paths and unchanged await rule.
Affected-corpus successor receipts are recorded below after completion.
