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
two lane roles. For loops with potentially conflicting iterations, the
reader therefore requires a carried token that orders every body memory
operation before the next iteration, and it checks actual continuation
operands. Each carried slot must retain its prior ancestry. Token-independent
write iterations, reset/swapped ancestry, and unsupported token control
structures produce a named `token-order` abstention. Read-only loops have
no conflicting pair between their iterations; zero-trip and single-trip
loops require no cross-iteration ordering. Loop exits preserve actual output
slots and the zero-trip initial token.

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

The affected-corpus receipt index will be appended after the committed
implementation is checked. These are successor implementation checks,
not relabelings of `31c48f5` results.
The harness stamps cuTile receipts with `fence_order_applies: true` and
`intra_instance_order: token` when the fence-order configuration is enabled.
The existing full common-pin paper measurements remain tied to their
original detector pin until the submission rerun is adopted.
