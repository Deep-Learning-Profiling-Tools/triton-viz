# The counted while-form loop in the CuTile reader

Date: 2026-09-10. Closes 4 of the 12 remaining `tilebench_cutile`
abstentions and moves a fifth onto the loaded-value queue.

## The gap

cuda.tile has no `for` construct in its final IR for a python `while`.
A kernel written as

```python
m_tile = start
while m_tile < n_tiles:
    ...
    m_tile += 1
```

lowers to the while-form `loop` construct: the counter is a loop-carried
value, the test is the first operation of the body, and the loop exits
through a `break` in the `else` arm of that test. The reader refused
every while-form loop that carried a non-token value
(`control-flow: while-form `loop` construct (carried values,
data-dependent trip) is not modeled`), because a carried value in general
means a data-dependent trip count. The counter shape is not that: it has
an ordinary `range` trip count, and the Triton twins of the same
operators, which spell it `for m_tile in range(...)`, were decided all
along.

## The shape that is lifted

`_counted_while_shape` in `triton_viz/clients/common/cutile_ir_reader.py`
matches, on the IR text and before any state is touched:

```
$r1: T1, ... = loop (with acc.0: Ta = $a0, i.0: Tile[int32,()] = $i0)
do (acc.0: Ta, i.0: Tile[int32,()])
    (acc.0: Ta, i.0: Tile[int32,()]):
    $c: Tile[bool_,()] = raw_cmp(lhs=i.0, rhs=<bound>, fn="lt")
    if(cond=$c)
    then
        ():
        yield
    else
        ():
        break acc.0, i.0
    ... body ...
    $in: Tile[int32,()] = raw_binary_arith(lhs=i.0, rhs=<K>, fn="add", ...)
    continue $an, $in
```

A match hands `_lift_counted_loop` the bounds `range($i0, <bound>, K)`.
That function is the former body of `_handle_for`, now shared by both
constructs, so a counted while and its `for` twin build the same
`AccessGraph`: one `LoopInfo` slot, the counter bound to `LoopVar`, other
carried values bound to `DataDep`, the same zero-trip rule, and the same
`_serial_loop_boundary` obligation on the body's accesses. A unit test
parses both spellings of one body and compares the graphs.

Every clause is required, and each one is a soundness obligation rather
than a convenience:

| clause | what it rules out |
|---|---|
| the test is the FIRST body operation | a do-while, whose body runs once more than the range |
| exactly one integer scalar carried slot is the counter | a trip count over several interacting values |
| the bound names no carried parameter | a bound that moves with the loop. The bound is read on the first body line, so by SSA dominance it is defined outside the loop |
| `then` is a bare `yield`, `else` is `break` | an inverted or result-carrying test |
| the `break` operands repeat the carried parameters in order | a result that is not the carried value at the top of the exiting iteration |
| no other `break` targets this loop, and the body's only terminator is its trailing `continue` | an early exit. This one matters most: the zero-trip rule DELETES in-loop accesses, so a loop that can also leave early would lose real accesses rather than gain widened ones |
| the counter's `continue` operand is `counter + K` for a positive integer constant `K`, defined at the top level of the body | a data-dependent or conditional advance |

A `break` inside a nested `if` region still exits the enclosing loop, so
the single-exit scan (`_nesting_scan`) treats only a nested `loop` /
`for` / combiner `do` block as shielding a terminator from this loop.

Anything that fails a clause falls through unchanged: first to the AWAIT
(spin) shape, then to the byte-identical control-flow refusal. Stream-K's
`first_wave_kernel` is the live example. Its outer iterator advances by
`min`, not by a constant, which is exactly the data-dependent walk the
refusal exists for, and it still abstains.

The lift is a reader capability, not a ladder rung: it applies at L0, L1
and L2 alike, and a unit test pins that both modes produce the same
graph. It does not set `assumes_termination`: a counter that advances by
a positive constant toward a fixed bound terminates.

## Measured effect

Every row of both cuTile corpora was rerun at L2 (130 rows). Exactly four
verdicts changed, all `abstain` to `race-free`:

| row | new terminal |
|---|---|
| `ctb_linear_self_attention___kv_kernel` | proved@T1 |
| `ctb_linear_self_attention___out_kernel` | proved@T1 |
| `ctb_linear_self_attention___z_kernel` | proved@T1 |
| `ctb_streamk_matmul__full_tiles_kernel` | proved@T1 |

`tilebench_cutile` goes from 49 proofs and 12 abstentions to 53 and 8.
All 69 `tritonracebench_cutile` rows are unchanged, verdict and reason
alike, so the AWAIT shape and the planted races are untouched. A
parse-level sweep over both corpora at both modes (260 outcomes) shows
exactly nine differences: the eight belonging to those four rows, plus
`ctb_block_sparse_attention__block_sparse_attention_cutile_kernel`, whose
refusal advances from the control-flow message to
`indirect-address: load tile index: data-dependent (loaded value)`. Its
loop is a counted walk; what still blocks it is the loaded block index,
which is Route 2 work.

The remaining eight `tilebench_cutile` abstentions are therefore seven
loaded-value addresses (`block_sparse_attention`, `cross_entropy`, both
`destindex` rows, `histogram_partial`, both `radix_sort` rows, the last
two of which additionally need `tile_scan`) and one genuine
data-dependent walk (`streamk_matmul first_wave_kernel`).

## Negative controls

Two unit tests carry the burden of "the lift does not launder a race":

* a counted while whose body writes tile `bid * N + i`, token-chained
  through a carried slot, proves race free;
* the same kernel with the block id dropped from the index, so every
  program writes tiles `0..N-1`, still reports the race.

Five more pin the refusals: a second exit, a runtime step, a bound taken
from a carried slot, permuted `break` operands, and a zero-trip bound
whose body contributes no footprint. A sixth pins that a body store with
no carried token inherits the `for` loop's cross-iteration obligation and
abstains with `token-order` rather than proving.

## Adoption note

The pinned evaluation numbers are unchanged by this commit. A cuTile
population that includes these four new proofs requires a new pinned run.

Superseded in part (2026-09-10): the seven loaded-value abstentions this
record leaves open were closed to three by
`evaluation/CUTILE_ROUTE2_SNAPSHOT.md`. The counts above stand as the
state at this commit.
