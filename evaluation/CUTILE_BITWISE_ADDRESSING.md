# Exact lowering of integer bitwise addressing in the CuTile reader

Date: 2026-09-10. Closes 2 of the 14 `tilebench_cutile` abstentions.

## The gap

cuda.tile emits integer `xor`, shift and mask operations directly into an
address. The bitonic network computes its partner index as `i ^ j` and the
radix network shifts a key by `bit`; the Triton twins reach the same
addresses through `floordiv`/`mod` arithmetic, which the reader's affine
fragment already accepts. The reader therefore refused the cuTile rows with
`indirect-address: pointer offset: data-dependent (bitwise fn xor)` while
their Triton twins were decided. The same corpus contains the control:
`ctb_top_k_selection___bitonic_step_kernel` is a bitonic step whose index is
written with `floordiv`/`c_mod`, and it proved all along.

## The lowering

`_fold_bitwise` in `triton_viz/clients/common/cutile_ir_reader.py` rewrites
an integer bitwise operation into the existing fragment when, and only when,
the second operand is a known integer of a shape whose arithmetic identity
is exact:

| IR | condition | lowering |
|---|---|---|
| `a rshift c` | `0 <= c < 63` | `floor(a / 2^c)` |
| `a lshift c` | `0 <= c < 63` | `a * 2^c` |
| `a and_ c` | `c = 2^k - 1` | `a mod 2^k` |
| `a xor c` | `c = 2^k` | `a + c - 2c * ((a / c) mod 2)` |

`Bin("//")` and `Bin("%")` are C-style truncation toward zero, so floor
division and floor modulo are written out rather than assumed:
`floor_mod(a, n) = ((a % n) + n) % n` and
`floor_div(a, n) = (a - floor_mod(a, n)) / n`, which agree with the
truncating operators because the numerator is a multiple of `n`. The
identities are exact for EVERY integer, negative included; a unit test
checks each one against Python's own operators on every lane, and an
offline check covered 192,240 values including large and negative ones.
Anything outside the table keeps its `DataDep` and the reader's refusal.
Nothing here widens a footprint.

## Why such a proof is T1, not T0

A shift count or mask is often a scalar kernel argument (`j`, `bit`), known
only from the captured launch. The reader now receives those values
(`parse_cutile_ir(..., params=...)`, bound by `evaluation/harness.py` before
the parse) and marks a graph that consumed one `param_pinned`. The tier
selector's `t0_linearity_gate` refuses a param-pinned graph outright, so the
ANY-params claim is never made from a rewrite that is exact for one launch's
parameters. A lowering from a literal constant pins nothing and leaves T0
available. Without captured params, for instance at T0, the refusal stands
exactly as before.

## A soundness fix found on the way

`raw_binary_bitwise` bound `and_`/`or_` to a BOOLEAN conjunction regardless
of the result type, so an INTEGER `and_` in an address became a modeled
boolean term instead of an abstention. The branch is now guarded on a
`Tile[bool_` result; an integer `and_`/`or_` that the table above cannot
lower abstains. No row in either cuTile corpus changed verdict from this
fix, so no published number rested on it, but the hole was real.

## Measured effect

Every row of both cuTile corpora was rerun at L2 (130 rows). Exactly two
verdicts changed, both `abstain` to `race-free (proved@T1)`:
`ctb_bitonic_sort___bitonic_step_kernel` and its `__s1` specialization. No
other verdict and no proof rung moved, and a Triton-track sample is
unchanged. `tilebench_cutile` goes from 47 proofs and 14 abstentions to 49
and 12.

The other 12 abstentions are untouched. Six are loaded-value addressing
(`cross_entropy`, `destindex` and its specialization, `histogram_partial`,
and the two `radix_sort` rows, which additionally need `tile_scan`); their
Triton twins are decided either statically through snapshot Selects or by
the enum rung. Six are the while-form loop with carried values. Both need
their own work.

Adoption note: the pinned evaluation numbers are unchanged by this commit.
A cuTile population that includes the two new proofs requires a new pinned
run.

Superseded in part (2026-09-10): the six while-form loop abstentions this
record leaves open were revisited in
`evaluation/CUTILE_COUNTED_WHILE_LOOP.md`, which closes four of them and
moves a fifth into the loaded-value group. The loaded-value group was
then closed by `evaluation/CUTILE_ROUTE2_SNAPSHOT.md`, which also
narrowed `param_pinned` from a whole-graph flag to the terms the CLAIM
rests on: a bitwise rewrite that feeds a value the footprint does not
depend on no longer closes the ANY-params tier. The counts above stand as
the state at this commit.
