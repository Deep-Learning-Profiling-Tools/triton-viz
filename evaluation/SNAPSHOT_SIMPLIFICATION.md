# Exact simplification of snapshot-dependent static queries

Implemented after the user stopped the first L2 rerun and authorized
performance changes on 2026-09-05. The old detector pin is 637f57f.
The three slow-case profiles showed that actual solver checks, rather
than HB or Solver construction, dominated their static time.

## Encoding changes

`compiled/global_records.py` uses the existing eligible, read-only
integer snapshots to expose values before the pair queries reach Z3:

1. Constant indices become their captured integers. An arithmetic
   progression becomes a guarded affine expression. A non-affine table
   with at most 32 elements becomes a finite conditional expression.
   All other cases retain the original array expression.
2. A loop bound whose unmasked snapshot indices differ by constant
   offsets is evaluated at every index in their common admissible
   domain, provided that domain has at most 32 elements. It becomes a
   constant only if every case simplifies to the same integer.
3. If the resulting trip count is one, body induction expressions use
   the lower bound and loop-carried pointer offsets use their initial
   offset. The loop ordinal remains a registered symbolic variable,
   with its original existence premises and independent copy renaming.

For the KDA capture, `cu_seqlens = [0, 29, 64]` gives lengths 29 and
35. Both round up to one block of 64. This permits eliminating the
iteration ordinal from addresses without restricting the program grid.

## Equivalence conditions

The original snapshot equalities remain. Every rewritten lookup uses
the original `Select` outside the captured element range, including
uncaptured in-bounds suffixes of incomplete metadata. Therefore lookup
equivalence does not depend on a consumer asserting an in-bounds
premise. Masked `other` values and free padding remain unchanged.

Bound propagation requires a complete snapshot and retains every
original source-domain premise. Masked sources, missing snapshots,
written or alias-written sources, nested loaded indices, uncorrelated
indices, and content-free/T0 evaluation do not use this shortcut.
The finite cases cover the complete admissible index domain, not merely
the launch's pids. Extra infeasible cases can prevent simplification
but cannot remove executions. A varying bound stays symbolic.

The 32-element limit selects an expression representation or skips
an optimization. It never refuses a previously supported input,
reduces solver timeouts, narrows a grid, or changes a row budget.
Existing loaded loop-carried advances still refuse by the same rule.

## Validation and measurement

`tests/unit/test_snapshot_simplification.py` and
`tests/unit/test_snapshot_loop_simplification.py` check equivalence
against the original array encoding, incomplete snapshots, masks,
independent program copies, source eligibility, exact loop domains,
zero/one/multiple trips, and both race-free and colliding varlen stores.

The paper repository's diagnostic probe
`baselines/profile_fla_static.py` measures KDA, RWKV7, and ABC in serial
fresh processes at the old and optimized commits. It preserves the
120 s and 10 s solver budgets, static confirmation and differential
paths, seed, TTIR, captured values, and fence-order setting. Raw data,
direct old/new result comparisons, environment fingerprints, and the
regression outcome are recorded in that repository's results index E13.
These diagnostics do not replace the interrupted run or the definitive
single-commit rerun; no full-corpus speedup is inferred from them.
