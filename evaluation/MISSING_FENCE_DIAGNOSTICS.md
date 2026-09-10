# Missing source-fence diagnostics (2026-09-07)

This change explains an already established global-memory race report.
It does not change fence capture, cuTile token handling, happens-before
constraints, conflict queries, replay, fallback routing, or the verdict
taxonomy. The implementation was prepared in an isolated worktree while
the `31c48f5` experiments continued in their frozen checkout.

## Eligibility and wording

The symbolic solver and concrete enumerator share a pure report formatter.
It appends an explanation only when fence order applies, both witness
accesses belong to the same program instance, the operations are distinct,
their source locations and distinct nonnegative sequence numbers are known,
and no captured fence separates them. Synthetic pre-exit representatives
are excluded. The original reason prefix is preserved.

The explanation names both source accesses, describes the absent captured
tile-level fence relative to the memory model, and gives conditional Triton
barrier guidance. It does not infer the compiled kernel's implicit barriers
or promise an automatic repair. A dependency between operations does not
exclude a diagnostic for their conflicting *different* positions.

Duplicate positions in one operation, cross-instance conflicts, legacy
ordering, absent/malformed source locations and unknown/equal sequence
numbers keep their existing reasons. Unlocated operations are not assigned
invented source lines.

## Output and compatibility

`RaceReport.reason` and `ConcreteRaceReport.reason` carry the explanation.
The evaluation harness additionally saves this field in each static,
interpreter and enumeration witness. Launch/stage-level `reason` fields
retain their previous refusal classifications. No public report field is
removed or renamed.

Only source coordinates and access modes are formatted into the explanation.
The numeric witness address and byte range remain in their existing fields,
so clone-to-original address translation cannot leave a stale address in
the diagnostic text. Text uses source order even when report canonicalization
orders its endpoints differently; endpoint order and RAW/WAR/WAW labels
remain unchanged.

Comparators that compare complete witness dictionaries will observe the
new explanatory field. To check semantic equivalence, omit only each
report's `reason`, retaining outcome-level refusal reasons, proof extents,
all endpoints, numerical witness assignments and report multiplicities.
Historical experimental artifacts are retained byte-for-byte and are not
relabeled as measurements of the diagnostic-enhanced source.

## Verification

Regression cases cover RAW/WAR/WAW with fences before, between, after or
absent; excluded report classes; reversed canonical endpoint order;
cross-position dependencies; enumeration address translation; and all
three witness serialization paths. Paired checks suppress only the new
formatter and compare every remaining verdict/witness field.
The source audit also removes only the annotation blocks/imports and
compares the solver and enumerator ASTs with `31c48f5`: both are identical.
Validation completed after the formal experiment sequence reported completion
at 14:39:58 UTC. The test process acquired the existing host admission lock
exclusively before importing the isolated checkout at 14:40:12 UTC.
The following selection passed all 169 tests in 6.04 seconds:

```sh
python -m pytest -q \
  tests/unit/test_two_copy_symbolic_hb_solver.py \
  tests/unit/test_concrete_enum_analysis.py \
  tests/unit/test_diagnostic_export.py \
  tests/unit/test_verdict_attributes.py \
  tests/end_to_end/test_fence_order.py \
  tests/end_to_end/test_concrete_enum_fence_order.py
```

These are correctness regressions, not new performance measurements. The
paper text also compiled successfully. No experiment was rerun or rewritten.
`git diff --check` and parsing of every changed Python source pass. The local
pre-commit hook points to an absent virtualenv and `pre-commit` is unavailable;
the commit therefore uses a command-local hook override, not a repository
configuration or shared-environment change.
