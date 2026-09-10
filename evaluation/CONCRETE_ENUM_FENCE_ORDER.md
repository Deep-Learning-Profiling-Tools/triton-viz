# Concrete enumeration under fence order

Date: 2026-09-06. This correction follows the completed L0/L1/L2 and
optimization-ablation measurements at `454d0321c46e7353619e6b4de52017b3852893e4`.
The implementation is `triton_viz/clients/race_detector/concrete_enum.py`.

## Defect and correction

The old enumerator excluded distinct operations of the same instance from
its conflict search, assuming whole-operation program order even with
`race_detector_fence_order=True`. For one instance, `v = load(x); store(x,
7); store(out, v)` returned `ok` while the symbolic frontend reported WAR.
An intervening `debug_barrier()` makes the pair ordered in both frontends.
The public regression was observed failing on the old implementation before
the correction. The paper repository preserves the original probe and output
under `baselines/results/pinned-ladder-ablation-454d032/`.

The recorder now snapshots the configured order mode, records a per-instance
fence epoch, and retains original positions before filtering masked lanes or
coalescing addresses. It tracks exact same-position dependencies separately
from the conservative taint used to identify footprint-determining loads.
Only supported elementwise operations with unchanged shapes carry exact
position tags. A memory operation creates a new anchor; dependencies through
an intermediate memory operation are not flattened without active-position
evidence. A masked-off intermediate load therefore cannot order a later store
against an earlier load.

The analyzer checks same-instance cross-operation overlaps. An intervening
fence orders the pair. Within an epoch, a captured dependency orders only
matching original positions; a shifted alias can still produce WAR. A
possible dependency whose position relation is unavailable yields the named
`dependency-order` abstention. Compatible same-address, same-width atomics
remain exempt. The sweep groups intervals by instance and epoch, separates
active read and write heaps, and caches each dependent operation pair's
position comparison.

Footprint determinism also needs the corrected order. A load feeding an
address, mask, branch or loop bound cannot rely on the replay's incidental
ordering of same-instance writes. The analyzer returns `value-source-order`
if a writer is unordered with that load, if earlier overlapping writers are
unordered with each other, or if a preceding store contains conflicting
duplicate positions. It applies these checks recursively to memory-relayed
sources. A fence after two unordered writers alone does not select a stable
winner. The existing cross-instance value-source and atomic-return refusals
remain in force.

`fence_order=False` explicitly retains legacy program order. In fence mode,
insufficient positional provenance produces a named abstention; the fix does
not add a general proof of arbitrary dependency paths. Shapes or widths that
cannot be compared precisely may therefore abstain. More abstentions and
additional recording cost are possible and require new measurements.

## Verification

`tests/end_to_end/test_concrete_enum_fence_order.py` exercises public CPU
interpreter launches on cloned tensors: RAW/WAR/WAW compared with the
symbolic frontend; fences before, between and after accesses; explicit legacy
mode; direct, shifted, masked and cast dependencies; reduction and transpose
boundaries; inactive intermediate anchors; earlier/later source writers;
competing and duplicate writers; and unstable memory relay. Existing
synthetic tests now provide fence/position metadata. Tests specifically about
legacy program order opt out explicitly. Valid relay and projection controls
use intervening fences.

Validation commands use the existing Python 3.12 environment with Triton 3.6,
the corrected checkout on `PYTHONPATH`, and an isolated Triton cache:

```sh
python -m pytest -q tests/unit/test_concrete_enum_analysis.py tests/end_to_end/test_concrete_enum.py tests/end_to_end/test_concrete_enum_fence_order.py tests/end_to_end/test_fence_order.py
python -m pytest -q tests/unit tests/end_to_end
```

The focused run passed 123 tests. The final unit/end-to-end suite passed
1,458 tests and skipped 29; one rehearsal test could not write the default
host lock directory under the filesystem sandbox. That test passed separately
with `TRITON_VIZ_PINNED_STATE_DIR` pointing to an isolated writable test
directory, completing 1,459 passing tests. All applicable pre-commit hooks,
including Ruff and mypy, pass. Exact logs and source hashes accompany the
paper's recovery record.

## Measurement boundary

This correction changes only the concrete-enumeration backend. L0 never
enters that backend; L1 and L2 can change verdicts, refusal reasons and costs.
No TTIR reader, shared solver, ladder gate, capture, budget, or measurement
protocol changes are included. The FLA `tt.dot` accumulator-dependency
question identified during ablation review remains a separate audit item;
this correction does not establish a real race in that kernel.

Do not replace records inside the completed `454d032` run or its receipts.
The 401 distinct old enum-proof candidates remain a review population.
Technical rerun coverage is 477 L1 and 144 L2 level/case configurations
(613 observed enum calls plus eight outer-timeout configurations where entry
cannot be ruled out), plus 12 paired ablation samples covering the affected
groups. These are proposed rerun populations, not measurements performed by
this correction. Final publication still needs a fresh common detector pin
and the full three-level measurement sequence with its ablation.
