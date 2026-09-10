# Route 2 in the CuTile reader: loaded values as snapshot Selects

Date: 2026-09-10. Closes 4 of the 8 remaining `tilebench_cutile`
abstentions and 5 of the 21 `tritonracebench_cutile` ones.

## The gap

The static frontend refused every kernel whose ADDRESS depends on a
loaded value. Route 2 gave the Triton track a source of values for those
addresses in September: an integer load with a modeled mask binds a
`Loaded` term, the encoder turns it into a Select over the tensor's
pre-launch contents, and the proof is CONTENT-QUALIFIED (this launch's
contents, any grid along the axes the kernel reads). The design,
correctness obligations and Triton-side evidence are
`evaluation/ROUTE2_SNAPSHOT_SELECT.md`; nothing in the model changes
here.

The cuTile track had neither half. Its reader bound `DataDep` for every
loaded value, and its captures carried no contents at all: the corpus
was captured before the value-capture rule landed, so all 166 tensor
descriptors were value-free. Six real-operator rows and five benchmark
rows abstained for that reason while their Triton twins were decided.

## What changed

**Reader** (`triton_viz/clients/common/cutile_ir_reader.py`).
`_loaded_binding` mirrors the TTIR reader's `loaded_binding` clause for
clause: bound only under `multipath` (the ladder's L2), so L0 and L1 keep
`DataDep` and every refusal message is byte-identical; a float pointee or
a DROPPED mask keeps `DataDep`, because only a modeled mask can keep a
masked-off lane, which holds `other` or an undefined value, apart from
the snapshot value. cuTile has no `other` operand, so it comes from the
source: a partition view's `padding_mode=ZERO` supplies `Const(0)`,
`UNDETERMINED` and `NEG_INF` leave the lane unspecified (a free pad
array, the widening direction), and `load_pointer` takes its
`padding_value` when that is a modelable term.

**Capture** (`evaluation/tilebench_cutile_capture.py`,
`evaluation/tritonracebench_cutile_capture.py`). Each tensor descriptor
now carries the address snapshot under the Triton track's own bound
(`CompiledRaceDetector.ADDRESS_SNAPSHOT_MAX_ELEMENTS`, 16384 elements),
taken at the same pre-launch moment as `init_values`, with
`snapshot_reason` recorded when there is none so a refusal can say why.

**Harness** (`evaluation/harness.py`). `_cutile_bindings` passes the
snapshot into `GlobalTensor`, gated on L2 exactly as
`CompiledRaceDetector` gates `_capture_snapshot`; below L2 the reason is
`"L2 only"` and the encoder's refusals are unchanged.

## The re-capture was value-only, and checked

Both corpora were re-captured on the pinned TileBench checkout
(224ec81f) with cuda.tile 1.5.0 on the RTX 4090, all 45 operators and all
69 benchmark rows, zero failures. The merge adds ONLY the new value
fields and refuses on anything else: the CuTile IR text and every
pre-existing descriptor, constexpr, grid and alias field must be
byte-identical, and no corpus row may appear or vanish. It reported 61
rows checked and 183 value fields added for `tilebench_cutile`, 69 rows
and 170 fields for `tritonracebench_cutile`, with zero problems. The one
field excluded from the identity check is the benchmark capture's
`launch`, which carries the smoke launch's wall time ("ok (0.113s)"); its
STATUS word is compared and the stored string is never rewritten.

## A claim-strength fix found on the way

The first build lost two proofs: `ctb_radix_sort___count_ones_in_block`
and its `__s1` fell from `proved@T0` to `proved@T1`. The cause was the
`param_pinned` flag introduced with the exact bitwise lowering
(`evaluation/CUTILE_BITWISE_ADDRESSING.md`). It was a WHOLE-GRAPH flag,
set by any bitwise rewrite that consumed a scalar param's captured value,
and `t0_linearity_gate` refuses a param-pinned graph. That was safe while
such rewrites only happened in address chains: a loaded operand was
`DataDep`, so the fold refused. With `Loaded` bound, radix sort's
`(key >> bit) & 1` now folds, and `bit` is a launch parameter. The value
is only COUNTED, never used to address, so the ANY-params claim was
still available and the flag threw it away.

The flag is now scoped to the claim. `_fold_bitwise` records the rewritten
TERM; `_pinned_claim` marks the graph `param_pinned` only when such a term
reaches a position the claim rests on: an address, a mask, a path or exit
predicate, an atomic operand, or a loop bound. A rewrite that feeds a
value the footprint does not depend on costs no scope. Structural
containment decides it, so an unrelated but identical term only
over-pins, which is the safe direction. The bitonic rows step 1 closed
still pin, and the T0 gate still refuses them.

## Measured effect

Both cuTile corpora were rerun at L2.

`tilebench_cutile` (61 real-operator configurations): 53 proofs and 8
abstentions become **57 proofs and 4 abstentions**. The four:

| row | new terminal |
|---|---|
| `ctb_block_sparse_attention__block_sparse_attention_cutile_kernel` | proved@T1+content |
| `ctb_cross_entropy___cross_entropy_kernel` | proved@T1+content |
| `ctb_destindex___copy_by_dest_kernel` | proved@T1-launch+content |
| `ctb_destindex___copy_by_dest_kernel__s1` | proved@T1-launch+content |

No other verdict and no proof rung moved.

`tritonracebench_cutile` (69 benchmark rows): 24 race-free / 24 race / 21
abstain become **26 / 27 / 16**. Five rows changed, and every one of them
now matches its ORACLE label:

| row | label | before | after |
|---|---|---|---|
| `trb006_dd_mask_dead_no` | race-free | abstain | proved@T1+content |
| `trb006_dd_mask_live_yes` | race | abstain | race |
| `trb010_gather_no` | race-free | abstain | proved@T0 |
| `trb010_scatter_yes` | race | abstain | race |
| `trb013_work_queue_plain_yes` | race | abstain | race |

`trb010_gather_no` proving at T0 is the rule the Route 2 record predicts:
read-only tensor groups are skipped at T0 by construction, so a kernel
whose loaded values steer only its reads proves for any input and any
contents.

Level invariance was measured, not assumed. The re-capture also filled in
`init_values`, which the corpora had lacked since they predate that
capture rule, and `init_values` is NOT gated on L2. Both corpora were
therefore rerun at L0 against the old and the new specs: 61 and 69 rows,
zero differences in verdict, terminal or reason.

The remaining four `tilebench_cutile` abstentions are one snapshot-bound
row (`histogram_partial`, a 262144-element index source against the
16384-element encoding bound), two `radix_sort` rows whose address goes
through `tile_scan`, and `streamk_matmul first_wave_kernel`, a genuinely
data-dependent walk.

## Negative controls

The scatter litmus, in cuTile IR, carries the burden that the snapshot
does not launder a race: over a PERMUTATION the kernel proves and the
encoding is content-qualified; with one index duplicated it reports
exactly one race. Without a snapshot the row refuses by name, and
single-path parsing reproduces the pre-Route-2 message verbatim. Four
more pin the abstention discipline (a float pointee, a dropped mask) and
the `other` rule (a view's padding mode, a pointer load's padding value),
and one pins the claim-scoped `param_pinned`: the same rewrite keeps T0
when it only feeds a counted value and closes T0 when it is the address.

## Adoption note

The pinned evaluation numbers are unchanged by this commit. A cuTile
population that includes these proofs, and a TritonRaceBench cuTile
population that includes the five settled rows, require a new pinned run.
